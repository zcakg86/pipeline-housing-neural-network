#!/usr/bin/env python3
"""Build a compact H3 historical-market and OSM place-label surface for Java."""
from __future__ import annotations

import argparse
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

import h3
import numpy as np
import osmium


DEFAULT_SNAPSHOT = Path("data/local_market_snapshot.json")
DEFAULT_COMMUNITIES = Path("data/community_map.json")
DEFAULT_OSM = Path("data/osm/washington-latest.osm.pbf")
DEFAULT_OUTPUT = Path("data/h3_market_surface.json")
DEPLOY_OUTPUTS = (
    Path("java-app/house-price-app/src/main/resources/model-artifacts/h3_market_surface.json"),
    Path("java-app/model-artifacts/h3_market_surface.json"),
)
CITY_TYPES = {"city", "town"}
NEIGHBORHOOD_TYPES = {"suburb", "quarter", "neighbourhood"}
MAX_CITY_DISTANCE_KM = 25.0
MAX_NEIGHBORHOOD_DISTANCE_KM = 6.0


class PlaceCatalogue(osmium.SimpleHandler):
    def __init__(self, bounds):
        super().__init__()
        self.bounds = bounds
        self.places = []

    def node(self, node):
        place_type = node.tags.get("place")
        name = node.tags.get("name")
        if place_type not in CITY_TYPES | NEIGHBORHOOD_TYPES or not name:
            return
        if not node.location.valid():
            return
        longitude, latitude = node.location.lon, node.location.lat
        west, south, east, north = self.bounds
        if west <= longitude <= east and south <= latitude <= north:
            self.places.append({
                "name": name,
                "type": place_type,
                "lat": latitude,
                "lng": longitude,
                "osmId": str(node.id),
            })


def nearest(latitude, longitude, places, maximum_km):
    if not places:
        return None
    latitudes = np.asarray([place["lat"] for place in places], dtype=np.float64)
    longitudes = np.asarray([place["lng"] for place in places], dtype=np.float64)
    latitude_radians = math.radians(latitude)
    delta_latitude = np.radians(latitudes - latitude)
    delta_longitude = np.radians(longitudes - longitude)
    values = np.sin(delta_latitude / 2.0) ** 2 + math.cos(latitude_radians) * np.cos(
        np.radians(latitudes)
    ) * np.sin(delta_longitude / 2.0) ** 2
    distances = 6371.0088 * 2.0 * np.arctan2(np.sqrt(values), np.sqrt(1.0 - values))
    index = int(np.argmin(distances))
    distance = float(distances[index])
    if distance > maximum_km:
        return None
    place = places[index]
    return {
        "name": place["name"],
        "placeType": place["type"],
        "distanceKm": round(distance, 3),
        "osmId": place["osmId"],
        "assignment": "nearest named OSM place node",
    }


def historical_surface(record, global_record, shrinkage_weight):
    if not record:
        return None
    support = max(0.0, math.expm1(float(record.get("log1p_decayed_sales_count", 0.0))))
    raw_premium = float(record.get("raw_decayed_log_price_premium", 0.0))
    premium = raw_premium * support / (support + shrinkage_weight) if support else 0.0
    dispersion = max(0.0, float(record.get("decayed_log_price_std", 0.0)))
    global_mean = float(global_record.get("decayed_mean_log_price", math.log(300_000.0)))
    return {
        "decayedSaleSupport": round(support, 3),
        "historicalTypicalSalePriceAllProperties": round(math.exp(global_mean + premium)),
        "pricePremiumPctVsRegionalBaseline": round(100.0 * math.expm1(premium), 2),
        "multiplicativePriceDispersionPct": round(100.0 * math.expm1(dispersion), 2),
        "lastSaleDate": record.get("last_sale_date"),
    }


def build(snapshot_path, community_path, osm_path):
    snapshot = json.loads(snapshot_path.read_text())
    communities = json.loads(community_path.read_text())
    snapshot_cells = snapshot.get("cells", {})
    base_cells = set(snapshot_cells) | set(communities)
    # Cover a small halo around observed/modelled cells so live listings just
    # outside historical coverage still receive place labels and neighboring
    # surface context without fabricating historical statistics.
    cell_ids = sorted({
        neighbor
        for cell in base_cells
        for neighbor in h3.grid_disk(cell, 2)
    })
    centers = {cell: h3.cell_to_latlng(cell) for cell in cell_ids}
    latitudes = [center[0] for center in centers.values()]
    longitudes = [center[1] for center in centers.values()]
    bounds = (
        min(longitudes) - 0.1, min(latitudes) - 0.1,
        max(longitudes) + 0.1, max(latitudes) + 0.1,
    )
    places = PlaceCatalogue(bounds)
    places.apply_file(str(osm_path))
    cities = [place for place in places.places if place["type"] in CITY_TYPES]
    neighborhoods = [
        place for place in places.places if place["type"] in NEIGHBORHOOD_TYPES
    ]
    shrinkage = float(snapshot.get("premium_shrinkage_weight", 3.0))
    global_record = snapshot.get("global", {})
    cells = {}
    for cell in cell_ids:
        latitude, longitude = centers[cell]
        cells[cell] = {
            "community": communities.get(cell),
            "city": nearest(latitude, longitude, cities, MAX_CITY_DISTANCE_KM),
            "neighborhood": nearest(
                latitude, longitude, neighborhoods, MAX_NEIGHBORHOOD_DISTANCE_KM
            ),
            "historicalMarket": historical_surface(
                snapshot_cells.get(cell), global_record, shrinkage
            ),
        }
    return {
        "schemaVersion": 1,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "h3Resolution": 8,
        "asOfDate": snapshot.get("as_of_date"),
        "latestHistoricalSaleDate": snapshot.get("latest_sale_date"),
        "marketSource": snapshot_path.name,
        "placeSource": osm_path.name,
        "placeAssignment": {
            "method": "nearest named OSM place node to H3 center",
            "cityTypes": sorted(CITY_TYPES),
            "neighborhoodTypes": sorted(NEIGHBORHOOD_TYPES),
            "maxCityDistanceKm": MAX_CITY_DISTANCE_KM,
            "maxNeighborhoodDistanceKm": MAX_NEIGHBORHOOD_DISTANCE_KM,
        },
        "cellCount": len(cells),
        "coverage": "historical/model cells plus an H3 ring-2 label halo",
        "placeCounts": {
            "citiesAndTowns": len(cities),
            "neighborhoods": len(neighborhoods),
        },
        "cells": cells,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--communities", type=Path, default=DEFAULT_COMMUNITIES)
    parser.add_argument("--osm", type=Path, default=DEFAULT_OSM)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-deploy", action="store_true")
    args = parser.parse_args(argv)
    for source in (args.snapshot, args.communities, args.osm):
        if not source.is_file():
            raise FileNotFoundError(source)
    payload = build(args.snapshot, args.communities, args.osm)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    if not args.no_deploy:
        for destination in DEPLOY_OUTPUTS:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(args.output, destination)
    print(
        f"Wrote {payload['cellCount']:,} H3 market-surface cells to {args.output}; "
        f"OSM labels: {payload['placeCounts']}"
    )


if __name__ == "__main__":
    main()
