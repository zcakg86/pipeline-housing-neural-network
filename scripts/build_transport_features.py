#!/usr/bin/env python3
"""Build static H3 transport-accessibility features from OSM map extracts.

The output is deliberately separate from the price-model feature contract.  It
is a versioned spatial artifact for map exploration today and for a future
static-feature GNN experiment; neither deployed price model consumes it yet.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from shapely import STRtree, points
from shapely.geometry import Point, shape
from shapely.ops import transform


DEFAULT_GRID = Path(
    "java-app/house-price-app/src/main/resources/model-artifacts/"
    "synthetic_h3_l8_grid.json"
)
DEFAULT_RAIL = Path("data/osm/king_county_rail.geojson")
DEFAULT_BUS = Path("data/osm/king_county_bus.geojson")
DEFAULT_ROADS = Path("data/osm/king_county_major_roads.geojson")
DEFAULT_MOTORWAY_CONNECTORS = Path("data/osm/king_county_motorway_connectors.geojson")
DEFAULT_OUTPUT = Path("data/osm/h3_l8_transport_features.json")
DEFAULT_DEPLOY_OUTPUT = Path(
    "java-app/house-price-app/src/main/resources/model-artifacts/"
    "h3_l8_transport_features.json"
)
DEFAULT_BUNDLE_OUTPUT = Path("java-app/model-artifacts/h3_l8_transport_features.json")

EARTH_RADIUS_METERS = 6_371_008.8
REFERENCE_LATITUDE_DEGREES = 47.4
LONGITUDE_SCALE = EARTH_RADIUS_METERS * math.cos(
    math.radians(REFERENCE_LATITUDE_DEGREES)
) * math.pi / 180.0
LATITUDE_SCALE = EARTH_RADIUS_METERS * math.pi / 180.0

BUS_RADIUS_METERS = 800.0
ROAD_DENSITY_RADIUS_METERS = 1_000.0
RAIL_DECAY_METERS = 1_000.0
MOTORWAY_CONNECTOR_DECAY_METERS = 500.0
MAJOR_ROAD_DECAY_METERS = 500.0

FEATURE_NAMES = (
    "light_rail_proximity",
    "bus_stop_density_per_sq_km",
    "nearby_bus_route_count",
    "bus_centrality_score",
    "motorway_connector_proximity",
    "major_road_accessibility",
    "major_road_density_km_per_sq_km",
)


def _project(longitude, latitude, altitude=None):
    return (
        np.asarray(longitude) * LONGITUDE_SCALE,
        np.asarray(latitude) * LATITUDE_SCALE,
    )


def _project_geometry(geometry):
    return transform(_project, geometry)


def _features(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    return payload.get("features", [])


def _usable_geometries(
    features: Iterable[dict[str, Any]], predicate
) -> list[tuple[Any, dict[str, Any]]]:
    result = []
    for feature in features:
        properties = feature.get("properties") or {}
        if not predicate(properties):
            continue
        geometry = feature.get("geometry")
        if not geometry:
            continue
        value = _project_geometry(shape(geometry))
        if not value.is_empty:
            result.append((value, properties))
    return result


def _nearest_distances(query_points, geometries: list[Any]) -> np.ndarray:
    """Return nearest distances in metres, or infinity if the source is empty."""
    if not geometries:
        return np.full(len(query_points), np.inf, dtype=np.float64)
    _, distance = STRtree(geometries).query_nearest(
        query_points, all_matches=False, return_distance=True
    )
    return np.asarray(distance, dtype=np.float64)


def _nearby_indices(tree: STRtree, query_points, radius: float) -> list[np.ndarray]:
    """Return source geometry indexes inside ``radius`` for every query point."""
    pairs = tree.query(query_points, predicate="dwithin", distance=radius)
    grouped: list[list[int]] = [[] for _ in range(len(query_points))]
    if pairs.size:
        for query_index, source_index in pairs.T:
            grouped[int(query_index)].append(int(source_index))
    return [np.asarray(indexes, dtype=np.int64) for indexes in grouped]


def _route_keys(properties: dict[str, Any]) -> set[str]:
    """Stable identifiers for route labels recorded on a bus stop."""
    return {
        str(route).strip().casefold()
        for route in properties.get("routes", [])
        if str(route).strip()
    }


def build_features(
    grid_path: Path,
    rail_path: Path,
    bus_path: Path,
    roads_path: Path,
    motorway_connectors_path: Path,
) -> dict[str, Any]:
    """Calculate the static transport fields for every synthetic H3 level-8 cell."""
    grid = json.loads(grid_path.read_text())
    cells = grid.get("cells", [])
    if not cells:
        raise ValueError(f"Synthetic grid contains no cells: {grid_path}")
    for source in (rail_path, bus_path, roads_path, motorway_connectors_path):
        if not source.exists():
            raise FileNotFoundError(f"OSM transport artifact not found: {source}")

    rail_stations = _usable_geometries(
        _features(rail_path),
        lambda props: props.get("kind") == "station"
        and props.get("category") == "light_rail",
    )
    bus_stops = _usable_geometries(
        _features(bus_path), lambda props: props.get("kind") == "station",
    )
    road_segments = _usable_geometries(
        _features(roads_path), lambda props: props.get("kind") == "road",
    )
    motorway_connectors = _usable_geometries(
        _features(motorway_connectors_path),
        lambda props: props.get("kind") == "motorway_connector",
    )

    longitude = np.asarray([cell["lng"] for cell in cells], dtype=np.float64)
    latitude = np.asarray([cell["lat"] for cell in cells], dtype=np.float64)
    x, y = _project(longitude, latitude)
    cell_points = list(points(x, y))

    rail_distances = _nearest_distances(
        cell_points, [geometry for geometry, _ in rail_stations]
    )
    motorway_distances = _nearest_distances(
        cell_points, [geometry for geometry, _ in motorway_connectors]
    )
    road_geometries = [geometry for geometry, _ in road_segments]
    road_distances = _nearest_distances(cell_points, road_geometries)

    bus_tree = STRtree([geometry for geometry, _ in bus_stops])
    nearby_stops = _nearby_indices(bus_tree, cell_points, BUS_RADIUS_METERS)
    road_tree = STRtree(road_geometries)
    nearby_roads = _nearby_indices(road_tree, cell_points, ROAD_DENSITY_RADIUS_METERS)
    density_area_sq_km = math.pi * (ROAD_DENSITY_RADIUS_METERS / 1_000.0) ** 2
    bus_area_sq_km = math.pi * (BUS_RADIUS_METERS / 1_000.0) ** 2

    raw_density = np.empty(len(cells), dtype=np.float64)
    raw_routes = np.empty(len(cells), dtype=np.float64)
    road_density = np.empty(len(cells), dtype=np.float64)
    for index, point in enumerate(cell_points):
        stop_indexes = nearby_stops[index]
        raw_density[index] = len(stop_indexes) / bus_area_sq_km
        route_ids: set[str] = set()
        for stop_index in stop_indexes:
            route_ids.update(_route_keys(bus_stops[int(stop_index)][1]))
        raw_routes[index] = len(route_ids)

        # Clipping candidates to the circular catchment produces road length,
        # rather than simply a count of split OSM way segments.
        catchment = point.buffer(ROAD_DENSITY_RADIUS_METERS)
        local_length = sum(
            road_geometries[int(road_index)].intersection(catchment).length
            for road_index in nearby_roads[index]
        )
        road_density[index] = local_length / 1_000.0 / density_area_sq_km

    # The score intentionally combines supply (stops / km²) and network choice
    # (distinct stop route refs) on log scales. p95 normalization maps normal
    # high-accessibility cells close to 1 while preserving a bounded [0, 1]
    # field for a later neural/GNN input.
    bus_raw_score = np.log1p(raw_density) + np.log1p(raw_routes)
    normalizer = max(float(np.percentile(bus_raw_score, 95)), 1e-9)
    bus_centrality = np.clip(bus_raw_score / normalizer, 0.0, 1.0)

    def proximity(distances: np.ndarray, decay: float) -> np.ndarray:
        return np.where(np.isfinite(distances), np.exp(-distances / decay), 0.0)

    records: dict[str, dict[str, float]] = {}
    for index, cell in enumerate(cells):
        records[cell["h3_l8"]] = {
            "light_rail_proximity": round(float(proximity(rail_distances, RAIL_DECAY_METERS)[index]), 7),
            "bus_stop_density_per_sq_km": round(float(raw_density[index]), 5),
            "nearby_bus_route_count": int(raw_routes[index]),
            "bus_centrality_score": round(float(bus_centrality[index]), 7),
            "motorway_connector_proximity": round(float(proximity(motorway_distances, MOTORWAY_CONNECTOR_DECAY_METERS)[index]), 7),
            "major_road_accessibility": round(float(proximity(road_distances, MAJOR_ROAD_DECAY_METERS)[index]), 7),
            "major_road_density_km_per_sq_km": round(float(road_density[index]), 5),
        }

    return {
        "schema_version": 1,
        "h3_resolution": 8,
        "feature_names": list(FEATURE_NAMES),
        "parameters": {
            "light_rail_decay_m": RAIL_DECAY_METERS,
            "bus_catchment_radius_m": BUS_RADIUS_METERS,
            "motorway_connector_decay_m": MOTORWAY_CONNECTOR_DECAY_METERS,
            "major_road_decay_m": MAJOR_ROAD_DECAY_METERS,
            "major_road_density_radius_m": ROAD_DENSITY_RADIUS_METERS,
            "bus_centrality": "min(1, (log1p(stop_density_per_sq_km) + log1p(distinct_route_count)) / p95)"
        },
        "definitions": {
            "light_rail_proximity": "exp(-straight_line_distance_to_OSM_light_rail_station_m / 1000)",
            "bus_stop_density_per_sq_km": "bus and trolleybus stops within 800 m, divided by circular catchment area",
            "nearby_bus_route_count": "distinct route_ref values recorded on bus stops within 800 m",
            "bus_centrality_score": "bounded combination of local bus-stop density and distinct nearby route count",
            "motorway_connector_proximity": "exp(-straight_line_distance_to_the_endpoint_where_an_OSM_motorway_link_touches_a_non_motorway_road_m / 500)",
            "major_road_accessibility": "exp(-straight_line_distance_to_nearest_motorway_trunk_primary_or_secondary_m / 500)",
            "major_road_density_km_per_sq_km": "clipped length of motorway, trunk, primary and secondary roads within 1 km, per km²",
        },
        "source": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "grid": grid_path.name,
            "rail": rail_path.name,
            "bus": bus_path.name,
            "roads": roads_path.name,
            "motorway_connectors": motorway_connectors_path.name,
            "light_rail_station_count": len(rail_stations),
            "bus_stop_count": len(bus_stops),
            "motorway_connector_count": len(motorway_connectors),
            "major_road_segment_count": len(road_segments),
        },
        "cell_count": len(records),
        "cells": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, default=DEFAULT_GRID)
    parser.add_argument("--rail", type=Path, default=DEFAULT_RAIL)
    parser.add_argument("--bus", type=Path, default=DEFAULT_BUS)
    parser.add_argument("--roads", type=Path, default=DEFAULT_ROADS)
    parser.add_argument("--motorway-connectors", type=Path, default=DEFAULT_MOTORWAY_CONNECTORS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--deploy-output", type=Path, default=DEFAULT_DEPLOY_OUTPUT)
    parser.add_argument("--bundle-output", type=Path, default=DEFAULT_BUNDLE_OUTPUT)
    args = parser.parse_args()

    payload = build_features(
        args.grid, args.rail, args.bus, args.roads, args.motorway_connectors
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, separators=(",", ":")))
    for target in (args.deploy_output, args.bundle_output):
        if target:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(args.output, target)
    print(
        f"Wrote {payload['cell_count']:,} H3 transport feature records to {args.output} "
        f"({payload['source']['light_rail_station_count']} light-rail stations, "
        f"{payload['source']['bus_stop_count']:,} bus stops)."
    )


if __name__ == "__main__":
    main()
