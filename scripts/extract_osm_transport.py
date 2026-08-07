#!/usr/bin/env python3
"""Extract compact King County public-transport and major-road map layers."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    from scripts.extract_osm_water import geometry_bounds, run_ogr2ogr
except ModuleNotFoundError:  # Direct execution adds scripts/, rather than the repo, to sys.path.
    from extract_osm_water import geometry_bounds, run_ogr2ogr


DEFAULT_INPUT = Path("data/osm/washington-latest.osm.pbf")
DEFAULT_OUTPUT_DIR = Path("data/osm")
DEFAULT_DEPLOY_DIR = Path(
    "java-app/house-price-app/src/main/resources/META-INF/resources/data"
)

RAIL_STATION_TYPES = {"station", "halt", "tram_stop"}
RAIL_ROUTE_TYPES = {"light_rail", "tram", "monorail", "train"}
BUS_ROUTE_TYPES = {"bus", "trolleybus"}
MAJOR_ROAD_TYPES = {
    "motorway", "motorway_link", "trunk", "trunk_link",
    "primary", "primary_link", "secondary", "secondary_link",
}


def text(properties: dict[str, Any], name: str) -> str:
    value = properties.get(name)
    return "" if value is None else str(value)


def yes(properties: dict[str, Any], name: str) -> bool:
    return text(properties, name).lower() in {"yes", "true", "1"}


def route_list(value: str) -> list[str]:
    return [item.strip() for item in value.replace(",", ";").split(";") if item.strip()]


def classify_rail_station(properties: dict[str, Any]) -> str:
    station = text(properties, "station").lower()
    railway = text(properties, "railway").lower()
    if yes(properties, "monorail") or station == "monorail":
        return "monorail"
    if yes(properties, "light_rail") or station == "light_rail":
        return "light_rail"
    if yes(properties, "tram") or station == "tram" or railway == "tram_stop":
        return "tram"
    if yes(properties, "subway") or station == "subway":
        return "subway"
    if yes(properties, "train") or railway in {"station", "halt"}:
        return "train"
    return "rail_station"


def coordinate_pairs(value: Any):
    if (
        isinstance(value, list)
        and len(value) >= 2
        and isinstance(value[0], (int, float))
        and isinstance(value[1], (int, float))
    ):
        yield float(value[0]), float(value[1])
        return
    if isinstance(value, list):
        for child in value:
            yield from coordinate_pairs(child)


def representative_point(geometry: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return points unchanged and reduce station outlines to their bbox centre."""
    if not geometry:
        return geometry
    if geometry.get("type") == "Point":
        return geometry
    coordinates = list(coordinate_pairs(geometry.get("coordinates", [])))
    if not coordinates:
        return None
    longitudes, latitudes = zip(*coordinates)
    return {
        "type": "Point",
        "coordinates": [
            (min(longitudes) + max(longitudes)) / 2,
            (min(latitudes) + max(latitudes)) / 2,
        ],
    }


def normalize_station(feature: dict[str, Any], mode: str) -> dict[str, Any]:
    properties = feature.get("properties") or {}
    if mode == "rail":
        category = classify_rail_station(properties)
    elif text(properties, "amenity") == "bus_station" or \
            text(properties, "public_transport") == "station":
        category = "bus_station"
    elif yes(properties, "trolleybus"):
        category = "trolleybus_stop"
    else:
        category = "bus_stop"
    return {
        "type": "Feature",
        "geometry": representative_point(feature.get("geometry")),
        "properties": {
            "kind": "station",
            "category": category,
            "osmId": text(properties, "osm_id") or text(properties, "osm_way_id"),
            "name": text(properties, "name"),
            "ref": text(properties, "ref") or text(properties, "local_ref"),
            "railway": text(properties, "railway"),
            "publicTransport": text(properties, "public_transport"),
            "routes": route_list(text(properties, "route_ref")),
            "network": text(properties, "network"),
            "operator": text(properties, "operator"),
            "wheelchair": text(properties, "wheelchair"),
            "shelter": text(properties, "shelter"),
            "bench": text(properties, "bench"),
        },
    }


def normalize_route(feature: dict[str, Any]) -> dict[str, Any]:
    properties = feature.get("properties") or {}
    route_type = text(properties, "route").lower()
    return {
        "type": "Feature",
        "geometry": feature.get("geometry"),
        "properties": {
            "kind": "route",
            "category": f"{route_type}_route",
            "routeType": route_type,
            "osmId": text(properties, "osm_id"),
            "name": text(properties, "name"),
            "ref": text(properties, "ref"),
            "from": text(properties, "from"),
            "to": text(properties, "to"),
            "via": text(properties, "via"),
            "network": text(properties, "network"),
            "operator": text(properties, "operator"),
            "colour": text(properties, "colour"),
        },
    }


def normalize_road(feature: dict[str, Any]) -> dict[str, Any]:
    properties = feature.get("properties") or {}
    road_type = text(properties, "highway").lower()
    category = road_type.removesuffix("_link")
    return {
        "type": "Feature",
        "geometry": feature.get("geometry"),
        "properties": {
            "kind": "road",
            "category": category,
            "roadType": road_type,
            "link": road_type.endswith("_link"),
            "osmId": text(properties, "osm_id"),
            "name": text(properties, "name"),
            "ref": text(properties, "ref"),
            "lanes": text(properties, "lanes"),
            "maxspeed": text(properties, "maxspeed"),
            "surface": text(properties, "surface"),
            "oneway": text(properties, "oneway"),
            "bridge": text(properties, "bridge"),
            "tunnel": text(properties, "tunnel"),
            "access": text(properties, "access"),
            "busway": text(properties, "busway"),
        },
    }


def line_coordinates(geometry: dict[str, Any]) -> list[tuple[float, float]]:
    """Flatten line geometry into its original OSM coordinate vertices."""
    coordinates = geometry.get("coordinates", [])
    if geometry.get("type") == "LineString":
        return [(float(x), float(y)) for x, y, *_ in coordinates]
    if geometry.get("type") == "MultiLineString":
        return [
            (float(x), float(y))
            for line in coordinates
            for x, y, *_ in line
        ]
    return []


def osm_node_key(longitude: float, latitude: float) -> tuple[int, int]:
    """Stable exact-coordinate proxy for an OSM shared node in GDAL output."""
    return round(longitude * 100_000_000), round(latitude * 100_000_000)


def level(properties: dict[str, Any]) -> str:
    return text(properties, "layer").strip() or "0"


def structure(properties: dict[str, Any], name: str) -> str:
    return text(properties, name).strip().lower() in {"yes", "true", "1"} and "yes" or ""


def compatible_connection(link: dict[str, Any], road: dict[str, Any]) -> bool:
    """Reject same-coordinate crossings that are mapped on different levels."""
    return (
        level(link) == level(road)
        and structure(link, "bridge") == structure(road, "bridge")
        and structure(link, "tunnel") == structure(road, "tunnel")
    )


def motorway_connector_features(
    motorway_link_features: list[dict[str, Any]],
    all_road_features: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return only link endpoints sharing an OSM node with a non-motorway road.

    A motorway link can touch a motorway at one end and a surface road at the
    other. The latter endpoint, not the full link geometry, is the access
    location used by the model feature.
    """
    link_endpoints: dict[tuple[int, int], list[tuple[float, float, dict[str, Any]]]] = {}
    for feature in motorway_link_features:
        geometry = feature.get("geometry")
        lines = line_coordinates(geometry) if geometry else []
        if len(lines) < 2:
            continue
        properties = feature.get("properties") or {}
        for longitude, latitude in (lines[0], lines[-1]):
            link_endpoints.setdefault(osm_node_key(longitude, latitude), []).append(
                (longitude, latitude, properties)
            )

    connected_by_node: dict[tuple[int, int], set[str]] = {}
    for feature in all_road_features:
        properties = feature.get("properties") or {}
        highway = text(properties, "highway").lower()
        geometry = feature.get("geometry")
        if not geometry or not highway or highway in {"motorway", "motorway_link"}:
            continue
        for longitude, latitude in line_coordinates(geometry):
            node = osm_node_key(longitude, latitude)
            links_at_node = link_endpoints.get(node, [])
            if not links_at_node:
                continue
            if any(compatible_connection(link_properties, properties)
                   for _, _, link_properties in links_at_node):
                connected_by_node.setdefault(node, set()).add(highway)
    if not connected_by_node:
        return []
    result: list[dict[str, Any]] = []
    for node, connected_types in connected_by_node.items():
        longitude, latitude, properties = link_endpoints[node][0]
        result.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [longitude, latitude]},
            "properties": {
                "kind": "motorway_connector",
                "category": "motorway_connector",
                "osmId": text(properties, "osm_id"),
                "connectedRoadTypes": sorted(connected_types),
            },
        })
    return result


def load_features(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return json.loads(path.read_text()).get("features", [])


def iter_features(path: Path) -> Iterable[dict[str, Any]]:
    """Stream GeoJSONSeq rather than materialising the full road network."""
    if path.suffix == ".geojsonseq":
        with open(path, "r") as source:
            for line in source:
                if line.strip():
                    yield json.loads(line)
        return
    yield from load_features(path)


def write_collection(
    path: Path,
    features: list[dict[str, Any]],
    *,
    source: Path,
    bounds: tuple[float, float, float, float],
    simplify_degrees: float,
) -> None:
    category_counts = Counter(
        feature["properties"]["category"] for feature in features
    )
    kind_counts = Counter(feature["properties"]["kind"] for feature in features)
    payload = {
        "type": "FeatureCollection",
        "metadata": {
            "source": source.name,
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "clipBounds": list(bounds),
            "simplifyDegrees": simplify_degrees,
            "featureCount": len(features),
            "countsByKind": dict(sorted(kind_counts.items())),
            "countsByCategory": dict(sorted(category_counts.items())),
        },
        "features": features,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--deploy-dir", type=Path, default=DEFAULT_DEPLOY_DIR)
    parser.add_argument("--buffer-degrees", type=float, default=0.05)
    parser.add_argument(
        "--simplify-degrees", type=float, default=0.00008,
        help="Line simplification tolerance, approximately six metres locally.",
    )
    args = parser.parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"OSM input not found: {args.input}")
    if shutil.which("ogr2ogr") is None:
        raise RuntimeError("ogr2ogr is required; install GDAL before running this script")

    config = Path(__file__).with_name("osm_transport_config.ini")
    with tempfile.TemporaryDirectory(prefix="osm-transport-") as temporary_directory:
        temporary = Path(temporary_directory)
        boundary_path = temporary / "king_county_boundary.geojson"
        run_ogr2ogr(
            [
                "-overwrite", "-f", "GeoJSON",
                "-where", "boundary = 'administrative' AND name = 'King County'",
                "-select", "osm_id,name,boundary,admin_level",
                str(boundary_path), str(args.input), "multipolygons",
            ],
            config,
        )
        west, south, east, north = geometry_bounds(
            json.loads(boundary_path.read_text())
        )
        bounds = (
            west - args.buffer_degrees,
            south - args.buffer_degrees,
            east + args.buffer_degrees,
            north + args.buffer_degrees,
        )
        bbox = [f"{coordinate:.7f}" for coordinate in bounds]
        common = [
            "-overwrite", "-f", "GeoJSON", "-skipfailures",
            "-spat", *bbox, "-clipsrc", *bbox,
        ]

        rail_stations_path = temporary / "rail_stations.geojson"
        rail_station_lines_path = temporary / "rail_station_lines.geojson"
        rail_station_areas_path = temporary / "rail_station_areas.geojson"
        rail_routes_path = temporary / "rail_routes.geojson"
        bus_stations_path = temporary / "bus_stations.geojson"
        bus_routes_path = temporary / "bus_routes.geojson"
        roads_path = temporary / "major_roads.geojson"
        motorway_links_path = temporary / "motorway_links.geojson"
        all_roads_path = temporary / "all_roads.geojsonseq"

        run_ogr2ogr(
            [
                *common,
                "-where",
                "railway IN ('station','halt','tram_stop') AND "
                "(light_rail = 'yes' OR tram = 'yes' OR subway = 'yes' OR "
                "monorail = 'yes' OR train = 'yes' OR "
                "station IN ('light_rail','tram','subway','monorail') OR "
                "railway IN ('station','halt','tram_stop'))",
                "-select",
                "osm_id,name,ref,railway,public_transport,station,train,"
                "light_rail,tram,subway,monorail,route_ref,network,operator,"
                "wheelchair,shelter,bench,local_ref",
                str(rail_stations_path), str(args.input), "points",
            ],
            config,
        )
        # Do not simplify these geometries: shared OSM-node matching requires
        # the original motorway-link endpoints and road vertices.
        run_ogr2ogr(
            [
                *common,
                "-where", "highway = 'motorway_link'",
                "-select", "osm_id,highway,bridge,tunnel,layer",
                str(motorway_links_path), str(args.input), "lines",
            ],
            config,
        )
        run_ogr2ogr(
            [
                "-overwrite", "-f", "GeoJSONSeq", "-skipfailures",
                "-spat", *bbox, "-clipsrc", *bbox,
                "-where", "highway IS NOT NULL AND highway <> ''",
                "-select", "osm_id,highway,bridge,tunnel,layer",
                str(all_roads_path), str(args.input), "lines",
            ],
            config,
        )
        station_filter = (
            "railway IN ('station','halt','tram_stop') AND "
            "(light_rail = 'yes' OR tram = 'yes' OR subway = 'yes' OR "
            "monorail = 'yes' OR train = 'yes' OR "
            "station IN ('light_rail','tram','subway','monorail') OR "
            "railway IN ('station','halt','tram_stop'))"
        )
        station_fields = (
            "osm_id,name,ref,railway,public_transport,station,train,"
            "light_rail,tram,subway,monorail,route_ref,network,operator,"
            "wheelchair"
        )
        station_area_fields = station_fields.replace(
            "osm_id,", "osm_id,osm_way_id,"
        )
        run_ogr2ogr(
            [
                *common, "-where", station_filter, "-select", station_fields,
                str(rail_station_lines_path), str(args.input), "lines",
            ],
            config,
        )
        run_ogr2ogr(
            [
                *common, "-where", station_filter, "-select", station_area_fields,
                str(rail_station_areas_path), str(args.input), "multipolygons",
            ],
            config,
        )
        run_ogr2ogr(
            [
                *common, "-simplify", str(args.simplify_degrees),
                "-where", "route IN ('light_rail','tram','monorail','train')",
                "-select", "osm_id,name,type,route,ref,from,to,via,operator,network,colour",
                str(rail_routes_path), str(args.input), "multilinestrings",
            ],
            config,
        )
        run_ogr2ogr(
            [
                *common,
                "-where",
                "highway = 'bus_stop' OR amenity = 'bus_station' OR "
                "(public_transport IN ('platform','station') AND "
                "(bus = 'yes' OR trolleybus = 'yes'))",
                "-select",
                "osm_id,name,ref,highway,amenity,public_transport,bus,trolleybus,"
                "route_ref,network,operator,wheelchair,shelter,bench,local_ref",
                str(bus_stations_path), str(args.input), "points",
            ],
            config,
        )
        run_ogr2ogr(
            [
                *common, "-simplify", str(args.simplify_degrees),
                "-where", "route IN ('bus','trolleybus')",
                "-select", "osm_id,name,type,route,ref,from,to,via,operator,network,colour",
                str(bus_routes_path), str(args.input), "multilinestrings",
            ],
            config,
        )
        run_ogr2ogr(
            [
                *common, "-simplify", str(args.simplify_degrees),
                "-where",
                "highway IN ('motorway','motorway_link','trunk','trunk_link',"
                "'primary','primary_link','secondary','secondary_link')",
                "-select",
                "osm_id,name,ref,highway,lanes,maxspeed,surface,oneway,bridge,"
                "tunnel,access,busway",
                str(roads_path), str(args.input), "lines",
            ],
            config,
        )

        rail_station_features = [
            normalize_station(feature, "rail")
            for feature in load_features(rail_stations_path)
        ]
        # Some stations are mapped only as closed ways or areas. Prefer an
        # existing station node with the same name/category, otherwise add the
        # outline's representative point.
        point_station_keys = {
            (
                feature["properties"]["name"].strip().casefold(),
                feature["properties"]["category"],
            )
            for feature in rail_station_features
            if feature["properties"]["name"].strip()
        }
        for source_path in (rail_station_lines_path, rail_station_areas_path):
            for raw_feature in load_features(source_path):
                station_feature = normalize_station(raw_feature, "rail")
                key = (
                    station_feature["properties"]["name"].strip().casefold(),
                    station_feature["properties"]["category"],
                )
                if key[0] and key in point_station_keys:
                    continue
                rail_station_features.append(station_feature)
                if key[0]:
                    point_station_keys.add(key)
        rail_features = rail_station_features + [
            normalize_route(feature) for feature in load_features(rail_routes_path)
        ]
        bus_features = [
            normalize_station(feature, "bus")
            for feature in load_features(bus_stations_path)
        ] + [normalize_route(feature) for feature in load_features(bus_routes_path)]
        road_features = [normalize_road(feature) for feature in load_features(roads_path)]
        connector_features = motorway_connector_features(
            load_features(motorway_links_path), iter_features(all_roads_path)
        )

        artifacts = {
            "king_county_rail.geojson": rail_features,
            "king_county_bus.geojson": bus_features,
            "king_county_major_roads.geojson": road_features,
            "king_county_motorway_connectors.geojson": connector_features,
        }
        for filename, features in artifacts.items():
            output_path = args.output_dir / filename
            write_collection(
                output_path, features, source=args.input, bounds=bounds,
                simplify_degrees=args.simplify_degrees,
            )
            if args.deploy_dir:
                args.deploy_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(output_path, args.deploy_dir / filename)
            counts = Counter(feature["properties"]["category"] for feature in features)
            print(f"Wrote {len(features):,} features to {output_path}: {dict(sorted(counts.items()))}")


if __name__ == "__main__":
    main()
