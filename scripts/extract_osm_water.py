#!/usr/bin/env python3
"""Extract a compact, buffered King County water layer from a Geofabrik PBF."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_INPUT = Path("data/osm/washington-latest.osm.pbf")
DEFAULT_OUTPUT = Path("data/osm/king_county_water.geojson")
DEFAULT_DEPLOY_OUTPUT = Path(
    "java-app/house-price-app/src/main/resources/META-INF/resources/data/king_county_water.geojson"
)


def run_ogr2ogr(arguments: list[str], config: Path) -> None:
    environment = os.environ.copy()
    environment["OSM_CONFIG_FILE"] = str(config.resolve())
    subprocess.run(["ogr2ogr", *arguments], check=True, env=environment)


def coordinate_pairs(value: Any) -> Iterable[tuple[float, float]]:
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


def geometry_bounds(feature_collection: dict[str, Any]) -> tuple[float, float, float, float]:
    coordinates = [
        pair
        for feature in feature_collection.get("features", [])
        for pair in coordinate_pairs(feature.get("geometry", {}).get("coordinates", []))
    ]
    if not coordinates:
        raise RuntimeError("King County boundary extraction returned no coordinates")
    longitudes, latitudes = zip(*coordinates)
    return min(longitudes), min(latitudes), max(longitudes), max(latitudes)


def classify(properties: dict[str, Any]) -> tuple[str, str]:
    natural = properties.get("natural") or ""
    waterway = properties.get("waterway") or ""
    if natural == "coastline":
        return "coastline", "coastline"
    return "natural_water", properties.get("water") or waterway or "water"


def normalized_features(path: Path) -> list[dict[str, Any]]:
    collection = json.loads(path.read_text())
    result: list[dict[str, Any]] = []
    for feature in collection.get("features", []):
        properties = feature.get("properties") or {}
        category, subtype = classify(properties)
        clean_properties = {
            "osmId": properties.get("osm_id") or properties.get("osm_way_id") or "",
            "name": properties.get("name") or "",
            "category": category,
            "type": subtype,
            "natural": properties.get("natural") or "",
            "waterway": properties.get("waterway") or "",
            "intermittent": properties.get("intermittent") or "",
        }
        result.append(
            {
                "type": "Feature",
                "geometry": feature.get("geometry"),
                "properties": clean_properties,
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--deploy-output", type=Path, default=DEFAULT_DEPLOY_OUTPUT,
        help="Optional Java static-resource copy of the canonical artifact.",
    )
    parser.add_argument(
        "--buffer-degrees",
        type=float,
        default=0.05,
        help="Buffer around the King County boundary envelope (roughly 4-6 km).",
    )
    parser.add_argument(
        "--simplify-degrees",
        type=float,
        default=0.00003,
        help="Geometry simplification tolerance (about 2-3 metres locally).",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"OSM input not found: {args.input}")
    if shutil.which("ogr2ogr") is None:
        raise RuntimeError("ogr2ogr is required; install GDAL before running this script")

    config = Path(__file__).with_name("osm_water_config.ini")
    with tempfile.TemporaryDirectory(prefix="osm-water-") as temporary_directory:
        temporary = Path(temporary_directory)
        boundary_path = temporary / "king_county_boundary.geojson"
        polygon_path = temporary / "water_polygons.geojson"
        line_path = temporary / "water_lines.geojson"
        normalized_path = temporary / "water_normalized.geojson"
        valid_path = temporary / "water_valid.geojson"
        clipped_path = temporary / "water_clipped.geojson"

        run_ogr2ogr(
            [
                "-overwrite", "-f", "GeoJSON",
                "-where", "boundary = 'administrative' AND name = 'King County'",
                "-select", "osm_id,name,boundary,admin_level",
                str(boundary_path), str(args.input), "multipolygons",
            ],
            config,
        )
        boundary = json.loads(boundary_path.read_text())
        west, south, east, north = geometry_bounds(boundary)
        bbox = (
            west - args.buffer_degrees,
            south - args.buffer_degrees,
            east + args.buffer_degrees,
            north + args.buffer_degrees,
        )
        bbox_args = [f"{coordinate:.7f}" for coordinate in bbox]

        common = [
            "-overwrite", "-f", "GeoJSON", "-skipfailures",
            "-simplify", str(args.simplify_degrees),
            "-spat", *bbox_args,
        ]
        polygon_fields = "osm_id,osm_way_id,name,natural,waterway,water,intermittent"
        line_fields = "osm_id,name,natural,waterway,water,intermittent"
        run_ogr2ogr(
            [
                *common,
                "-where",
                "natural = 'water' AND waterway IS NULL AND "
                "(water IS NULL OR water NOT IN ('stream','drain','ditch','weir'))",
                "-select", polygon_fields,
                str(polygon_path), str(args.input), "multipolygons",
            ],
            config,
        )
        run_ogr2ogr(
            [
                *common,
                "-where", "natural = 'coastline'",
                "-select", line_fields,
                str(line_path), str(args.input), "lines",
            ],
            config,
        )

        selected_features = normalized_features(polygon_path) + normalized_features(line_path)
        normalized_path.write_text(json.dumps({
            "type": "FeatureCollection",
            "features": selected_features,
        }))
        # Clipping invalid OSM polygons directly can drop features. Repair first, then clip.
        run_ogr2ogr(
            ["-f", "GeoJSON", "-makevalid", str(valid_path), str(normalized_path)],
            config,
        )
        run_ogr2ogr(
            ["-f", "GeoJSON", "-clipsrc", *bbox_args, str(clipped_path), str(valid_path)],
            config,
        )
        features = json.loads(clipped_path.read_text()).get("features", [])
        counts = Counter(feature["properties"]["category"] for feature in features)
        output = {
            "type": "FeatureCollection",
            "metadata": {
                "source": args.input.name,
                "generatedAt": datetime.now(timezone.utc).isoformat(),
                "kingCountyBounds": [west, south, east, north],
                "clipBounds": list(bbox),
                "bufferDegrees": args.buffer_degrees,
                "simplifyDegrees": args.simplify_degrees,
                "featureCount": len(features),
                "countsByCategory": dict(sorted(counts.items())),
            },
            "features": features,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, separators=(",", ":")))
        if args.deploy_output:
            args.deploy_output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(args.output, args.deploy_output)

    print(f"Wrote {len(features):,} features to {args.output}")
    if args.deploy_output:
        print(f"Copied deployment layer to {args.deploy_output}")
    print(f"Clip bounds: {bbox}")
    print(f"Counts: {dict(sorted(counts.items()))}")


if __name__ == "__main__":
    main()
