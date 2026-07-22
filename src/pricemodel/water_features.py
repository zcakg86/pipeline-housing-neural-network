"""Static proximity-to-water features shared by model training pipelines."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from shapely import points
from shapely.geometry import shape
from shapely.ops import transform
from shapely.strtree import STRtree


WATERFRONT_DISTANCE_METERS = 50.0
WATER_PROXIMITY_TAU_METERS = 100.0
DEFAULT_WATER_PATH = Path("data/osm/king_county_water.geojson")
DEFAULT_CACHE_DIR = Path("data/cache/water_features")
EARTH_RADIUS_METERS = 6_371_008.8
REFERENCE_LATITUDE_DEGREES = 47.4
_LONGITUDE_SCALE = EARTH_RADIUS_METERS * np.cos(
    np.radians(REFERENCE_LATITUDE_DEGREES)
) * np.pi / 180.0
_LATITUDE_SCALE = EARTH_RADIUS_METERS * np.pi / 180.0


def _project(longitude, latitude, altitude=None):
    return np.asarray(longitude) * _LONGITUDE_SCALE, np.asarray(latitude) * _LATITUDE_SCALE


def _water_boundaries(path: Path):
    collection = json.loads(path.read_text())
    boundaries = []
    for feature in collection.get("features", []):
        geometry = feature.get("geometry")
        if not geometry:
            continue
        value = shape(geometry)
        if value.is_empty:
            continue
        # Distance is deliberately measured to the shoreline/bank, not to the
        # polygon interior. Coastlines are already represented as linework.
        if value.geom_type in {"Polygon", "MultiPolygon"}:
            value = value.boundary
        boundaries.append(transform(_project, value))
    if not boundaries:
        raise ValueError(f"Water artifact contains no usable boundaries: {path}")
    return boundaries


def _cache_key(frame, water_path: Path) -> str:
    digest = hashlib.sha256()
    stat = water_path.stat()
    digest.update(f"{water_path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    digest.update(np.asarray(frame[["lng", "lat"]], dtype=np.float64).tobytes())
    return digest.hexdigest()[:24]


def add_water_proximity_features(
    frame,
    water_path=DEFAULT_WATER_PATH,
    waterfront_distance_m=WATERFRONT_DISTANCE_METERS,
    cache_dir=DEFAULT_CACHE_DIR,
):
    """Return distance diagnostics plus bounded model-facing water features.

    ``distance_to_water_m`` is retained for inspection and map display. Models
    consume ``water_proximity = exp(-distance / 100m)`` instead, so shoreline
    influence decays rapidly and is effectively zero by 500 metres.
    """
    water_path = Path(water_path)
    if not water_path.exists():
        raise FileNotFoundError(
            f"Water boundary artifact not found: {water_path}. "
            "Run scripts/extract_osm_water.py first."
        )
    if not {"lat", "lng"}.issubset(frame.columns):
        raise ValueError("Water features require lat and lng columns")
    if frame[["lat", "lng"]].isna().any().any():
        raise ValueError("Water features cannot be calculated for missing coordinates")

    result = frame.copy()
    cache_path = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"water_{_cache_key(result, water_path)}.npz"
        if cache_path.exists():
            cached = np.load(cache_path)
            distances = cached["distance_to_water_m"]
            if len(distances) == len(result):
                result["distance_to_water_m"] = distances
                result["water_proximity"] = np.exp(
                    -distances / WATER_PROXIMITY_TAU_METERS
                ).astype(np.float32)
                result["is_waterfront"] = (distances <= waterfront_distance_m).astype(np.float32)
                return result

    longitude = result["lng"].to_numpy(dtype=np.float64)
    latitude = result["lat"].to_numpy(dtype=np.float64)
    x, y = _project(longitude, latitude)
    property_points = points(x, y)
    tree = STRtree(_water_boundaries(water_path))
    _, distances = tree.query_nearest(
        property_points, all_matches=False, return_distance=True
    )
    distances = np.asarray(distances, dtype=np.float64)
    if len(distances) != len(result) or not np.isfinite(distances).all():
        raise RuntimeError("Nearest-water calculation returned invalid distances")

    result["distance_to_water_m"] = distances
    result["water_proximity"] = np.exp(
        -distances / WATER_PROXIMITY_TAU_METERS
    ).astype(np.float32)
    result["is_waterfront"] = (distances <= waterfront_distance_m).astype(np.float32)
    if cache_path is not None:
        np.savez_compressed(cache_path, distance_to_water_m=distances)
    return result
