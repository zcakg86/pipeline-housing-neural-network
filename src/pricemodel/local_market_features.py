"""Leakage-safe, time-aware market features for H3 neighborhoods."""

from collections import deque
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


LOCAL_MARKET_FEATURES = [
    "local_mean_log_price",
    "local_log_price_std",
    "local_log1p_sales_count",
    "local_recency_years",
    "local_price_trend",
]
_EPOCH_DAY = np.datetime64('1970-01-01', 'D')
LOCAL_MARKET_FEATURE_VERSION = 2


def _day_to_timestamp(day: int) -> pd.Timestamp:
    return pd.Timestamp(_EPOCH_DAY + np.timedelta64(int(day), 'D'))


def _serialize_ring(value) -> str:
    if not isinstance(value, (list, tuple)):
        return ""
    return "|".join("" if item is None else str(item) for item in value)


def _cache_identity(
    dataframe: pd.DataFrame,
    neighbor_cells_col: str,
    location_col: str,
    date_col: str,
    target_col: str,
    recent_window_days: int,
    max_recency_years: float,
    default_log_price: float,
    default_log_price_std: float,
    h3_resolution: int,
    snapshot_as_of_date,
    cache_dependencies,
):
    """Return a content key plus per-row hashes used to validate cache order."""
    identity = pd.DataFrame({
        "sale_day": pd.to_datetime(dataframe[date_col]).to_numpy(dtype='datetime64[D]').astype(np.int64),
        "location": dataframe[location_col].fillna("").astype(str),
        "log_price": dataframe[target_col].to_numpy(dtype=np.float64),
        "neighbor_cells": dataframe[neighbor_cells_col].map(_serialize_ring),
    })
    if "community_neighbors" in dataframe.columns:
        identity["community_neighbors"] = dataframe["community_neighbors"].map(
            _serialize_ring
        )
    row_hashes = pd.util.hash_pandas_object(
        identity, index=False, categorize=True
    ).to_numpy(dtype=np.uint64)

    configuration = {
        "feature_version": LOCAL_MARKET_FEATURE_VERSION,
        "feature_order": LOCAL_MARKET_FEATURES,
        "recent_window_days": recent_window_days,
        "max_recency_years": max_recency_years,
        "default_log_price": default_log_price,
        "default_log_price_std": default_log_price_std,
        "h3_resolution": h3_resolution,
        "snapshot_as_of_date": (
            pd.Timestamp(snapshot_as_of_date).normalize().date().isoformat()
            if snapshot_as_of_date is not None else None
        ),
        "rows": len(dataframe),
    }
    digest = hashlib.sha256()
    digest.update(json.dumps(configuration, sort_keys=True).encode("utf-8"))
    digest.update(row_hashes.tobytes())

    dependency_metadata = {}
    for dependency in sorted(
        (Path(path) for path in (cache_dependencies or []) if path is not None),
        key=lambda path: path.name,
    ):
        dependency_digest = hashlib.sha256()
        if dependency.exists():
            with open(dependency, "rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    dependency_digest.update(chunk)
            dependency_hash = dependency_digest.hexdigest()
        else:
            dependency_hash = "missing"
        dependency_metadata[dependency.name] = dependency_hash
        digest.update(dependency.name.encode("utf-8"))
        digest.update(dependency_hash.encode("ascii"))

    configuration["dependencies"] = dependency_metadata
    return digest.hexdigest(), row_hashes, configuration


def _load_feature_cache(cache_dir, cache_key, expected_row_hashes, expected_shape):
    npz_path = cache_dir / f"{cache_key}.npz"
    snapshot_path = cache_dir / f"{cache_key}.snapshot.json"
    if not npz_path.exists() or not snapshot_path.exists():
        return None
    try:
        with np.load(npz_path, allow_pickle=False) as archive:
            features = archive["local_market_features"]
            cached_row_hashes = archive["row_identity"]
        if features.shape != expected_shape:
            raise ValueError(f"feature shape {features.shape} != {expected_shape}")
        if not np.array_equal(cached_row_hashes, expected_row_hashes):
            raise ValueError("row identity/order does not match")
        with open(snapshot_path, "r") as source:
            snapshot = json.load(source)
        print(f"Local market feature cache hit: {npz_path}")
        return features.astype(np.float32, copy=False), snapshot
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"Ignoring invalid local market cache {npz_path}: {exc}")
        return None


def _save_feature_cache(
    cache_dir,
    cache_key,
    features,
    row_hashes,
    snapshot,
    cache_metadata,
):
    cache_dir.mkdir(parents=True, exist_ok=True)
    npz_path = cache_dir / f"{cache_key}.npz"
    snapshot_path = cache_dir / f"{cache_key}.snapshot.json"
    suffix = f".{os.getpid()}.tmp"
    temporary_npz = cache_dir / f"{cache_key}{suffix}.npz"
    temporary_snapshot = cache_dir / f"{cache_key}{suffix}.snapshot.json"

    np.savez_compressed(
        temporary_npz,
        local_market_features=features,
        row_identity=row_hashes,
    )
    snapshot = dict(snapshot)
    snapshot["cache"] = {
        "key": cache_key,
        **cache_metadata,
    }
    with open(temporary_snapshot, "w") as target:
        json.dump(snapshot, target)
    os.replace(temporary_npz, npz_path)
    os.replace(temporary_snapshot, snapshot_path)
    print(f"Saved local market feature cache: {npz_path}")
    return snapshot


@dataclass
class _CellState:
    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    last_sale_day: Optional[int] = None
    recent_sales: deque = field(default_factory=deque)
    recent_total: float = 0.0

    def update(self, day: int, log_price: float, track_recent: bool = True) -> None:
        self.count += 1
        self.total += log_price
        self.total_sq += log_price * log_price
        self.last_sale_day = day
        if track_recent:
            self.recent_sales.append((day, log_price))
            self.recent_total += log_price


def _mean_std(state: _CellState, default_mean: float, default_std: float) -> Tuple[float, float]:
    if state.count == 0:
        return default_mean, default_std
    mean = state.total / state.count
    if state.count < 2:
        return mean, default_std
    variance = max((state.total_sq - state.count * mean * mean) / (state.count - 1), 0.0)
    return mean, float(np.sqrt(variance))


def _cell_features(
    state: Optional[_CellState],
    as_of_day: int,
    recent_cutoff_day: int,
    fallback_mean: float,
    fallback_std: float,
    max_recency_years: float,
) -> List[float]:
    if state is None or state.count == 0:
        return [fallback_mean, fallback_std, 0.0, max_recency_years, 0.0]

    mean, std = _mean_std(state, fallback_mean, fallback_std)
    while state.recent_sales and state.recent_sales[0][0] < recent_cutoff_day:
        _, expired_price = state.recent_sales.popleft()
        state.recent_total -= expired_price

    if state.recent_sales:
        recent_mean = state.recent_total / len(state.recent_sales)
        trend = recent_mean - mean
    else:
        trend = 0.0

    recency_years = min(
        max((as_of_day - state.last_sale_day) / 365.25, 0.0),
        max_recency_years,
    )
    return [mean, std, float(np.log1p(state.count)), recency_years, trend]


def build_historical_local_features(
    dataframe: pd.DataFrame,
    neighbor_cells_col: str = "h3_neighbor_cells",
    location_col: str = "h3_08",
    date_col: str = "sale_date",
    target_col: str = "log_price",
    recent_window_days: int = 365,
    max_recency_years: float = 5.0,
    default_log_price: float = float(np.log(300_000.0)),
    default_log_price_std: float = 0.5,
    h3_resolution: int = 8,
    snapshot_as_of_date=None,
    cache_dir=None,
    cache_dependencies=None,
    force_recompute: bool = False,
) -> Tuple[np.ndarray, Dict]:
    """Build a ``[rows, 7, features]`` tensor using only sales before each row.

    Rows sharing a sale date are evaluated before any of that date's sales are
    added to state. This prevents both self leakage and same-day cross leakage.
    The returned snapshot is suitable for future inference after training.
    ``snapshot_as_of_date`` can move the snapshot beyond the newest sale so
    rolling trend and recency features correctly age during deployment. All
    input sales must be strictly earlier than that date.
    """
    required = {neighbor_cells_col, location_col, date_col, target_col}
    missing = required.difference(dataframe.columns)
    if missing:
        raise ValueError(f"Cannot build local market features; missing columns: {sorted(missing)}")

    dates = pd.to_datetime(dataframe[date_col]).dt.normalize()
    date_days = dates.to_numpy(dtype='datetime64[D]').astype(np.int64)
    if len(date_days) > 1 and np.any(date_days[1:] < date_days[:-1]):
        raise ValueError("Dataframe must be sorted chronologically before local features are built")

    n_rows = len(dataframe)
    cache_key = None
    row_hashes = None
    cache_metadata = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_key, row_hashes, cache_metadata = _cache_identity(
            dataframe,
            neighbor_cells_col,
            location_col,
            date_col,
            target_col,
            recent_window_days,
            max_recency_years,
            default_log_price,
            default_log_price_std,
            h3_resolution,
            snapshot_as_of_date,
            cache_dependencies,
        )
        if not force_recompute:
            cached = _load_feature_cache(
                cache_dir,
                cache_key,
                row_hashes,
                (n_rows, 7, len(LOCAL_MARKET_FEATURES)),
            )
            if cached is not None:
                return cached

    values = np.zeros((n_rows, 7, len(LOCAL_MARKET_FEATURES)), dtype=np.float32)
    states: Dict[str, _CellState] = {}
    global_state = _CellState()
    neighbor_lists = dataframe[neighbor_cells_col].tolist()
    locations = dataframe[location_col].tolist()
    log_prices = dataframe[target_col].to_numpy(dtype=float)

    # Work by date so no sale can see itself or another sale from the same day.
    # Integer day ordinals avoid Timestamp/Timedelta allocation in the hot loop.
    unique_days, date_starts = np.unique(date_days, return_index=True)
    date_ends = np.r_[date_starts[1:], n_rows]
    for sale_day, start, end in zip(unique_days, date_starts, date_ends):
        sale_day = int(sale_day)
        recent_cutoff_day = sale_day - recent_window_days
        fallback_mean, fallback_std = _mean_std(
            global_state, default_log_price, default_log_price_std
        )
        missing_features = (
            fallback_mean, fallback_std, 0.0, max_recency_years, 0.0
        )
        # A cell's state is identical for every property on the same date.
        # Cache it once rather than recalculating it for every overlapping ring.
        date_feature_cache = {}

        for position in range(start, end):
            cells = neighbor_lists[position]
            if not isinstance(cells, list) or len(cells) != 7:
                cells = [locations[position]] + [None] * 6
            for ring_position, cell in enumerate(cells):
                if cell is None:
                    features = missing_features
                else:
                    features = date_feature_cache.get(cell)
                    if features is None:
                        features = _cell_features(
                            states.get(cell),
                            sale_day,
                            recent_cutoff_day,
                            fallback_mean,
                            fallback_std,
                            max_recency_years,
                        )
                        date_feature_cache[cell] = features
                values[position, ring_position] = features

        # Update only after every row on this date has been encoded.
        for position in range(start, end):
            cell = locations[position]
            log_price = log_prices[position]
            if cell is None or not np.isfinite(log_price):
                continue
            state = states.setdefault(cell, _CellState())
            state.update(sale_day, log_price)
            # The global state is used only for fallback moments, so retaining
            # its own duplicate rolling deque wastes memory.
            global_state.update(sale_day, log_price, track_recent=False)

    if snapshot_as_of_date is not None:
        snapshot_timestamp = pd.Timestamp(snapshot_as_of_date).normalize()
        snapshot_day = int(
            np.datetime64(snapshot_timestamp.date(), 'D').astype(np.int64)
        )
        if n_rows and snapshot_day <= int(date_days[-1]):
            latest_sale = _day_to_timestamp(int(date_days[-1])).date().isoformat()
            raise ValueError(
                "snapshot_as_of_date must be strictly after every included "
                f"sale; latest included sale is {latest_sale}"
            )
    elif n_rows:
        snapshot_day = int(date_days[-1]) + 1
    else:
        snapshot_day = int(
            np.datetime64(pd.Timestamp.utcnow().date(), 'D').astype(np.int64)
        )
    snapshot_date = _day_to_timestamp(snapshot_day)
    global_mean, global_std = _mean_std(
        global_state, default_log_price, default_log_price_std
    )

    cells_snapshot = {}
    for cell, state in states.items():
        feature_values = _cell_features(
            state,
            snapshot_day,
            snapshot_day - recent_window_days,
            global_mean,
            global_std,
            max_recency_years,
        )
        cells_snapshot[cell] = {
            "mean_log_price": feature_values[0],
            "log_price_std": feature_values[1],
            "log1p_sales_count": feature_values[2],
            "last_sale_date": _day_to_timestamp(state.last_sale_day).date().isoformat(),
            "price_trend": feature_values[4],
            # Java uses these dated observations to recompute the rolling
            # trend for any future prediction date. They contain only sales
            # already included in this snapshot.
            "recent_sales": [
                {
                    "sale_date": _day_to_timestamp(day).date().isoformat(),
                    "log_price": log_price,
                }
                for day, log_price in state.recent_sales
            ],
        }

    snapshot = {
        "as_of_date": snapshot_date.date().isoformat(),
        "latest_sale_date": (
            _day_to_timestamp(int(date_days[-1])).date().isoformat()
            if n_rows else None
        ),
        "recent_window_days": recent_window_days,
        "max_recency_years": max_recency_years,
        "feature_order": LOCAL_MARKET_FEATURES,
        "global": {
            "mean_log_price": global_mean,
            "log_price_std": global_std,
            "log1p_sales_count": 0.0,
            "last_sale_date": None,
            "price_trend": 0.0,
        },
        "cells": cells_snapshot,
    }
    if cache_dir is not None:
        snapshot = _save_feature_cache(
            cache_dir,
            cache_key,
            values,
            row_hashes,
            snapshot,
            cache_metadata,
        )
    return values, snapshot
