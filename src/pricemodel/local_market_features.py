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
    "local_decayed_log_price_premium",
    "local_decayed_log_price_std",
    "local_log1p_decayed_sales_count",
    "local_recency_years",
    "local_relative_price_trend",
]
_EPOCH_DAY = np.datetime64('1970-01-01', 'D')
LOCAL_MARKET_FEATURE_VERSION = 4
DEFAULT_DECAY_HALF_LIFE_DAYS = 730.0
DEFAULT_TREND_WINDOW_DAYS = 365
DEFAULT_PREMIUM_SHRINKAGE_WEIGHT = 3.0


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
    decay_half_life_days: float,
    trend_window_days: int,
    premium_shrinkage_weight: float,
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
        "decay_half_life_days": decay_half_life_days,
        "trend_window_days": trend_window_days,
        "premium_shrinkage_weight": premium_shrinkage_weight,
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
    """Lazily decayed sale-price moments for one H3 cell.

    ``weight`` is the sum of exponentially decayed transaction weights at
    ``last_decay_day``. Advancing the state therefore costs O(1), rather than
    rescanning every historical sale for each prediction date.
    """

    weight: float = 0.0
    weighted_total: float = 0.0
    weighted_total_sq: float = 0.0
    observation_count: int = 0
    last_decay_day: Optional[int] = None
    last_sale_day: Optional[int] = None
    trend_last_day: Optional[int] = None
    recent_trend_sales: deque = field(default_factory=deque)
    prior_trend_sales: deque = field(default_factory=deque)
    recent_trend_weight: float = 0.0
    recent_trend_total: float = 0.0
    prior_trend_weight: float = 0.0
    prior_trend_total: float = 0.0

    def advance_to(self, day: int, decay_half_life_days: float) -> None:
        """Age weighted moments to ``day`` without changing observed sales."""
        if self.last_decay_day is None:
            self.last_decay_day = day
            return
        if day < self.last_decay_day:
            raise ValueError("Local market state cannot move backwards in time")
        elapsed_days = day - self.last_decay_day
        if elapsed_days:
            decay = 0.5 ** (elapsed_days / decay_half_life_days)
            self.weight *= decay
            self.weighted_total *= decay
            self.weighted_total_sq *= decay
            self.last_decay_day = day

    def update(
        self,
        day: int,
        log_price: float,
        decay_half_life_days: float,
        trend_window_days: int,
        track_trend: bool = True,
    ) -> None:
        self.advance_to(day, decay_half_life_days)
        self.weight += 1.0
        self.weighted_total += log_price
        self.weighted_total_sq += log_price * log_price
        self.observation_count += 1
        self.last_sale_day = day
        if track_trend:
            self.advance_trend_to(day, trend_window_days, decay_half_life_days)
            self.recent_trend_sales.append((day, log_price))
            self.recent_trend_weight += 1.0
            self.recent_trend_total += log_price

    def advance_trend_to(
        self,
        day: int,
        trend_window_days: int,
        decay_half_life_days: float,
    ) -> None:
        """Age and roll recent/prior trend windows forward in amortized O(1)."""
        if self.trend_last_day is None:
            self.trend_last_day = day
        elif day < self.trend_last_day:
            raise ValueError("Local market trend state cannot move backwards in time")
        else:
            elapsed_days = day - self.trend_last_day
            if elapsed_days:
                decay = 0.5 ** (elapsed_days / decay_half_life_days)
                self.recent_trend_weight *= decay
                self.recent_trend_total *= decay
                self.prior_trend_weight *= decay
                self.prior_trend_total *= decay
                self.trend_last_day = day

        recent_start = day - trend_window_days
        prior_start = recent_start - trend_window_days
        while self.recent_trend_sales and self.recent_trend_sales[0][0] < recent_start:
            sale_day, log_price = self.recent_trend_sales.popleft()
            weight = 0.5 ** ((day - sale_day) / decay_half_life_days)
            self.recent_trend_weight -= weight
            self.recent_trend_total -= weight * log_price
            self.prior_trend_sales.append((sale_day, log_price))
            self.prior_trend_weight += weight
            self.prior_trend_total += weight * log_price
        while self.prior_trend_sales and self.prior_trend_sales[0][0] < prior_start:
            sale_day, log_price = self.prior_trend_sales.popleft()
            weight = 0.5 ** ((day - sale_day) / decay_half_life_days)
            self.prior_trend_weight -= weight
            self.prior_trend_total -= weight * log_price

        # Repeated floating-point subtraction can produce tiny negative values.
        self.recent_trend_weight = max(self.recent_trend_weight, 0.0)
        self.prior_trend_weight = max(self.prior_trend_weight, 0.0)

    def snapshot_trend_sales(self) -> List[Tuple[int, float]]:
        """Return the retained two-window sale history in chronological order."""
        return list(self.prior_trend_sales) + list(self.recent_trend_sales)


def _decayed_mean_std(
    state: Optional[_CellState],
    as_of_day: int,
    decay_half_life_days: float,
    default_mean: float,
    default_std: float,
) -> Tuple[float, float]:
    """Return price moments after ageing state to the prediction date."""
    if state is None or state.weight <= 0.0:
        return default_mean, default_std
    state.advance_to(as_of_day, decay_half_life_days)
    if state.weight <= 0.0:
        return default_mean, default_std
    mean = state.weighted_total / state.weight
    if state.observation_count < 2:
        return mean, default_std
    variance = max(state.weighted_total_sq / state.weight - mean * mean, 0.0)
    return mean, float(np.sqrt(variance))


def _window_trend(
    state: Optional[_CellState],
    as_of_day: int,
    trend_window_days: int,
    decay_half_life_days: float,
) -> Optional[float]:
    """Return recent-minus-prior weighted mean, or ``None`` if unsupported."""
    if state is None:
        return None
    state.advance_trend_to(as_of_day, trend_window_days, decay_half_life_days)
    if state.recent_trend_weight <= 0.0 or state.prior_trend_weight <= 0.0:
        return None
    recent_mean = state.recent_trend_total / state.recent_trend_weight
    prior_mean = state.prior_trend_total / state.prior_trend_weight
    # Incremental eviction is normally exact, but after many years of repeated
    # decay/subtraction a nearly empty window can accumulate cancellation error.
    # Sale log-prices are positive and normally around 10--16; a mean outside
    # the deliberately generous 0--20 range is numerical corruption, not a market signal. Repair
    # from the retained two-window events before returning a trend feature.
    if (
        not np.isfinite(recent_mean)
        or not np.isfinite(prior_mean)
        or recent_mean <= 0.0
        or prior_mean <= 0.0
        or recent_mean > 20.0
        or prior_mean > 20.0
        or abs(recent_mean - prior_mean) > 2.0
    ):
        def _recompute(sales):
            weights = np.asarray(
                [0.5 ** ((as_of_day - sale_day) / decay_half_life_days)
                 for sale_day, _ in sales],
                dtype=np.float64,
            )
            prices = np.asarray([price for _, price in sales], dtype=np.float64)
            return float(weights.sum()), float(np.dot(weights, prices))

        state.recent_trend_weight, state.recent_trend_total = _recompute(
            state.recent_trend_sales
        )
        state.prior_trend_weight, state.prior_trend_total = _recompute(
            state.prior_trend_sales
        )
        if state.recent_trend_weight <= 0.0 or state.prior_trend_weight <= 0.0:
            return None
        recent_mean = state.recent_trend_total / state.recent_trend_weight
        prior_mean = state.prior_trend_total / state.prior_trend_weight
    return recent_mean - prior_mean


def _cell_features(
    state: Optional[_CellState],
    global_state: _CellState,
    as_of_day: int,
    decay_half_life_days: float,
    trend_window_days: int,
    premium_shrinkage_weight: float,
    fallback_mean: float,
    fallback_std: float,
    max_recency_years: float,
    global_trend: Optional[float] = None,
) -> List[float]:
    global_mean, global_std = _decayed_mean_std(
        global_state,
        as_of_day,
        decay_half_life_days,
        fallback_mean,
        fallback_std,
    )
    if state is None or state.weight <= 0.0:
        return [0.0, global_std, 0.0, max_recency_years, 0.0]

    mean, std = _decayed_mean_std(
        state,
        as_of_day,
        decay_half_life_days,
        global_mean,
        global_std,
    )
    support = max(state.weight, 0.0)
    shrinkage = support / (support + premium_shrinkage_weight)
    premium = shrinkage * (mean - global_mean)

    local_trend = _window_trend(
        state, as_of_day, trend_window_days, decay_half_life_days
    )
    trend = 0.0 if local_trend is None else local_trend - (global_trend or 0.0)

    recency_years = min(
        max((as_of_day - state.last_sale_day) / 365.25, 0.0),
        max_recency_years,
    )
    return [premium, std, float(np.log1p(support)), recency_years, trend]


def build_historical_local_features(
    dataframe: pd.DataFrame,
    neighbor_cells_col: str = "h3_neighbor_cells",
    location_col: str = "h3_08",
    date_col: str = "sale_date",
    target_col: str = "log_price",
    decay_half_life_days: float = DEFAULT_DECAY_HALF_LIFE_DAYS,
    trend_window_days: int = DEFAULT_TREND_WINDOW_DAYS,
    premium_shrinkage_weight: float = DEFAULT_PREMIUM_SHRINKAGE_WEIGHT,
    max_recency_years: float = 5.0,
    default_log_price: float = float(np.log(300_000.0)),
    default_log_price_std: float = 0.5,
    h3_resolution: int = 8,
    snapshot_as_of_date=None,
    cache_dir=None,
    cache_dependencies=None,
    force_recompute: bool = False,
) -> Tuple[np.ndarray, Dict]:
    """Build a leakage-safe ``[rows, 7, features]`` local-market tensor.

    Rows sharing a sale date are evaluated before any of that date's sales are
    added to state, preventing both self leakage and same-day cross leakage.
    Price moments use exponential time decay, and the price-level feature is a
    shrinkage-regularised local premium relative to the contemporaneous global
    market. ``snapshot_as_of_date`` can move the snapshot beyond the newest
    sale so support, trend, and recency age correctly during deployment.
    """
    required = {neighbor_cells_col, location_col, date_col, target_col}
    missing = required.difference(dataframe.columns)
    if missing:
        raise ValueError(f"Cannot build local market features; missing columns: {sorted(missing)}")
    if decay_half_life_days <= 0:
        raise ValueError("decay_half_life_days must be positive")
    if trend_window_days <= 0:
        raise ValueError("trend_window_days must be positive")
    if premium_shrinkage_weight < 0:
        raise ValueError("premium_shrinkage_weight must be non-negative")

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
            decay_half_life_days,
            trend_window_days,
            premium_shrinkage_weight,
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
        fallback_mean, fallback_std = _decayed_mean_std(
            global_state,
            sale_day,
            decay_half_life_days,
            default_log_price,
            default_log_price_std,
        )
        global_trend = _window_trend(
            global_state, sale_day, trend_window_days, decay_half_life_days
        )
        missing_features = (
            0.0, fallback_std, 0.0, max_recency_years, 0.0
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
                            global_state,
                            sale_day,
                            decay_half_life_days,
                            trend_window_days,
                            premium_shrinkage_weight,
                            fallback_mean,
                            fallback_std,
                            max_recency_years,
                            global_trend,
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
            state.update(sale_day, log_price, decay_half_life_days, trend_window_days)
            global_state.update(sale_day, log_price, decay_half_life_days, trend_window_days)

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
    global_mean, global_std = _decayed_mean_std(
        global_state,
        snapshot_day,
        decay_half_life_days,
        default_log_price,
        default_log_price_std,
    )
    global_trend = _window_trend(
        global_state, snapshot_day, trend_window_days, decay_half_life_days
    )

    cells_snapshot = {}
    for cell, state in states.items():
        feature_values = _cell_features(
            state,
            global_state,
            snapshot_day,
            decay_half_life_days,
            trend_window_days,
            premium_shrinkage_weight,
            global_mean,
            global_std,
            max_recency_years,
            global_trend,
        )
        raw_mean, _ = _decayed_mean_std(
            state,
            snapshot_day,
            decay_half_life_days,
            global_mean,
            global_std,
        )
        cells_snapshot[cell] = {
            # Store the unshrunk premium and decayed support. Java can then
            # age the support and apply the same conservative shrinkage for a
            # future prediction date without retaining the full sale history.
            "raw_decayed_log_price_premium": raw_mean - global_mean,
            "decayed_log_price_std": feature_values[1],
            "log1p_decayed_sales_count": feature_values[2],
            "last_sale_date": _day_to_timestamp(state.last_sale_day).date().isoformat(),
            # Java recomputes the relative recent-versus-prior trend from only
            # transactions already included in this snapshot.
            "trend_sales": [
                {
                    "sale_date": _day_to_timestamp(day).date().isoformat(),
                    "log_price": log_price,
                }
                for day, log_price in state.snapshot_trend_sales()
            ],
        }

    snapshot = {
        "as_of_date": snapshot_date.date().isoformat(),
        "latest_sale_date": (
            _day_to_timestamp(int(date_days[-1])).date().isoformat()
            if n_rows else None
        ),
        "feature_version": LOCAL_MARKET_FEATURE_VERSION,
        "decay_half_life_days": decay_half_life_days,
        "trend_window_days": trend_window_days,
        "premium_shrinkage_weight": premium_shrinkage_weight,
        "max_recency_years": max_recency_years,
        "feature_order": LOCAL_MARKET_FEATURES,
        "global": {
            "decayed_mean_log_price": global_mean,
            "decayed_log_price_std": global_std,
            "log1p_decayed_sales_count": float(np.log1p(max(global_state.weight, 0.0))),
            "last_sale_date": None,
            "trend_sales": [
                {
                    "sale_date": _day_to_timestamp(day).date().isoformat(),
                    "log_price": log_price,
                }
                for day, log_price in global_state.snapshot_trend_sales()
            ],
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


def build_monthly_h3_market_snapshots(
    dataframe: pd.DataFrame,
    *,
    location_col: str = "h3_08",
    date_col: str = "sale_date",
    target_col: str = "log_price",
    cell_ids: Optional[List[str]] = None,
    decay_half_life_days: float = DEFAULT_DECAY_HALF_LIFE_DAYS,
    trend_window_days: int = DEFAULT_TREND_WINDOW_DAYS,
    premium_shrinkage_weight: float = DEFAULT_PREMIUM_SHRINKAGE_WEIGHT,
    max_recency_years: float = 5.0,
    default_log_price: float = float(np.log(300_000.0)),
    default_log_price_std: float = 0.5,
) -> Dict[str, object]:
    """Build causal H3 market-state snapshots at calendar-month boundaries.

    This is the node-state source for the standalone spatial GNN baseline.  A
    snapshot labelled ``2025-03-01`` contains only transactions dated strictly
    before 1 March 2025, so every March sale shares the same pre-month graph
    state.  Unlike :func:`build_historical_local_features`, this function
    produces one five-field state per H3 cell rather than a seven-cell tensor
    attached to each sale.

    The numerical definitions intentionally reuse the existing local-market
    implementation.  That gives the baseline the same decay, shrinkage, and
    relative-trend semantics while moving spatial aggregation into GraphSAGE.
    """
    required = {location_col, date_col, target_col}
    missing = required.difference(dataframe.columns)
    if missing:
        raise ValueError(
            "Cannot build monthly H3 snapshots; missing columns: "
            f"{sorted(missing)}"
        )
    if dataframe.empty:
        raise ValueError("Cannot build monthly H3 snapshots from an empty dataframe")
    if decay_half_life_days <= 0 or trend_window_days <= 0:
        raise ValueError("decay_half_life_days and trend_window_days must be positive")

    ordered = dataframe.copy()
    ordered[date_col] = pd.to_datetime(ordered[date_col], errors="coerce").dt.normalize()
    ordered = ordered.dropna(subset=[location_col, date_col, target_col]).sort_values(date_col)
    if ordered.empty:
        raise ValueError("No dated, located sales remain for monthly H3 snapshots")

    dates = ordered[date_col].to_numpy(dtype="datetime64[D]").astype(np.int64)
    locations = ordered[location_col].astype(str).to_numpy()
    log_prices = ordered[target_col].to_numpy(dtype=np.float64)
    if not np.isfinite(log_prices).all():
        raise ValueError("Monthly H3 snapshots require finite log prices")

    if cell_ids is None:
        cells = sorted(set(locations.tolist()))
    else:
        cells = sorted({str(cell) for cell in cell_ids if cell is not None})
        cells = sorted(set(cells).union(locations.tolist()))
    cell_index = {cell: index for index, cell in enumerate(cells)}

    first_month = ordered[date_col].iloc[0].replace(day=1)
    last_month = ordered[date_col].iloc[-1].replace(day=1)
    month_starts = pd.date_range(first_month, last_month, freq="MS")
    values = np.empty(
        (len(month_starts), len(cells), len(LOCAL_MARKET_FEATURES)),
        dtype=np.float32,
    )

    states: Dict[str, _CellState] = {}
    global_state = _CellState()
    cursor = 0
    for month_index, month_start in enumerate(month_starts):
        snapshot_day = int(np.datetime64(month_start.date(), "D").astype(np.int64))
        # Consume all strictly earlier transactions before querying every node.
        # Sales are only added after their own date; a calendar snapshot has no
        # same-day ambiguity because its boundary is the first of the month.
        while cursor < len(dates) and int(dates[cursor]) < snapshot_day:
            sale_day = int(dates[cursor])
            end = cursor + 1
            while end < len(dates) and int(dates[end]) == sale_day:
                end += 1
            for position in range(cursor, end):
                cell = locations[position]
                price = float(log_prices[position])
                states.setdefault(cell, _CellState()).update(
                    sale_day, price, decay_half_life_days, trend_window_days
                )
                global_state.update(
                    sale_day, price, decay_half_life_days, trend_window_days
                )
            cursor = end

        fallback_mean, fallback_std = _decayed_mean_std(
            global_state,
            snapshot_day,
            decay_half_life_days,
            default_log_price,
            default_log_price_std,
        )
        global_trend = _window_trend(
            global_state, snapshot_day, trend_window_days, decay_half_life_days
        )
        for cell, index in cell_index.items():
            values[month_index, index] = _cell_features(
                states.get(cell),
                global_state,
                snapshot_day,
                decay_half_life_days,
                trend_window_days,
                premium_shrinkage_weight,
                fallback_mean,
                fallback_std,
                max_recency_years,
                global_trend,
            )

    return {
        "schema_version": 1,
        "feature_version": LOCAL_MARKET_FEATURE_VERSION,
        "feature_order": list(LOCAL_MARKET_FEATURES),
        "month_starts": [month.date().isoformat() for month in month_starts],
        "cell_ids": cells,
        "features": values,
        "parameters": {
            "decay_half_life_days": decay_half_life_days,
            "trend_window_days": trend_window_days,
            "premium_shrinkage_weight": premium_shrinkage_weight,
            "max_recency_years": max_recency_years,
        },
    }
