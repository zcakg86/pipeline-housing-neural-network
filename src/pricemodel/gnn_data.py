"""Leakage-safe monthly graph data for the standalone H3 price GNN.

This module deliberately does not depend on community assignment or the
existing seven-cell local-market tensor.  It reuses the canonical property,
time, economic, water, and local-market feature definitions instead.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Iterable

import h3
import numpy as np
import pandas as pd
import torch

from .feature_contract import MARKET_FEATURES, PROPERTY_FEATURES, TIME_FEATURES
from .local_market_features import (
    LOCAL_MARKET_FEATURES,
    build_monthly_h3_market_snapshots,
)
from .market_indicators import join_indicators_backward_asof
from .water_features import DEFAULT_WATER_PATH, add_water_proximity_features


@dataclass(frozen=True)
class MonthlyGraphData:
    """Graph topology, dated node state, and sale-to-node/month lookups."""

    dataframe: pd.DataFrame
    cell_ids: tuple[str, ...]
    month_starts: tuple[str, ...]
    edge_index: torch.Tensor
    node_features: np.ndarray
    sale_node_index: np.ndarray
    sale_month_index: np.ndarray
    sale_pool_node_index: np.ndarray | None = None
    sale_pool_weights: np.ndarray | None = None

    @property
    def node_feature_names(self) -> tuple[str, ...]:
        return tuple(LOCAL_MARKET_FEATURES)


def prepare_gnn_sales_dataframe(
    dataframe: pd.DataFrame,
    *,
    market_indicator_cache_path: str | Path,
    water_geojson_path: str | Path = DEFAULT_WATER_PATH,
    water_feature_cache_dir: str | Path | None = None,
    community_map_path: str | Path = "data/community_map.json",
) -> pd.DataFrame:
    """Return the clean shared sale features needed by the GNN baseline.

    This is intentionally equivalent to the non-community portions of
    :class:`DatasetBuilder`: temporal/economic joins are backward-as-of and
    water proximity is calculated with the common spatial utility.
    """
    required = {"sale_price", "sale_date", "lat", "lng", "sqft", "sqft_lot", "beds"}
    missing = required.difference(dataframe.columns)
    if missing:
        raise ValueError(f"GNN sales data is missing columns: {sorted(missing)}")
    frame = dataframe.copy()
    frame["sale_date"] = pd.to_datetime(frame["sale_date"], errors="coerce")
    numeric = ["sale_price", "lat", "lng", "sqft", "sqft_lot", "beds"]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["sale_date", *numeric])
    frame = frame[
        (frame["sale_price"] > 0)
        & (frame["sqft"] > 0)
        & (frame["sqft_lot"] > 0)
        & (frame["beds"] > 0)
    ].copy()
    if frame.empty:
        raise ValueError("No valid sales remain for GNN training")

    if "h3_08" not in frame.columns:
        frame["h3_08"] = [
            h3.latlng_to_cell(lat, lng, 8)
            for lat, lng in zip(frame["lat"], frame["lng"])
        ]
    else:
        missing_h3 = frame["h3_08"].isna()
        if missing_h3.any():
            frame.loc[missing_h3, "h3_08"] = [
                h3.latlng_to_cell(lat, lng, 8)
                for lat, lng in zip(
                    frame.loc[missing_h3, "lat"], frame.loc[missing_h3, "lng"]
                )
            ]

    indicator_path = Path(market_indicator_cache_path)
    if not indicator_path.is_file():
        raise FileNotFoundError(f"Frozen market indicators not found: {indicator_path}")
    indicators = pd.read_csv(indicator_path, index_col="date", parse_dates=True).sort_index()
    frame = join_indicators_backward_asof(frame, indicators, require_complete=True)
    frame = frame.sort_values("sale_date").reset_index(drop=True)
    frame = add_water_proximity_features(
        frame,
        water_path=water_geojson_path,
        cache_dir=water_feature_cache_dir,
    )
    reference_date = frame["sale_date"].min()
    frame["log_price"] = np.log(frame["sale_price"])
    frame["time_trend"] = (frame["sale_date"] - reference_date).dt.days / 365.25
    day_index = frame["sale_date"].dt.dayofyear.to_numpy(dtype=np.float64) - 1.0
    days_in_year = np.where(frame["sale_date"].dt.is_leap_year.to_numpy(), 366.0, 365.0)
    phase = 2.0 * np.pi * day_index / days_in_year
    frame["annual_sin"] = np.sin(phase)
    frame["annual_cos"] = np.cos(phase)

    # Communities remain evaluation labels only.  They are deliberately not
    # returned as a model tensor and therefore cannot become an identity input
    # for this no-community-embedding GNN baseline.
    community_path = Path(community_map_path)
    if community_path.is_file():
        community_map = json.loads(community_path.read_text(encoding="utf-8"))
        frame["community"] = frame["h3_08"].map(community_map)
    else:
        frame["community"] = np.nan
    return frame


def build_h3_edge_index(cell_ids: Iterable[str]) -> torch.Tensor:
    """Create directed one-ring H3 edges, without adding artificial cells."""
    cells = tuple(cell_ids)
    index = {cell: position for position, cell in enumerate(cells)}
    sources: list[int] = []
    targets: list[int] = []
    for cell, target in index.items():
        for neighbour in h3.grid_ring(cell, 1):
            source = index.get(neighbour)
            if source is not None:
                sources.append(source)
                targets.append(target)
    if not sources:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor([sources, targets], dtype=torch.long)


def _haversine_km(
    latitude: float,
    longitude: float,
    candidate_latitudes: np.ndarray,
    candidate_longitudes: np.ndarray,
) -> np.ndarray:
    """Return great-circle distances from one point to candidate coordinates."""
    earth_radius_km = 6371.0088
    latitude_radians = np.radians(latitude)
    candidate_latitude_radians = np.radians(candidate_latitudes)
    latitude_delta = candidate_latitude_radians - latitude_radians
    longitude_delta = np.radians(candidate_longitudes - longitude)
    haversine = (
        np.sin(latitude_delta / 2.0) ** 2
        + np.cos(latitude_radians)
        * np.cos(candidate_latitude_radians)
        * np.sin(longitude_delta / 2.0) ** 2
    )
    return 2.0 * earth_radius_km * np.arcsin(np.sqrt(np.clip(haversine, 0.0, 1.0)))


def build_continuous_embedding_pooling(
    frame: pd.DataFrame,
    cell_ids: Iterable[str],
    *,
    radius_km: float = 1.5,
    h3_ring: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Build compact-support spatial interpolation weights for every sale.

    A Wendland C2 kernel reaches zero smoothly at ``radius_km``.  Candidate
    discovery uses the same H3 disk operation as Java deployment, while the
    weights depend on the property's continuous latitude/longitude.
    """
    if radius_km <= 0:
        raise ValueError("Continuous embedding pooling radius must be positive")
    if h3_ring < 1:
        raise ValueError("Continuous embedding pooling H3 ring must be positive")
    cells = tuple(cell_ids)
    cell_index = {cell: position for position, cell in enumerate(cells)}
    centers = np.asarray([h3.cell_to_latlng(cell) for cell in cells], dtype=np.float64)
    candidate_cache: dict[str, np.ndarray] = {}
    row_indices: list[np.ndarray] = []
    row_weights: list[np.ndarray] = []

    for cell, latitude, longitude in zip(frame["h3_08"], frame["lat"], frame["lng"]):
        candidates = candidate_cache.get(cell)
        if candidates is None:
            candidates = np.asarray(
                [cell_index[value] for value in h3.grid_disk(cell, h3_ring) if value in cell_index],
                dtype=np.int64,
            )
            candidate_cache[cell] = candidates
        distances = _haversine_km(
            float(latitude), float(longitude), centers[candidates, 0], centers[candidates, 1]
        )
        keep = distances < radius_km
        selected = candidates[keep]
        selected_distances = distances[keep]
        if not len(selected):
            selected = np.asarray([cell_index[cell]], dtype=np.int64)
            weights = np.ones(1, dtype=np.float64)
        else:
            scaled = selected_distances / radius_km
            weights = (1.0 - scaled) ** 4 * (4.0 * scaled + 1.0)
            weights /= weights.sum()
        row_indices.append(selected)
        row_weights.append(weights.astype(np.float32))

    width = max(map(len, row_indices))
    pooled_indices = np.zeros((len(frame), width), dtype=np.int64)
    pooled_weights = np.zeros((len(frame), width), dtype=np.float32)
    for row, (indices, weights) in enumerate(zip(row_indices, row_weights)):
        pooled_indices[row, :len(indices)] = indices
        pooled_weights[row, :len(weights)] = weights
    return pooled_indices, pooled_weights


def build_monthly_graph_data(
    frame: pd.DataFrame,
    *,
    continuous_embedding_pooling: bool = False,
    spatial_pool_radius_km: float = 1.5,
    spatial_pool_h3_ring: int = 3,
) -> MonthlyGraphData:
    """Build all calendar-month snapshots and sale lookup indexes for training."""
    snapshots = build_monthly_h3_market_snapshots(frame)
    cell_ids = tuple(snapshots["cell_ids"])
    month_starts = tuple(snapshots["month_starts"])
    cell_index = {cell: index for index, cell in enumerate(cell_ids)}
    month_index = {month: index for index, month in enumerate(month_starts)}
    month_labels = frame["sale_date"].dt.to_period("M").dt.to_timestamp().dt.date.astype(str)
    try:
        sale_node_index = np.asarray([cell_index[cell] for cell in frame["h3_08"]], dtype=np.int64)
        sale_month_index = np.asarray([month_index[month] for month in month_labels], dtype=np.int64)
    except KeyError as exc:
        raise RuntimeError(f"GNN graph lookup was unexpectedly incomplete: {exc}") from exc
    sale_pool_node_index = None
    sale_pool_weights = None
    if continuous_embedding_pooling:
        sale_pool_node_index, sale_pool_weights = build_continuous_embedding_pooling(
            frame,
            cell_ids,
            radius_km=spatial_pool_radius_km,
            h3_ring=spatial_pool_h3_ring,
        )
    return MonthlyGraphData(
        dataframe=frame,
        cell_ids=cell_ids,
        month_starts=month_starts,
        edge_index=build_h3_edge_index(cell_ids),
        node_features=np.asarray(snapshots["features"], dtype=np.float32),
        sale_node_index=sale_node_index,
        sale_month_index=sale_month_index,
        sale_pool_node_index=sale_pool_node_index,
        sale_pool_weights=sale_pool_weights,
    )


def shared_sale_feature_names() -> Dict[str, tuple[str, ...]]:
    """Expose the inherited model inputs for artifact metadata and tests."""
    return {
        "property": tuple(PROPERTY_FEATURES),
        "time": tuple(TIME_FEATURES),
        "market": tuple(MARKET_FEATURES),
    }
