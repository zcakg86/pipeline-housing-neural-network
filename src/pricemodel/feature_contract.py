"""Canonical feature ordering and tensor metadata shared by model families.

The Python training/export paths import these constants directly. Deployment
exports serialize :data:`FEATURE_CONTRACT` so Java can validate its ONNX and
TreeSHAP adapters before serving predictions.
"""

from __future__ import annotations

import json
from pathlib import Path

from .local_market_features import LOCAL_MARKET_FEATURES


FEATURE_CONTRACT_VERSION = 4
NEIGHBOR_COUNT = 7
PROPERTY_FEATURES = ("sqft", "sqft_lot", "beds", "water_proximity")
TIME_FEATURES = ("time_trend", "annual_sin", "annual_cos")
MARKET_FEATURES = ("mortgage_rate", "unemployment_rate")
LOCAL_FEATURES = tuple(LOCAL_MARKET_FEATURES)
RING_NAMES = ("center",) + tuple(
    f"neighbor_{index}" for index in range(1, NEIGHBOR_COUNT)
)
DEFAULT_MARKET_VALUES = {
    "mortgage_rate": 6.5,
    "unemployment_rate": 4.0,
}


def lightgbm_feature_names() -> tuple[str, ...]:
    """Return the exact ordered feature vector consumed by LightGBM."""
    categorical = ("community_center",) + tuple(
        f"community_neighbor_{index}" for index in range(1, NEIGHBOR_COUNT)
    )
    local = tuple(
        f"{ring}_{feature}"
        for ring in RING_NAMES
        for feature in LOCAL_FEATURES
    )
    return categorical + PROPERTY_FEATURES + TIME_FEATURES + MARKET_FEATURES + local


FEATURE_CONTRACT = {
    "version": FEATURE_CONTRACT_VERSION,
    "neighbor_count": NEIGHBOR_COUNT,
    "defaults": DEFAULT_MARKET_VALUES,
    "groups": {
        "property": list(PROPERTY_FEATURES),
        "time": list(TIME_FEATURES),
        "market": list(MARKET_FEATURES),
        "local_market": list(LOCAL_FEATURES),
        "rings": list(RING_NAMES),
    },
    "neural_inputs": {
        "community_indices": ["batch", NEIGHBOR_COUNT],
        "property_features": ["batch", len(PROPERTY_FEATURES)],
        "time_features": ["batch", len(TIME_FEATURES)],
        "market_features": ["batch", len(MARKET_FEATURES)],
        "local_market_features": ["batch", NEIGHBOR_COUNT, len(LOCAL_FEATURES)],
    },
    "lightgbm_feature_names": list(lightgbm_feature_names()),
    "display_units": {
        "sqft": "sqft",
        "sqft_lot": "sqft",
        "beds": "count",
        "water_proximity": "score_0_to_1",
        "time_trend": "years",
        "annual_sin": "unitless",
        "annual_cos": "unitless",
        "mortgage_rate": "percent",
        "unemployment_rate": "percent",
        "local_decayed_log_price_premium": "log_dollars_relative_to_global_market",
        "local_decayed_log_price_std": "log_dollars",
        "local_log1p_decayed_sales_count": "log_effective_count",
        "local_recency_years": "years",
        "local_relative_price_trend": "log_dollars_recent_minus_prior_relative_to_global",
    },
}


def write_feature_contract(path: str | Path) -> Path:
    """Serialize the canonical contract deterministically for deployment."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(FEATURE_CONTRACT, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination
