"""Random Forest baseline using the neural network's prepared inputs."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import RandomForestRegressor

from .feature_contract import MARKET_FEATURES, PROPERTY_FEATURES, TIME_FEATURES
from .local_market_features import LOCAL_MARKET_FEATURES


PROPERTY_FEATURES = list(PROPERTY_FEATURES)
TIME_FEATURES = list(TIME_FEATURES)
MARKET_FEATURES = list(MARKET_FEATURES)
NUMERIC_FEATURES = PROPERTY_FEATURES + TIME_FEATURES + MARKET_FEATURES


class RandomForestPriceModel:
    """Leakage-safe baseline comparable to the attention model.

    Community IDs are categorical, not ordinal. The center cell is represented
    separately, while the six-neighbor IDs are represented as a multi-hot/count
    vector. This mirrors the neural model's mean pooling without assigning a
    false numeric ordering to community IDs.
    """

    def __init__(
        self,
        n_estimators=100,
        max_depth=24,
        min_samples_leaf=5,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    ):
        self.parameters = {
            "n_estimators": int(n_estimators),
            "max_depth": max_depth,
            "min_samples_leaf": int(min_samples_leaf),
            "max_features": max_features,
            "n_jobs": int(n_jobs),
            "random_state": int(random_state),
        }
        self.model = RandomForestRegressor(**self.parameters)
        self.feature_metadata = None
        self.train_indices = None
        self.val_indices = None
        self.metrics = None

    @staticmethod
    def _one_hot(values, width):
        values = np.asarray(values, dtype=np.int64)
        if np.any(values < 0) or np.any(values >= width):
            raise ValueError(f"Categorical value outside valid range [0, {width})")
        rows = np.arange(len(values), dtype=np.int64)
        return sparse.csr_matrix(
            (np.ones(len(values), dtype=np.float32), (rows, values)),
            shape=(len(values), width),
        )

    def build_feature_matrix(self, prepared_data):
        """Create a sparse matrix from an already prepared ``dataset`` object."""
        frame = prepared_data.dataframe
        required = set(NUMERIC_FEATURES + ["community_neighbors"])
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"Prepared dataframe is missing features: {missing}")

        local_features = getattr(prepared_data, "local_market_features", None)
        expected_local_shape = (len(frame), 7, len(LOCAL_MARKET_FEATURES))
        if local_features is None or np.asarray(local_features).shape != expected_local_shape:
            raise ValueError(
                "Expected leakage-safe local market features with shape "
                f"{expected_local_shape}; got "
                f"{None if local_features is None else np.asarray(local_features).shape}"
            )

        unknown_community = int(prepared_data.n_communities)
        community_width = unknown_community + 1
        neighborhoods = np.asarray(
            [
                values if isinstance(values, list) and len(values) == 7
                else [unknown_community] * 7
                for values in frame["community_neighbors"]
            ],
            dtype=np.int64,
        )
        if np.any(neighborhoods < 0) or np.any(neighborhoods >= community_width):
            raise ValueError("Community-neighbor indices do not match n_communities")

        center = self._one_hot(neighborhoods[:, 0], community_width)
        neighbor_rows = np.repeat(np.arange(len(frame), dtype=np.int64), 6)
        neighbor_values = neighborhoods[:, 1:].reshape(-1)
        neighbors = sparse.csr_matrix(
            (
                np.ones(len(neighbor_values), dtype=np.float32),
                (neighbor_rows, neighbor_values),
            ),
            shape=(len(frame), community_width),
        )

        numeric = frame[NUMERIC_FEATURES].to_numpy(dtype=np.float32)
        local = np.asarray(local_features, dtype=np.float32).reshape(len(frame), -1)
        if not np.isfinite(numeric).all() or not np.isfinite(local).all():
            raise ValueError("Random Forest inputs contain NaN or infinite values")

        matrix = sparse.hstack(
            [center, neighbors, sparse.csr_matrix(numeric), sparse.csr_matrix(local)],
            format="csr",
            dtype=np.float32,
        )
        self.feature_metadata = {
            "encoding": {
                "community_center": "one_hot",
                "community_neighbors": "six-neighbor multi_hot_counts",
                "annual_cycle": "continuous_sine_cosine",
            },
            "numeric_features": NUMERIC_FEATURES,
            "local_market_features": list(LOCAL_MARKET_FEATURES),
            "local_market_shape": [7, len(LOCAL_MARKET_FEATURES)],
            "community_width": community_width,
            "feature_count": int(matrix.shape[1]),
        }
        return matrix

    def fit(self, prepared_data, train_ratio=0.7):
        """Fit on the early chronological split and evaluate on later sales."""
        if not 0.0 < train_ratio < 1.0:
            raise ValueError("train_ratio must be between 0 and 1")
        dates = pd.to_datetime(prepared_data.dataframe["sale_date"])
        if not dates.is_monotonic_increasing:
            raise ValueError("Prepared data must be chronologically sorted")

        matrix = self.build_feature_matrix(prepared_data)
        split = int(len(prepared_data.dataframe) * train_ratio)
        self.train_indices = np.arange(split, dtype=np.int64)
        self.val_indices = np.arange(split, len(prepared_data.dataframe), dtype=np.int64)
        target = prepared_data.dataframe["log_price"].to_numpy(dtype=np.float64)

        self.model.fit(matrix[self.train_indices], target[self.train_indices])
        predicted_log_price = self.model.predict(matrix[self.val_indices])
        actual_price = np.exp(target[self.val_indices])
        predicted_price = np.exp(predicted_log_price)
        absolute_percentage_error = np.abs(predicted_price - actual_price) / actual_price * 100.0

        self.metrics = {
            "validation_mape": float(absolute_percentage_error.mean()),
            "validation_median_ape": float(np.median(absolute_percentage_error)),
            "validation_rmse": float(np.sqrt(np.mean((predicted_price - actual_price) ** 2))),
            "validation_sample_count": int(len(self.val_indices)),
            "training_sample_count": int(len(self.train_indices)),
            "train_ratio": float(train_ratio),
            "split": "chronological",
        }
        predictions = prepared_data.dataframe.iloc[self.val_indices][
            ["sale_date", "sale_price", "community"]
        ].copy()
        predictions["predicted_price"] = predicted_price
        predictions["abs_pct_error"] = absolute_percentage_error
        return predictions

    def save(self, output_dir, predictions=None):
        if self.metrics is None:
            raise RuntimeError("Fit the model before saving it")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "parameters": self.parameters,
                "feature_metadata": self.feature_metadata,
            },
            output_dir / "random_forest.joblib",
            compress=3,
        )
        payload = {
            "model": "RandomForestRegressor",
            "target": "log_price",
            "prediction_transform": "exp",
            "parameters": self.parameters,
            "features": self.feature_metadata,
            "metrics": self.metrics,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        (output_dir / "results.json").write_text(json.dumps(payload, indent=2))
        if predictions is not None:
            predictions.to_csv(output_dir / "validation_predictions.csv", index=False)
        return output_dir
