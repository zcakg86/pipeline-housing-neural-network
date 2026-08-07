"""LightGBM baseline using the neural network's prepared inputs."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
# On macOS, the project PyTorch and LightGBM wheels both use OpenMP. Loading
# PyTorch first avoids a native-runtime termination if both models are imported
# in the same process (as they are during test discovery and model comparison).
import torch  # noqa: F401
import lightgbm as lgb
import numpy as np
import pandas as pd

from .feature_contract import (
    LOCAL_FEATURES,
    MARKET_FEATURES,
    PROPERTY_FEATURES,
    RING_NAMES,
    TIME_FEATURES,
)


LOCAL_MARKET_FEATURES = list(LOCAL_FEATURES)
NUMERIC_FEATURES = list(PROPERTY_FEATURES + TIME_FEATURES + MARKET_FEATURES)


class LightGBMPriceModel:
    """Gradient-boosted tree baseline with native categorical features."""

    def __init__(
        self,
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=100,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        n_jobs=-1,
        random_state=42,
    ):
        self.parameters = {
            "objective": "regression_l1",
            "metric": "None",
            "n_estimators": int(n_estimators),
            "learning_rate": float(learning_rate),
            "num_leaves": int(num_leaves),
            "max_depth": int(max_depth),
            "min_child_samples": int(min_child_samples),
            "subsample": float(subsample),
            "subsample_freq": 1,
            "colsample_bytree": float(colsample_bytree),
            "reg_lambda": float(reg_lambda),
            "n_jobs": int(n_jobs),
            "random_state": int(random_state),
            "verbosity": -1,
        }
        self.model = None
        self.feature_metadata = None
        self.categorical_features = None
        self.train_indices = None
        self.val_indices = None
        self.metrics = None
        self.best_iteration = None
        self.feature_importances = None

    @staticmethod
    def _price_mape(y_true, y_pred):
        actual = np.exp(np.clip(y_true, -50, 50))
        predicted = np.exp(np.clip(y_pred, -50, 50))
        value = np.mean(np.abs(predicted - actual) / actual) * 100.0
        return "price_mape", float(value), False

    def build_feature_frame(self, prepared_data):
        """Build a DataFrame with the same information used by the neural model."""
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
                values if isinstance(values, (list, tuple, np.ndarray)) and len(values) == 7
                else [unknown_community] * 7
                for values in frame["community_neighbors"]
            ],
            dtype=np.int32,
        )
        if np.any(neighborhoods < 0) or np.any(neighborhoods >= community_width):
            raise ValueError("Community-neighbor indices do not match n_communities")

        features = pd.DataFrame(index=frame.index)
        community_columns = []
        for position, ring_name in enumerate(RING_NAMES):
            name = f"community_{ring_name}"
            features[name] = pd.Categorical(
                neighborhoods[:, position], categories=range(community_width)
            )
            community_columns.append(name)

        numeric = frame[NUMERIC_FEATURES].to_numpy(dtype=np.float32)
        local = np.asarray(local_features, dtype=np.float32)
        if not np.isfinite(numeric).all() or not np.isfinite(local).all():
            raise ValueError("LightGBM inputs contain NaN or infinite values")
        features[NUMERIC_FEATURES] = numeric

        local_columns = []
        for ring_position, ring_name in enumerate(RING_NAMES):
            for feature_position, feature_name in enumerate(LOCAL_MARKET_FEATURES):
                name = f"{ring_name}_{feature_name}"
                features[name] = local[:, ring_position, feature_position]
                local_columns.append(name)

        self.categorical_features = community_columns
        self.feature_metadata = {
            "encoding": {
                "community_center_and_neighbors": "native_categorical_7_columns",
                "annual_cycle": "continuous_sine_cosine",
            },
            "categorical_features": self.categorical_features,
            "numeric_features": NUMERIC_FEATURES,
            "local_market_features": list(LOCAL_MARKET_FEATURES),
            "local_market_columns": local_columns,
            "local_market_shape": [7, len(LOCAL_MARKET_FEATURES)],
            "feature_count": int(features.shape[1]),
        }
        return features

    def fit(
        self,
        prepared_data,
        train_ratio=0.7,
        selection_fraction=0.15,
        early_stopping_rounds=100,
        log_period=50,
    ):
        """Select iterations chronologically, refit, and evaluate once."""
        if not 0.0 < train_ratio < 1.0:
            raise ValueError("train_ratio must be between 0 and 1")
        if not 0.0 < selection_fraction < 0.5:
            raise ValueError("selection_fraction must be between 0 and 0.5")
        dates = pd.to_datetime(prepared_data.dataframe["sale_date"])
        if not dates.is_monotonic_increasing:
            raise ValueError("Prepared data must be chronologically sorted")

        features = self.build_feature_frame(prepared_data)
        target = prepared_data.dataframe["log_price"].to_numpy(dtype=np.float64)
        split = int(len(features) * train_ratio)
        selection_size = max(1, int(split * selection_fraction))
        selection_start = split - selection_size
        if selection_start < 2:
            raise ValueError("Not enough training rows for chronological model selection")

        self.train_indices = np.arange(split, dtype=np.int64)
        self.val_indices = np.arange(split, len(features), dtype=np.int64)

        selection_model = lgb.LGBMRegressor(**self.parameters)
        selection_model.fit(
            features.iloc[:selection_start],
            target[:selection_start],
            categorical_feature=self.categorical_features,
            eval_set=[
                (
                    features.iloc[selection_start:split],
                    target[selection_start:split],
                )
            ],
            eval_metric=self._price_mape,
            callbacks=[
                lgb.early_stopping(
                    int(early_stopping_rounds), first_metric_only=True, verbose=True
                ),
                lgb.log_evaluation(period=int(log_period)),
            ],
        )
        self.best_iteration = int(
            selection_model.best_iteration_ or self.parameters["n_estimators"]
        )

        final_parameters = dict(self.parameters)
        final_parameters["n_estimators"] = self.best_iteration
        self.model = lgb.LGBMRegressor(**final_parameters)
        self.model.fit(
            features.iloc[self.train_indices],
            target[self.train_indices],
            categorical_feature=self.categorical_features,
        )

        predicted_log_price = self.model.predict(features.iloc[self.val_indices])
        actual_price = np.exp(target[self.val_indices])
        predicted_price = np.exp(predicted_log_price)
        absolute_percentage_error = (
            np.abs(predicted_price - actual_price) / actual_price * 100.0
        )
        signed_percentage_error = (
            (predicted_price - actual_price) / actual_price * 100.0
        )

        self.metrics = {
            "validation_mape": float(absolute_percentage_error.mean()),
            "validation_median_ape": float(np.median(absolute_percentage_error)),
            "validation_rmse": float(
                np.sqrt(np.mean((predicted_price - actual_price) ** 2))
            ),
            "validation_mean_signed_percentage_error": float(
                signed_percentage_error.mean()
            ),
            "validation_sample_count": int(len(self.val_indices)),
            "training_sample_count": int(len(self.train_indices)),
            "model_selection_training_count": int(selection_start),
            "model_selection_validation_count": int(selection_size),
            "train_ratio": float(train_ratio),
            "split": "chronological",
            "best_iteration": self.best_iteration,
        }
        predictions = prepared_data.dataframe.iloc[self.val_indices][
            ["sale_date", "sale_price", "community"]
        ].copy()
        predictions["predicted_price"] = predicted_price
        predictions["abs_pct_error"] = absolute_percentage_error
        predictions["signed_pct_error"] = signed_percentage_error

        importance = pd.DataFrame({
            "feature": self.model.feature_name_,
            "split_importance": self.model.feature_importances_,
            "gain_importance": self.model.booster_.feature_importance(
                importance_type="gain"
            ),
        }).sort_values("gain_importance", ascending=False)
        gain_total = importance["gain_importance"].sum()
        importance["gain_fraction"] = (
            importance["gain_importance"] / gain_total if gain_total else 0.0
        )
        self.feature_importances = importance.reset_index(drop=True)
        return predictions

    def save(self, output_dir, predictions=None):
        if self.metrics is None or self.model is None:
            raise RuntimeError("Fit the model before saving it")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "parameters": self.parameters,
                "best_iteration": self.best_iteration,
                "feature_metadata": self.feature_metadata,
            },
            output_dir / "lightgbm.joblib",
            compress=3,
        )
        self.model.booster_.save_model(output_dir / "lightgbm_model.txt")
        self.feature_importances.to_csv(
            output_dir / "feature_importance.csv", index=False
        )
        payload = {
            "model": "LGBMRegressor",
            "lightgbm_version": lgb.__version__,
            "target": "log_price",
            "prediction_transform": "exp",
            "parameters": self.parameters,
            "best_iteration": self.best_iteration,
            "features": self.feature_metadata,
            "metrics": self.metrics,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        (output_dir / "results.json").write_text(json.dumps(payload, indent=2))
        if predictions is not None:
            predictions.to_csv(
                output_dir / "validation_predictions.csv", index=False
            )
        return output_dir
