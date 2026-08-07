import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from pricemodel.lightgbm_model import LightGBMPriceModel
from pricemodel.feature_contract import lightgbm_feature_names
from pricemodel.local_market_features import LOCAL_MARKET_FEATURES


class PreparedData:
    def __init__(self, rows=40):
        self.n_communities = 3
        dates = pd.date_range("2023-01-01", periods=rows, freq="20D")
        phase = 2 * np.pi * (dates.dayofyear - 1) / np.where(dates.is_leap_year, 366, 365)
        prices = np.linspace(400_000, 800_000, rows)
        self.dataframe = pd.DataFrame({
            "sale_date": dates,
            "sale_price": prices,
            "log_price": np.log(prices),
            "community": np.arange(rows) % 3,
            "community_neighbors": [
                [i % 3, (i + 1) % 3, (i + 2) % 3, 3, 3, 3, 3]
                for i in range(rows)
            ],
            "sqft": np.linspace(1000, 2400, rows),
            "sqft_lot": np.linspace(3000, 9000, rows),
            "beds": np.where(np.arange(rows) % 2, 3, 4),
            "water_proximity": np.exp(-np.linspace(10, 5000, rows) / 100.0),
            "time_trend": np.arange(rows) / 12,
            "annual_sin": np.sin(phase),
            "annual_cos": np.cos(phase),
            "mortgage_rate": np.linspace(5, 7, rows),
            "unemployment_rate": np.linspace(3.5, 4.5, rows),
        })
        self.local_market_features = np.zeros(
            (rows, 7, len(LOCAL_MARKET_FEATURES)), dtype=np.float32
        )


class LightGBMPriceModelTests(unittest.TestCase):
    def test_feature_frame_contains_all_neural_inputs(self):
        data = PreparedData()
        model = LightGBMPriceModel(n_estimators=5, n_jobs=1)
        features = model.build_feature_frame(data)
        self.assertEqual(
            features.shape, (len(data.dataframe), len(lightgbm_feature_names()))
        )
        self.assertEqual(len(model.categorical_features), 7)
        for name in model.categorical_features:
            self.assertEqual(str(features[name].dtype), "category")

    def test_fit_uses_final_chronological_holdout_and_saves_artifacts(self):
        data = PreparedData()
        model = LightGBMPriceModel(
            n_estimators=10,
            learning_rate=0.1,
            num_leaves=7,
            min_child_samples=2,
            n_jobs=1,
        )
        predictions = model.fit(
            data,
            train_ratio=0.7,
            selection_fraction=0.2,
            early_stopping_rounds=3,
            log_period=0,
        )
        np.testing.assert_array_equal(model.train_indices, np.arange(28))
        np.testing.assert_array_equal(model.val_indices, np.arange(28, 40))
        self.assertEqual(len(predictions), 12)
        self.assertIn("validation_mape", model.metrics)

        with tempfile.TemporaryDirectory() as directory:
            output = model.save(directory, predictions)
            self.assertTrue((Path(output) / "lightgbm.joblib").exists())
            self.assertTrue((Path(output) / "lightgbm_model.txt").exists())
            self.assertTrue((Path(output) / "validation_predictions.csv").exists())


if __name__ == "__main__":
    unittest.main()
