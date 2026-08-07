import unittest

import numpy as np
import pandas as pd

from pricemodel.local_market_features import LOCAL_MARKET_FEATURES
from pricemodel.random_forest_model import NUMERIC_FEATURES, RandomForestPriceModel


class PreparedData:
    def __init__(self, rows=20):
        self.n_communities = 3
        dates = pd.date_range("2023-01-01", periods=rows, freq="30D")
        phase = 2 * np.pi * (dates.dayofyear - 1) / np.where(dates.is_leap_year, 366, 365)
        prices = np.linspace(400_000, 700_000, rows)
        self.dataframe = pd.DataFrame({
            "sale_date": dates,
            "sale_price": prices,
            "log_price": np.log(prices),
            "community": np.arange(rows) % 3,
            "community_neighbors": [[i % 3, (i + 1) % 3, (i + 2) % 3, 3, 3, 3, 3] for i in range(rows)],
            "sqft": np.linspace(1000, 2200, rows),
            "sqft_lot": np.linspace(3000, 8000, rows),
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


class RandomForestPriceModelTests(unittest.TestCase):
    def test_matrix_contains_all_neural_model_inputs(self):
        data = PreparedData()
        model = RandomForestPriceModel(n_estimators=2, n_jobs=1)
        matrix = model.build_feature_matrix(data)
        expected = (
            (data.n_communities + 1) * 2
            + len(NUMERIC_FEATURES)
            + 7 * len(LOCAL_MARKET_FEATURES)
        )
        self.assertEqual(matrix.shape, (len(data.dataframe), expected))
        self.assertEqual(model.feature_metadata["feature_count"], expected)

    def test_fit_uses_chronological_holdout_and_returns_metrics(self):
        data = PreparedData()
        model = RandomForestPriceModel(n_estimators=4, max_depth=3, n_jobs=1)
        predictions = model.fit(data, train_ratio=0.7)
        np.testing.assert_array_equal(model.train_indices, np.arange(14))
        np.testing.assert_array_equal(model.val_indices, np.arange(14, 20))
        self.assertEqual(len(predictions), 6)
        self.assertIn("validation_mape", model.metrics)


if __name__ == "__main__":
    unittest.main()
