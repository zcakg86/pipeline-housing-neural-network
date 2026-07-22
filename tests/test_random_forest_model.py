import unittest

import numpy as np
import pandas as pd

from pricemodel.local_market_features import LOCAL_MARKET_FEATURES
from pricemodel.random_forest_model import RandomForestPriceModel


class PreparedData:
    def __init__(self, rows=20):
        self.n_communities = 3
        self.year_vocab = {2023: 0, 2024: 1, "unknown": 2}
        self.week_vocab = {week: week - 1 for week in range(1, 54)}
        self.week_vocab["unknown"] = 53
        self.year_length = len(self.year_vocab)
        self.week_length = len(self.week_vocab)
        dates = pd.date_range("2023-01-01", periods=rows, freq="30D")
        prices = np.linspace(400_000, 700_000, rows)
        self.dataframe = pd.DataFrame({
            "sale_date": dates,
            "sale_price": prices,
            "log_price": np.log(prices),
            "community": np.arange(rows) % 3,
            "community_neighbors": [[i % 3, (i + 1) % 3, (i + 2) % 3, 3, 3, 3, 3] for i in range(rows)],
            "year": dates.isocalendar().year,
            "week": dates.isocalendar().week,
            "sqft": np.linspace(1000, 2200, rows),
            "sqft_lot": np.linspace(3000, 8000, rows),
            "beds": np.where(np.arange(rows) % 2, 3, 4),
            "time_trend": np.arange(rows) / 12,
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
        expected = (data.n_communities + 1) * 2 + data.year_length + data.week_length + 6 + 35
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
