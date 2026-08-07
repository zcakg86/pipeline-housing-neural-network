"""Focused causality and shape tests for the independent monthly GNN baseline."""
from __future__ import annotations

import unittest

import h3
import numpy as np
import pandas as pd
import torch

from pricemodel.gnn_data import MonthlyGraphData, build_h3_edge_index
from pricemodel.gnn_network import H3GraphPriceModel
from pricemodel.gnn_trainer import GNNBaselineTrainer
from pricemodel.local_market_features import build_monthly_h3_market_snapshots


class MonthlyGnnSnapshotTests(unittest.TestCase):
    def test_month_snapshot_uses_only_sales_before_month_start(self):
        frame = pd.DataFrame(
            {
                "sale_date": pd.to_datetime(["2024-01-15", "2024-02-01"]),
                "h3_08": ["a", "a"],
                "log_price": np.log([300_000.0, 600_000.0]),
            }
        )

        snapshots = build_monthly_h3_market_snapshots(frame)
        self.assertEqual(snapshots["month_starts"], ["2024-01-01", "2024-02-01"])
        # January begins before the first sale; February sees January, but not
        # the sale occurring exactly on its own first day.
        self.assertEqual(snapshots["features"][0, 0, 2], 0.0)
        self.assertGreater(snapshots["features"][1, 0, 2], 0.0)
        self.assertAlmostEqual(
            snapshots["features"][1, 0, 2],
            np.log1p(0.5 ** (17 / 730.0)),
            places=6,
        )

    def test_h3_topology_only_connects_observed_graph_cells(self):
        center = h3.latlng_to_cell(47.61, -122.33, 8)
        neighbour = next(iter(h3.grid_ring(center, 1)))
        edge_index = build_h3_edge_index([center, neighbour])
        self.assertEqual(edge_index.shape, (2, 2))
        self.assertEqual({tuple(edge) for edge in edge_index.T.tolist()}, {(0, 1), (1, 0)})


class GraphNetworkTests(unittest.TestCase):
    def test_network_returns_one_log_price_per_sale(self):
        model = H3GraphPriceModel(
            node_feature_dim=5,
            property_dim=4,
            time_dim=3,
            market_dim=2,
            graph_hidden_dim=8,
            head_hidden_dim=16,
        )
        prediction = model(
            torch.randn(3, 5),
            torch.tensor([[0, 1, 2, 1], [1, 0, 1, 2]]),
            torch.tensor([0, 2]),
            torch.randn(2, 4),
            torch.randn(2, 3),
            torch.randn(2, 2),
        )
        self.assertEqual(tuple(prediction.shape), (2, 1))
        self.assertTrue(torch.isfinite(prediction).all())

    def test_trainer_uses_month_grouped_graph_batches(self):
        frame = pd.DataFrame(
            {
                "sale_date": pd.date_range("2024-01-01", periods=5, freq="MS"),
                "h3_08": ["a", "b", "a", "b", "a"],
                "sale_price": [300_000, 310_000, 320_000, 330_000, 340_000],
                "log_price": np.log([300_000, 310_000, 320_000, 330_000, 340_000]),
                "sqft": [1000, 1010, 1020, 1030, 1040],
                "sqft_lot": [3000, 3010, 3020, 3030, 3040],
                "beds": [2, 2, 3, 3, 3],
                "water_proximity": [0.0, 0.1, 0.0, 0.2, 0.1],
                "time_trend": [0, 1 / 12, 2 / 12, 3 / 12, 4 / 12],
                "annual_sin": [0.0, 0.5, 0.9, 1.0, 0.8],
                "annual_cos": [1.0, 0.8, 0.4, 0.0, -0.4],
                "mortgage_rate": [6.0] * 5,
                "unemployment_rate": [4.0] * 5,
            }
        )
        graph = MonthlyGraphData(
            dataframe=frame,
            cell_ids=("a", "b"),
            month_starts=tuple(frame["sale_date"].dt.date.astype(str)),
            edge_index=torch.tensor([[0, 1], [1, 0]]),
            node_features=np.ones((5, 2, 5), dtype=np.float32),
            sale_node_index=np.asarray([0, 1, 0, 1, 0]),
            sale_month_index=np.arange(5),
        )
        trainer = GNNBaselineTrainer(
            graph, graph_hidden_dim=4, head_hidden_dim=8, random_seed=3
        ).fit(train_ratio=0.6, epochs=2, patience=2)
        reported = trainer.add_predictions_and_metrics()
        explanation = trainer.explain_prediction(0, coalitions=32)
        self.assertIn(trainer.results["best_epoch"], {1, 2})
        self.assertEqual(len(trainer.results["val_losses"]), 2)
        self.assertIn("whole_dataset_mape", trainer.results["metrics"])
        self.assertIn("predicted_price", reported.columns)
        self.assertEqual(len(explanation.groups), 5)
        self.assertEqual(len(explanation.features), 24)
        self.assertAlmostEqual(
            explanation.predicted_log_price,
            explanation.reference_log_price + sum(
                effect.log_contribution for effect in explanation.groups
            ),
            places=5,
        )


if __name__ == "__main__":
    unittest.main()
