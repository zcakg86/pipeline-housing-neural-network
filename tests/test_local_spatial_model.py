from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import h3
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from pricemodel.embedding_model import EnhancedEmbeddingModel, price_predictor
from pricemodel.h3_neighbor_mapper import ordered_k_ring
from pricemodel.local_market_features import build_historical_local_features
from pricemodel.model_manager import modelmanager
from spatial.spatial_graph_detection import create_location_network
from fetch_market_indicators import join_indicators_backward_asof
from refresh_local_market_snapshot import refresh_snapshot


class MarketIndicatorCausalityTests(unittest.TestCase):
    def test_indicator_join_never_uses_a_future_observation(self):
        sales = pd.DataFrame({
            "sale_date": pd.to_datetime([
                "2024-01-15", "2024-01-05", "2024-01-20", "2024-02-01",
            ]),
        })
        indicators = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-10", "2024-01-20"]),
            "mortgage_rate": [6.5, 6.8],
            "unemployment_rate": [4.0, 4.1],
        })

        joined = join_indicators_backward_asof(sales, indicators)

        self.assertEqual(joined.loc[0, "mortgage_rate"], 6.5)
        self.assertTrue(pd.isna(joined.loc[1, "mortgage_rate"]))
        self.assertEqual(joined.loc[2, "mortgage_rate"], 6.8)
        self.assertEqual(joined.loc[3, "mortgage_rate"], 6.8)

    def test_complete_join_rejects_sales_before_indicator_history(self):
        sales = pd.DataFrame({"sale_date": pd.to_datetime(["2024-01-05"])})
        indicators = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-10"]),
            "mortgage_rate": [6.5],
            "unemployment_rate": [4.0],
        })

        with self.assertRaisesRegex(ValueError, "Future observations will not be backfilled"):
            join_indicators_backward_asof(
                sales,
                indicators,
                require_complete=True,
            )

    def test_complete_join_rejects_a_missing_source_series(self):
        sales = pd.DataFrame({
            "sale_date": pd.to_datetime(["2024-01-15"]),
            "unemployment_rate": [99.0],
        })
        indicators = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-10"]),
            "mortgage_rate": [6.5],
        })

        with self.assertRaisesRegex(ValueError, "unemployment_rate=1"):
            join_indicators_backward_asof(
                sales,
                indicators,
                require_complete=True,
            )


class LocalMarketFeatureTests(unittest.TestCase):
    def test_explicit_snapshot_date_ages_local_state(self):
        frame = pd.DataFrame({
            "sale_date": pd.to_datetime(["2024-01-01", "2024-12-01"]),
            "h3_08": ["a", "a"],
            "h3_neighbor_cells": [["a", None, None, None, None, None, None]] * 2,
            "log_price": np.log([300_000.0, 600_000.0]),
        })

        _, snapshot = build_historical_local_features(
            frame,
            snapshot_as_of_date="2025-07-01",
        )

        self.assertEqual(snapshot["as_of_date"], "2025-07-01")
        self.assertAlmostEqual(
            snapshot["cells"]["a"]["price_trend"],
            np.log(600_000.0) - np.mean(np.log([300_000.0, 600_000.0])),
            places=5,
        )
        self.assertEqual(
            snapshot["cells"]["a"]["recent_sales"],
            [{"sale_date": "2024-12-01", "log_price": np.log(600_000.0)}],
        )

    def test_snapshot_date_must_follow_all_included_sales(self):
        frame = pd.DataFrame({
            "sale_date": pd.to_datetime(["2024-01-02"]),
            "h3_08": ["a"],
            "h3_neighbor_cells": [["a", None, None, None, None, None, None]],
            "log_price": np.log([300_000.0]),
        })
        with self.assertRaisesRegex(ValueError, "strictly after every included sale"):
            build_historical_local_features(
                frame,
                snapshot_as_of_date="2024-01-02",
            )

    def test_content_addressed_cache_hits_and_invalidates(self):
        frame = pd.DataFrame({
            "sale_date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "h3_08": ["a", "a"],
            "h3_neighbor_cells": [["a", "b", None, None, None, None, None]] * 2,
            "community_neighbors": [[1, 2, 3, 3, 3, 3, 3]] * 2,
            "log_price": np.log([300_000.0, 330_000.0]),
        })
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory) / "cache"
            dependency = Path(directory) / "community_map.json"
            dependency.write_text('{"a": 1}')

            first_features, first_snapshot = build_historical_local_features(
                frame,
                cache_dir=cache_dir,
                cache_dependencies=[dependency],
            )
            self.assertIn("cache", first_snapshot)
            self.assertEqual(len(list(cache_dir.glob("*.npz"))), 1)

            output = io.StringIO()
            with redirect_stdout(output):
                second_features, second_snapshot = build_historical_local_features(
                    frame,
                    cache_dir=cache_dir,
                    cache_dependencies=[dependency],
                )
            self.assertIn("cache hit", output.getvalue().lower())
            np.testing.assert_array_equal(first_features, second_features)
            self.assertEqual(first_snapshot["cache"]["key"], second_snapshot["cache"]["key"])

            dependency.write_text('{"a": 2}')
            _, invalidated_snapshot = build_historical_local_features(
                frame,
                cache_dir=cache_dir,
                cache_dependencies=[dependency],
            )
            self.assertNotEqual(
                first_snapshot["cache"]["key"],
                invalidated_snapshot["cache"]["key"],
            )
            self.assertEqual(len(list(cache_dir.glob("*.npz"))), 2)

    def test_features_use_prior_dates_but_not_same_day_sales(self):
        frame = pd.DataFrame({
            "sale_date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"]),
            "h3_08": ["a", "a", "a"],
            "h3_neighbor_cells": [["a", "b", None, None, None, None, None]] * 3,
            "log_price": np.log([300_000.0, 330_000.0, 360_000.0]),
        })
        features, _ = build_historical_local_features(frame)

        self.assertEqual(features[0, 0, 2], 0.0)
        self.assertEqual(features[1, 0, 2], 0.0)
        self.assertAlmostEqual(features[2, 0, 2], np.log1p(2), places=5)
        self.assertAlmostEqual(
            features[2, 0, 0], np.mean(np.log([300_000.0, 330_000.0])), places=5
        )

    def test_ring_is_center_first_and_deterministic(self):
        center = h3.latlng_to_cell(47.61, -122.33, 8)
        first = ordered_k_ring(center)
        second = ordered_k_ring(center)
        self.assertEqual(first, second)
        self.assertEqual(first[0], center)
        self.assertEqual(len(first), 7)


class SnapshotRefreshTests(unittest.TestCase):
    def test_refresh_accumulates_and_deduplicates_rentcast_sales(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "data"
            deploy_dir = root / "deploy"
            data_dir.mkdir()
            historical_path = data_dir / "historical.csv"
            ledger_path = data_dir / "ledger.csv"
            rentcast_path = root / "rentcast.json"

            lat, lng = 47.61, -122.33
            center = h3.latlng_to_cell(lat, lng, 8)
            pd.DataFrame({
                "sale_date": ["2024-01-01"],
                "sale_price": [500_000],
                "lat": [lat],
                "lng": [lng],
            }).to_csv(historical_path, index=False)
            (data_dir / "community_map.json").write_text(json.dumps({center: 0}))
            rentcast_path.write_text(json.dumps({
                "properties": [{
                    "id": "property-1",
                    "lastSaleDate": "2024-02-01",
                    "lastSalePrice": 550_000,
                    "latitude": lat,
                    "longitude": lng,
                }] * 2,
            }))

            first = refresh_snapshot(
                historical_path,
                [rentcast_path],
                ledger_path,
                data_dir,
                [deploy_dir],
                "2024-03-01",
            )
            second = refresh_snapshot(
                historical_path,
                [rentcast_path],
                ledger_path,
                data_dir,
                [deploy_dir],
                "2024-03-01",
            )

            self.assertEqual(first["rentcast_ledger_rows"], 1)
            self.assertEqual(second["rentcast_ledger_rows"], 1)
            self.assertEqual(second["snapshot_sales"], 2)
            snapshot = json.loads((deploy_dir / "local_market_snapshot.json").read_text())
            self.assertEqual(snapshot["latest_sale_date"], "2024-02-01")
            self.assertEqual(snapshot["as_of_date"], "2024-02-02")
            self.assertEqual(snapshot["refresh"]["input_cutoff_date"], "2024-03-01")
            self.assertEqual(len(pd.read_csv(ledger_path)), 1)


class LocalResidualModelTests(unittest.TestCase):
    @staticmethod
    def _build_model():
        return EnhancedEmbeddingModel(
            embedding_dim=16,
            hidden_dim=32,
            property_dim=3,
            continuous_time_dim=1,
            market_dim=2,
            community_embedding_length=4,
            year_length=3,
            week_length=54,
            use_neighborhood_pooling=True,
            local_feature_dim=5,
        )

    def test_community_balanced_loss_matches_group_means(self):
        predictor = price_predictor.__new__(price_predictor)
        predictor.local_feature_dim = 5
        predictor.balance_community_loss = True
        predictor.train_community_loss_weights = torch.tensor(
            [0.0, 0.0, 0.75, 0.0, 0.0, 0.0, 0.0, 1.5]
        )
        predictor.val_community_loss_weights = None
        losses = torch.tensor([1.0, 3.0, 10.0])
        communities = torch.tensor([
            [2, 0, 0, 0, 0, 0, 0],
            [2, 0, 0, 0, 0, 0, 0],
            [7, 0, 0, 0, 0, 0, 0],
        ])
        reduced = predictor._reduce_loss(losses, communities)
        self.assertAlmostEqual(reduced.item(), 6.0)

    def test_model_exposes_gated_local_correction(self):
        torch.manual_seed(4)
        model = self._build_model()
        batch = 3
        output, components = model(
            torch.zeros(batch, 7, dtype=torch.long),
            torch.zeros(batch, dtype=torch.long),
            torch.zeros(batch, dtype=torch.long),
            torch.zeros(batch, 3),
            torch.zeros(batch, 1),
            torch.zeros(batch, 2),
            torch.randn(batch, 7, 5),
            return_components=True,
        )
        expected = (
            components["global_output"] +
            components["local_gate"] * components["local_delta"]
        )
        torch.testing.assert_close(output, expected)
        self.assertTrue(torch.all(components["local_gate"] >= 0))
        self.assertTrue(torch.all(components["local_gate"] <= 1))

    def test_global_attention_weights_are_explicitly_requested(self):
        model = self._build_model()
        batch = 3
        inputs = (
            torch.zeros(batch, 7, dtype=torch.long),
            torch.zeros(batch, dtype=torch.long),
            torch.zeros(batch, dtype=torch.long),
            torch.zeros(batch, 3),
            torch.zeros(batch, 1),
            torch.zeros(batch, 2),
            torch.randn(batch, 7, 5),
        )

        model(*inputs)
        self.assertIsNone(model.last_attention_weights)
        self.assertIsNone(model.last_cls_attention)

        model(*inputs, need_weights=True)
        self.assertEqual(tuple(model.last_attention_weights.shape), (batch, 4, 7, 7))
        self.assertEqual(tuple(model.last_cls_attention.shape), (batch, 4, 6))

    def test_local_gate_starts_conservatively(self):
        model = self._build_model()
        torch.testing.assert_close(
            model.local_confidence_gate.bias,
            torch.full_like(model.local_confidence_gate.bias, -2.0),
        )

    def test_training_restores_best_validation_state(self):
        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = torch.nn.Parameter(torch.tensor(0.0))

            def forward(
                self, community, year, week, property_feat, time_feat,
                market_feat, local_feat=None, return_uncertainty=False,
                return_components=False,
            ):
                prediction = self.bias.expand(len(community), 1)
                components = {
                    "global_output": prediction,
                    "local_delta": torch.zeros_like(prediction),
                }
                if return_components:
                    return prediction, components
                return prediction

        predictor = price_predictor.__new__(price_predictor)
        predictor.device = torch.device("cpu")
        predictor.model = DummyModel()
        predictor.optimizer = torch.optim.SGD(predictor.model.parameters(), lr=0.1)
        predictor.local_feature_dim = 0
        predictor.estimate_uncertainty = False
        predictor.balance_community_loss = False
        predictor.global_aux_weight = 0.5
        predictor.residual_penalty = 1e-2
        predictor.configure_scheduler(factor=0.5, patience=1, min_lr=1e-6)

        states = iter([1.0, 2.0, 3.0])

        def fake_train_step(_batch):
            with torch.no_grad():
                predictor.model.bias.fill_(next(states))
            return predictor.model.bias.detach().square()

        predictor.train_step = fake_train_step
        batch = (
            torch.zeros(1, 7, dtype=torch.long),
            torch.zeros(1, dtype=torch.long),
            torch.zeros(1, dtype=torch.long),
            torch.zeros(1, 3),
            torch.zeros(1, 1),
            torch.zeros(1, 2),
            torch.zeros(1),
        )

        predictor.train([batch], [batch], epochs=3, patience=2)

        self.assertEqual(predictor.best_epoch, 1)
        self.assertAlmostEqual(predictor.best_val_loss, 1.0)
        self.assertAlmostEqual(predictor.model.bias.item(), 1.0)

    def test_two_stage_training_freezes_price_during_uncertainty_calibration(self):
        torch.manual_seed(11)
        predictor = price_predictor(
            torch.device("cpu"),
            embedding_dim=8,
            hidden_dim=16,
            property_dim=3,
            continuous_time_dim=1,
            market_dim=2,
            community_embedding_length=3,
            year_length=2,
            week_length=4,
            learning_rate=1e-3,
            epochs=2,
            len_train_loader=2,
            estimate_uncertainty=True,
            use_neighborhood_pooling=True,
            local_feature_dim=0,
        )
        rows = 12
        tensors = TensorDataset(
            torch.zeros(rows, 7, dtype=torch.long),
            torch.zeros(rows, dtype=torch.long),
            torch.zeros(rows, dtype=torch.long),
            torch.randn(rows, 3),
            torch.randn(rows, 1),
            torch.randn(rows, 2),
            torch.randn(rows),
        )
        loader = DataLoader(tensors, batch_size=4)
        predictor.train_two_stage(
            loader,
            loader,
            mean_epochs=2,
            mean_patience=2,
            uncertainty_epochs=2,
            uncertainty_patience=2,
            learning_rate=1e-3,
        )

        phases = [row["phase"] for row in predictor.diagnostic_history]
        self.assertEqual(phases, ["mean", "mean", "uncertainty", "uncertainty"])
        calibration_mse = [
            row["validation"]["prediction_mse"]
            for row in predictor.diagnostic_history
            if row["phase"] == "uncertainty"
        ]
        self.assertAlmostEqual(calibration_mse[0], calibration_mse[1], places=7)
        self.assertIsNotNone(predictor.best_val_nll)

    def test_checkpoint_uses_live_pooling_strategy(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = modelmanager.__new__(modelmanager)
            manager.device = torch.device("cpu")
            manager.directory = str(Path(directory) / "model")
            manager.scalers = {}
            manager.predictor = price_predictor(
                manager.device,
                embedding_dim=8,
                hidden_dim=16,
                property_dim=3,
                continuous_time_dim=1,
                market_dim=2,
                community_embedding_length=3,
                year_length=2,
                week_length=4,
                learning_rate=1e-3,
                epochs=1,
                len_train_loader=1,
                use_neighborhood_pooling=True,
                pooling_strategy="center_weighted",
                local_feature_dim=0,
            )
            manager.results = {}
            manager.embedding_dim = 8
            manager.hidden_dim = 16
            manager.property_dim = 3
            manager.continuous_time_dim = 1
            manager.market_dim = 2
            manager.n_communities = 3
            manager.local_feature_dim = 0
            manager.dropout_rate = 0.1
            manager.epochs = 1
            manager.year_length = 2
            manager.week_length = 4
            manager.learning_rate = 1e-3
            manager.pooling_strategy = "mean"  # intentionally stale metadata
            manager.use_neighborhood_pooling = True
            manager.estimate_uncertainty = False
            manager.global_aux_weight = 0.5
            manager.residual_penalty = 1e-2
            manager.lr_plateau_factor = 0.5
            manager.lr_plateau_patience = 3
            manager.min_learning_rate = 1e-6
            manager.reference_date = None
            manager.year_vocab = {2024: 0, "unknown": 1}
            manager.week_vocab = {1: 0, 2: 1, 3: 2, "unknown": 3}
            manager.local_market_snapshot = None
            manager.neighbor_cells_map = None
            manager.train_indices = np.array([0, 1])
            manager.val_indices = np.array([2, 3])

            manager.save_and_reload_for_evaluation()

            self.assertEqual(manager.pooling_strategy, "center_weighted")
            self.assertEqual(
                manager.predictor.model.community_embedding.pooling_strategy,
                "center_weighted",
            )
            np.testing.assert_array_equal(manager.train_indices, [0, 1])
            np.testing.assert_array_equal(manager.val_indices, [2, 3])
            self.assertTrue(manager.results["evaluation_checkpoint_reloaded"])
            self.assertEqual(manager.year_vocab[2024], 0)
            self.assertEqual(manager.week_vocab[1], 0)

    def test_community_metrics_use_validation_rows_and_shrink_small_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = modelmanager.__new__(modelmanager)
            manager.directory = directory
            manager.results = {"metrics": {}}
            manager.val_indices = np.array([1, 2, 3])
            manager.dataframe = pd.DataFrame({
                # Row 0 is training data and must not affect the calculation.
                "community": [1, 1, 1, 2],
                "pct_error": [1000.0, 10.0, -20.0, 100.0],
            })

            summary = manager._calculate_validation_community_metrics(
                min_support=2,
                shrinkage_strength=2.0,
            )

            metrics = manager.results["metrics"]
            self.assertAlmostEqual(metrics["validation_mape"], 130.0 / 3.0)
            self.assertEqual(metrics["validation_sample_count"], 3)
            self.assertEqual(metrics["community_supported_count"], 1)
            self.assertAlmostEqual(
                summary.loc[1, "mean_abs_pct_error"], 15.0
            )
            self.assertAlmostEqual(
                summary.loc[2, "shrunk_mean_abs_pct_error"],
                (100.0 + 2.0 * (130.0 / 3.0)) / 3.0,
            )
            self.assertFalse(bool(summary.loc[2, "meets_min_support"]))
            self.assertTrue(
                (Path(directory) / "community_error_summary.csv").exists()
            )


class SpatialGraphTests(unittest.TestCase):
    def test_edges_include_property_and_price_similarity(self):
        center = h3.latlng_to_cell(47.61, -122.33, 8)
        cells = ordered_k_ring(center)[:3]
        frame = pd.DataFrame({
            "h3_08": cells,
            "sale_date": pd.to_datetime(["2024-01-01"] * 3),
            "price_per_sqft": [300.0, 305.0, 900.0],
            "sqft": [1500.0, 1520.0, 5000.0],
            "sqft_lot": [4000.0, 4100.0, 20000.0],
            "beds": [3, 3, 8],
        })
        graph, _, _, _ = create_location_network(
            frame, "h3_08", max_k=1, min_neighbors=1
        )
        for _, _, attributes in graph.edges(data=True):
            self.assertIn("feature_similarity", attributes)
            self.assertIn("spatial_weight", attributes)
            self.assertLessEqual(attributes["weight"], attributes["spatial_weight"])


if __name__ == "__main__":
    unittest.main()
