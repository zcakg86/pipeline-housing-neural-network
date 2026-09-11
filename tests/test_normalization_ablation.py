"""Architecture-switch and run-manifest tests for normalization ablations."""
from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from pricemodel.gnn_network import H3GraphPriceModel
from pricemodel.gnn_training_config import GNNTrainingConfig
from pricemodel.network import EnhancedEmbeddingModel
from pricemodel.model_manager import ModelManager
from pricemodel.run_manifest import build_run_manifest, write_run_manifest
from pricemodel.training_config import TrainingConfig


class NormalizationArchitectureTests(unittest.TestCase):
    def test_attention_supports_all_ablation_combinations(self):
        for layer_norm in (False, True):
            for residual in (False, True):
                model = EnhancedEmbeddingModel(
                    embedding_dim=8,
                    hidden_dim=16,
                    property_dim=4,
                    continuous_time_dim=3,
                    market_dim=2,
                    community_embedding_length=3,
                    community_embedding_dim=4,
                    use_neighborhood_pooling=False,
                    attention_layer_norm=layer_norm,
                    attention_residual=residual,
                ).eval()
                output = model(
                    torch.tensor([0, 1]),
                    torch.randn(2, 4),
                    torch.randn(2, 3),
                    torch.randn(2, 2),
                )
                self.assertEqual(tuple(output.shape), (2, 1))
                self.assertEqual(hasattr(model, "attention_output_norm"), layer_norm)
                self.assertEqual(model.use_attention_residual, residual)

    def test_gnn_supports_all_ablation_combinations(self):
        edge_index = torch.tensor([[0, 1], [1, 0]])
        for layer_norm in (False, True):
            for residual in (False, True):
                model = H3GraphPriceModel(
                    node_feature_dim=5,
                    property_dim=4,
                    time_dim=3,
                    market_dim=2,
                    graph_hidden_dim=8,
                    head_hidden_dim=16,
                    graph_layer_norm=layer_norm,
                    graph_residual=residual,
                ).eval()
                output = model(
                    torch.randn(2, 5), edge_index, torch.tensor([0, 1]),
                    torch.randn(2, 4), torch.randn(2, 3), torch.randn(2, 2),
                )
                self.assertEqual(tuple(output.shape), (2, 1))
                self.assertEqual(hasattr(model, "graph_norm_one"), layer_norm)
                self.assertEqual(model.use_graph_residual, residual)

    def test_training_configs_serialize_ablation_settings(self):
        attention = replace(
            TrainingConfig(), attention_layer_norm=True, attention_residual=True,
            use_local_correction=False,
        ).as_dict()
        graph = replace(
            GNNTrainingConfig(), graph_layer_norm=True, graph_residual=True
        ).as_dict()
        self.assertTrue(attention["attention_layer_norm"])
        self.assertTrue(attention["attention_residual"])
        self.assertFalse(attention["use_local_correction"])
        self.assertTrue(graph["graph_layer_norm"])
        self.assertTrue(graph["graph_residual"])

    def test_manager_can_remove_local_correction_before_processing(self):
        manager = ModelManager().configure_local_correction(False)
        self.assertFalse(manager.use_local_correction)
        self.assertEqual(manager.local_feature_dim, 0)
        manager.configure_local_correction(True)
        self.assertTrue(manager.use_local_correction)
        self.assertEqual(manager.local_feature_dim, len(manager._LOCAL_FEATURES))


class RunManifestTests(unittest.TestCase):
    def test_manifest_records_architecture_data_and_split_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            sales = Path(directory) / "sales.csv"
            sales.write_text("sale_price\n100000\n", encoding="utf-8")
            manifest = build_run_manifest(
                run_id="test-run",
                model_family="attention",
                architecture_version=5,
                architecture={
                    "attention_layer_norm": True,
                    "attention_residual": False,
                    "use_local_correction": False,
                },
                training_config={
                    "sales_csv": str(sales),
                    "train_ratio": 0.7,
                    "temporal_split": True,
                    "random_seed": 42,
                },
                metrics={"validation_mape": 12.3},
                train_indices=[0, 1],
                validation_indices=[2],
            )
            destination = write_run_manifest(directory, manifest)
            self.assertTrue(destination.is_file())
            self.assertEqual(manifest["architecture_variant"], "layer_norm+no_residual")
            self.assertIsNotNone(manifest["dataset"]["sha256"])
            self.assertIsNotNone(manifest["split"]["validation_indices_sha256"])


if __name__ == "__main__":
    unittest.main()
