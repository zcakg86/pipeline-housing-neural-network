"""Regression tests for package boundaries and safe deployment primitives."""
import importlib
import json
import random
import tempfile
import unittest
from pathlib import Path

from pricemodel.deployment import (
    REQUIRED_LIGHTGBM_ARTIFACTS,
    REQUIRED_NEURAL_ARTIFACTS,
    validate_bundle,
    write_manifest,
)
from pricemodel.feature_contract import FEATURE_CONTRACT, write_feature_contract
from pricemodel.reproducibility import seed_everything
from pricemodel.training_config import TrainingConfig

import numpy as np
import torch


class StructureReliabilityTests(unittest.TestCase):
    def test_export_modules_are_import_safe(self):
        """Importing an exporter must not start model loading or file output."""
        importlib.import_module("export_model_for_java")
        importlib.import_module("export_lightgbm_for_java")

    def test_feature_contract_round_trip(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "feature_contract.json"
            write_feature_contract(path)
            self.assertEqual(json.loads(path.read_text()), FEATURE_CONTRACT)

    def test_incomplete_deployment_is_rejected_before_swap(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            (bundle / "model.onnx").write_bytes(b"partial")
            write_manifest(bundle)
            with self.assertRaises(FileNotFoundError):
                validate_bundle(bundle)

    def test_deployment_rejects_a_missing_synthetic_grid(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            required = (
                REQUIRED_NEURAL_ARTIFACTS
                | REQUIRED_LIGHTGBM_ARTIFACTS
                | {"king_county_water.geojson", "fred_indicators.csv"}
            )
            for name in required - {"synthetic_h3_l8_grid.json"}:
                (bundle / name).write_bytes(b"test")
            with self.assertRaisesRegex(
                FileNotFoundError, "synthetic_h3_l8_grid.json"
            ):
                validate_bundle(bundle)

    def test_training_seed_controls_python_numpy_torch_and_loader_generator(self):
        first_generator = seed_everything(42)
        first = (random.random(), np.random.random(), torch.rand(1).item(),
                 torch.rand(1, generator=first_generator).item())
        second_generator = seed_everything(42)
        second = (random.random(), np.random.random(), torch.rand(1).item(),
                  torch.rand(1, generator=second_generator).item())
        self.assertEqual(first, second)
        config_kwargs = TrainingConfig().train_kwargs(5)
        self.assertEqual(config_kwargs["random_seed"], 42)
        self.assertEqual(config_kwargs["patience"], 8)
        self.assertEqual(config_kwargs["lr_plateau_factor"], 0.5)
        self.assertEqual(config_kwargs["lr_plateau_patience"], 2)
        self.assertEqual(config_kwargs["residual_penalty"], 1e-2)


if __name__ == "__main__":
    unittest.main()
