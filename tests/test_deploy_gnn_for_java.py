"""GNN-only deployment must preserve the other installed model artifacts."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from deploy_gnn_for_java import deploy_gnn_only, latest_gnn_model_dir
from pricemodel.deployment import (
    REQUIRED_GNN_ARTIFACTS,
    REQUIRED_LIGHTGBM_ARTIFACTS,
    REQUIRED_NEURAL_ARTIFACTS,
    verify_manifest,
    write_manifest,
)


class GnnOnlyDeploymentTests(unittest.TestCase):
    def test_replaces_only_gnn_files_and_updates_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            deployed = root / "model-artifacts"
            deployed.mkdir()
            required = (
                REQUIRED_NEURAL_ARTIFACTS
                | REQUIRED_LIGHTGBM_ARTIFACTS
                | REQUIRED_GNN_ARTIFACTS
                | {"king_county_water.geojson", "fred_indicators.csv"}
            )
            for name in required:
                (deployed / name).write_bytes(f"original:{name}".encode())
            previous = write_manifest(deployed, sources={
                "neural_model_dir": "attention-run",
                "lightgbm_model_dir": "lightgbm-run",
                "gnn_model_dir": "old-gnn-run",
            })
            model_dir = root / "new-gnn-run"
            model_dir.mkdir()
            (model_dir / "gnn_model.pth").write_bytes(b"checkpoint")

            def fake_export(args):
                self.assertFalse(args.deploy)
                self.assertEqual(Path(args.model_dir), model_dir)
                for name in REQUIRED_GNN_ARTIFACTS:
                    (Path(args.bundle_dir) / name).write_bytes(f"updated:{name}".encode())

            with patch("deploy_gnn_for_java.export_gnn", side_effect=fake_export):
                result = deploy_gnn_only(
                    model_dir, deployed_dir=deployed, staging_root=root / "staging"
                )

            self.assertEqual(result, deployed)
            current = verify_manifest(deployed)
            self.assertEqual(current["sources"]["gnn_model_dir"], str(model_dir))
            self.assertEqual(current["sources"]["neural_model_dir"], "attention-run")
            self.assertEqual(current["sources"]["lightgbm_model_dir"], "lightgbm-run")
            for name, details in previous["files"].items():
                if name in REQUIRED_GNN_ARTIFACTS:
                    self.assertNotEqual(current["files"][name], details)
                else:
                    self.assertEqual(current["files"][name], details)

    def test_rejects_changed_installed_bundle_before_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            deployed = root / "model-artifacts"
            deployed.mkdir()
            required = (
                REQUIRED_NEURAL_ARTIFACTS
                | REQUIRED_LIGHTGBM_ARTIFACTS
                | REQUIRED_GNN_ARTIFACTS
                | {"king_county_water.geojson", "fred_indicators.csv"}
            )
            for name in required:
                (deployed / name).write_bytes(b"original")
            write_manifest(deployed)
            (deployed / "model.onnx").write_bytes(b"changed")
            model_dir = root / "gnn"
            model_dir.mkdir()
            (model_dir / "gnn_model.pth").write_bytes(b"checkpoint")
            with patch("deploy_gnn_for_java.export_gnn") as export:
                with self.assertRaisesRegex(ValueError, "Deployment manifest mismatch"):
                    deploy_gnn_only(
                        model_dir, deployed_dir=deployed, staging_root=root / "staging"
                    )
                export.assert_not_called()

    def test_rejects_exporter_that_changes_another_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            deployed = root / "model-artifacts"
            deployed.mkdir()
            required = (
                REQUIRED_NEURAL_ARTIFACTS
                | REQUIRED_LIGHTGBM_ARTIFACTS
                | REQUIRED_GNN_ARTIFACTS
                | {"king_county_water.geojson", "fred_indicators.csv"}
            )
            for name in required:
                (deployed / name).write_bytes(b"original")
            write_manifest(deployed)
            model_dir = root / "gnn"
            model_dir.mkdir()
            (model_dir / "gnn_model.pth").write_bytes(b"checkpoint")

            def bad_export(args):
                (Path(args.bundle_dir) / "model.onnx").write_bytes(b"unexpected change")

            with patch("deploy_gnn_for_java.export_gnn", side_effect=bad_export):
                with self.assertRaisesRegex(ValueError, "Non-GNN artifact changed"):
                    deploy_gnn_only(
                        model_dir, deployed_dir=deployed, staging_root=root / "staging"
                    )
            self.assertEqual((deployed / "model.onnx").read_bytes(), b"original")
            verify_manifest(deployed)

    def test_latest_run_requires_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            older = root / "20260916_094016"
            newer = root / "20260916_094433"
            older.mkdir()
            newer.mkdir()
            (older / "gnn_model.pth").write_bytes(b"checkpoint")
            self.assertEqual(latest_gnn_model_dir(root), older)
            (newer / "gnn_model.pth").write_bytes(b"checkpoint")
            self.assertEqual(latest_gnn_model_dir(root), newer)


if __name__ == "__main__":
    unittest.main()
