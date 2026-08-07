"""Stage and verify a trained neural model for Java ONNX inference.

Nothing is written during import. Run with ``--deploy`` to atomically replace
the Java resource bundle after all required neural and LightGBM artifacts have
been validated; without it, the versioned staging bundle is retained for review.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from pricemodel.deployment import atomic_deploy, create_staging_directory, write_manifest
from pricemodel.feature_contract import FEATURE_CONTRACT, write_feature_contract
from pricemodel.model_manager import ModelManager


class ModelWithExtras(torch.nn.Module):
    """Expose price, uncertainty, and averaged CLS attention to ONNX/Java."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, community_indices, property_features,
                time_features, market_features, local_market_features=None):
        log_price, log_var = self.model(
            community_indices, property_features, time_features,
            market_features, local_market_features, return_uncertainty=True,
            need_weights=True,
        )
        return log_price, log_var, self.model.last_cls_attention.mean(dim=1)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", help="Training checkpoint directory; defaults to latest")
    parser.add_argument("--bundle-dir", help="Existing/new staging bundle directory")
    parser.add_argument("--deploy", action="store_true", help="Atomically install the complete bundle")
    return parser.parse_args(argv)


def find_latest_model():
    """Return the latest timestamped directory containing model.pth."""
    candidates = sorted(Path("outputs/models").glob("*/model.pth"))
    if not candidates:
        raise FileNotFoundError("No trained model found in outputs/models")
    return candidates[-1].parent


def _copy_required(source, destination, description):
    if not Path(source).is_file():
        raise FileNotFoundError(f"{source} is required for {description}")
    shutil.copy2(source, destination)


def _compatible_local_market_snapshot(model_dir: Path) -> Path:
    """Choose a snapshot whose local-field schema matches the exported model.

    The mutable data snapshot is preferred when a scheduled refresh has already
    produced the current schema. Otherwise use the checkpoint snapshot, rather
    than silently pairing a new ONNX model with retired local-feature values.
    """
    expected_features = FEATURE_CONTRACT["groups"]["local_market"]
    candidates = [
        Path("data/local_market_snapshot.json"),
        model_dir / "local_market_snapshot.json",
    ]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            snapshot = json.loads(candidate.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid local market snapshot: {candidate}") from exc
        if snapshot.get("feature_order") == expected_features:
            return candidate
    raise ValueError(
        "No local-market snapshot matches the current feature contract. "
        "Retrain the neural model or run refresh_local_market_snapshot.py first."
    )


def _export_onnx(manager, destination):
    """Export the model and verify numeric parity on a dynamic batch."""
    model = manager.predictor.model.cpu().eval()
    wrapper = ModelWithExtras(model).eval()
    batch = 2
    inputs = [
        torch.zeros(batch, 7, dtype=torch.long),
        torch.zeros(batch, manager.property_dim),
        torch.zeros(batch, manager.continuous_time_dim),
        torch.zeros(batch, manager.market_dim),
    ]
    names = ["community_indices", "property_features",
             "time_features", "market_features"]
    if manager.local_feature_dim:
        inputs.append(torch.zeros(batch, 7, manager.local_feature_dim))
        names.append("local_market_features")
    outputs = ["log_price_scaled", "log_var_scaled", "cls_attention"]
    dynamic_axes = {name: {0: "batch"} for name in names + outputs}
    torch.onnx.export(
        wrapper, tuple(inputs), str(destination), input_names=names,
        output_names=outputs, dynamic_axes=dynamic_axes, opset_version=17,
    )

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("onnxruntime is required to verify deployment parity") from exc
    with torch.no_grad():
        expected = wrapper(*inputs)[0].numpy()
    actual = ort.InferenceSession(str(destination), providers=["CPUExecutionProvider"]).run(
        None, {name: value.numpy() for name, value in zip(names, inputs)}
    )[0]
    maximum_difference = float(np.max(np.abs(expected - actual)))
    if maximum_difference >= 1e-4:
        raise ValueError(f"Neural ONNX parity failed: max log-price difference {maximum_difference}")
    return maximum_difference


def export_neural(args):
    """Create a self-describing neural artifact bundle and optionally deploy it."""
    model_dir = Path(args.model_dir) if args.model_dir else find_latest_model()
    bundle_dir = Path(args.bundle_dir) if args.bundle_dir else create_staging_directory()
    bundle_dir.mkdir(parents=True, exist_ok=True)
    for retired_name in ("year_vocab.json", "week_vocab.json"):
        (bundle_dir / retired_name).unlink(missing_ok=True)

    manager = ModelManager().load_model(model_dir)
    if list(manager._PROPERTY_FEATURES) != FEATURE_CONTRACT["groups"]["property"]:
        raise ValueError("Neural property feature order differs from feature_contract.json")
    if list(manager._LOCAL_FEATURES[:manager.local_feature_dim]) != FEATURE_CONTRACT["groups"]["local_market"]:
        raise ValueError("Neural local feature order differs from feature_contract.json")

    maximum_difference = _export_onnx(manager, bundle_dir / "model.onnx")
    scaler_features = (
        list(manager._PROPERTY_FEATURES) + list(manager._TIME_FEATURES)
        + list(manager._MARKET_FEATURES) + ["log_price"]
        + list(manager._LOCAL_FEATURES[:manager.local_feature_dim])
    )
    scalers = {
        name: {"mean": float(manager.scalers[name].mean_[0]),
               "scale": float(manager.scalers[name].scale_[0])}
        for name in scaler_features
        if name in manager.scalers
    }
    missing_scalers = sorted(set(scaler_features).difference(scalers))
    if missing_scalers:
        raise ValueError("Checkpoint is missing scalers: " + ", ".join(missing_scalers))
    (bundle_dir / "scalers.json").write_text(json.dumps(scalers, indent=2) + "\n")
    checkpoint = torch.load(model_dir / "model.pth", map_location="cpu")
    metadata = {
        "model_version": model_dir.name,
        "embedding_dim": manager.embedding_dim,
        "community_embedding_dim": manager.community_embedding_dim,
        "hidden_dim": manager.hidden_dim,
        "property_dim": manager.property_dim,
        "continuous_time_dim": manager.continuous_time_dim,
        "market_dim": manager.market_dim,
        "local_feature_dim": manager.local_feature_dim,
        "n_communities": manager.n_communities,
        "use_neighborhood_pooling": manager.use_neighborhood_pooling,
        "pooling_strategy": manager.pooling_strategy,
        "reference_date": checkpoint.get("reference_date"),
        "property_features": list(manager._PROPERTY_FEATURES),
        "time_features": list(manager._TIME_FEATURES),
        "market_features": list(manager._MARKET_FEATURES),
        "local_market_features": list(manager._LOCAL_FEATURES[:manager.local_feature_dim]),
        "max_verified_log_price_difference": maximum_difference,
    }
    (bundle_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    # Static geography and dated state are copied only after model/contract
    # validation, so a failed export cannot disturb the live Java application.
    sources = {
        "community_map.json": Path("data/community_map.json"),
        "h3_l8_neighbor_communities.json": Path("data/h3_l8_neighbor_communities.json"),
        "h3_l8_neighbor_cells.json": Path("data/h3_l8_neighbor_cells.json"),
        "local_market_snapshot.json": _compatible_local_market_snapshot(model_dir),
        "synthetic_h3_l8_grid.json": Path("data/synthetic_h3_l8_grid.json"),
        "king_county_water.geojson": Path("data/osm/king_county_water.geojson"),
        "fred_indicators.csv": Path("data/market_indicators/fred_indicators.csv"),
    }
    for name, source in sources.items():
        if not source.exists() and name in {"h3_l8_neighbor_cells.json"}:
            source = model_dir / name
        _copy_required(source, bundle_dir / name, "Java neural inference")
    write_feature_contract(bundle_dir / "feature_contract.json")
    write_manifest(bundle_dir, sources={"neural_model_dir": str(model_dir)})
    if args.deploy:
        atomic_deploy(bundle_dir)
    print(f"Verified neural ONNX max log-price difference: {maximum_difference:.8f}")
    print(f"Exported neural artifacts to: {bundle_dir}")
    return bundle_dir


def main(argv=None):
    return export_neural(parse_args(argv))


if __name__ == "__main__":
    main()
