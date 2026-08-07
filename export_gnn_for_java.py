"""Export the monthly H3 GNN as a Java price-head plus dated embedding lookup."""
from __future__ import annotations

import argparse
import gzip
import json
import shutil
import struct
from pathlib import Path

import numpy as np
import torch

from pricemodel.deployment import atomic_deploy, create_staging_directory, write_manifest
from pricemodel.gnn_trainer import GNNBaselineTrainer


class GNNPriceHead(torch.nn.Module):
    """The deployment portion of GraphSAGE after Python has encoded each H3 month."""

    def __init__(self, model):
        super().__init__()
        self.price_head = model.price_head

    def forward(self, gnn_embedding, property_features, time_features, market_features):
        return self.price_head(torch.cat(
            [gnn_embedding, property_features, time_features, market_features], dim=1
        ))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--bundle-dir")
    parser.add_argument("--deploy", action="store_true")
    return parser.parse_args(argv)


def _export_head(trainer, destination: Path) -> float:
    model = trainer.model.cpu().eval()
    wrapper = GNNPriceHead(model).eval()
    dimensions = (trainer.graph_hidden_dim, 4, 3, 2)
    inputs = tuple(torch.zeros(2, width) for width in dimensions)
    names = ["gnn_embedding", "property_features", "time_features", "market_features"]
    torch.onnx.export(
        wrapper, inputs, str(destination), input_names=names,
        output_names=["log_price_scaled"], dynamic_axes={name: {0: "batch"} for name in names + ["log_price_scaled"]},
        opset_version=17,
    )
    import onnxruntime as ort
    expected = wrapper(*inputs).detach().numpy()
    actual = ort.InferenceSession(str(destination), providers=["CPUExecutionProvider"]).run(
        None, {name: value.numpy() for name, value in zip(names, inputs)}
    )[0]
    difference = float(np.max(np.abs(expected - actual)))
    if difference >= 1e-5:
        raise ValueError(f"GNN price-head ONNX parity failed: {difference}")
    return difference


def _write_embeddings(trainer, destination: Path) -> None:
    """Write a compact gzip binary artifact; Java loads it once at startup."""
    trainer.model.eval()
    with torch.no_grad():
        embeddings = np.stack([
            trainer.model.encode_nodes(
                trainer.node_features[month].cpu(), trainer.edge_index.cpu()
            )
            .detach().cpu().numpy()
            for month in range(len(trainer.graph_data.month_starts))
        ]).astype(">f4", copy=False)
    cells = trainer.graph_data.cell_ids
    months = trainer.graph_data.month_starts
    with gzip.open(destination, "wb") as output:
        output.write(b"HPGNN01")
        output.write(struct.pack(">III", len(months), len(cells), embeddings.shape[-1]))
        for value in months:
            encoded = value.encode("ascii")
            output.write(struct.pack(">H", len(encoded)))
            output.write(encoded)
        for value in cells:
            encoded = value.encode("ascii")
            output.write(struct.pack(">H", len(encoded)))
            output.write(encoded)
        output.write(embeddings.tobytes())


def export_gnn(args):
    model_dir = Path(args.model_dir)
    bundle = Path(args.bundle_dir) if args.bundle_dir else create_staging_directory()
    bundle.mkdir(parents=True, exist_ok=True)
    trainer = GNNBaselineTrainer.load_for_explanations(model_dir)
    difference = _export_head(trainer, bundle / "gnn_price_head.onnx")
    _write_embeddings(trainer, bundle / "gnn_monthly_embeddings.bin.gz")
    validation = trainer.graph_data.dataframe.iloc[trainer.validation_indices]
    if not {"log_price", "predicted_log_price"}.issubset(validation.columns):
        raise ValueError(
            "GNN export needs validation predicted_log_price values; retrain the GNN first"
        )
    validation_log_residual = np.abs(
        validation["log_price"].to_numpy(dtype=np.float64)
        - validation["predicted_log_price"].to_numpy(dtype=np.float64)
    )
    conformal_intervals = {
        "0.90": float(np.quantile(validation_log_residual, 0.90, method="higher")),
        "0.95": float(np.quantile(validation_log_residual, 0.95, method="higher")),
    }
    scalers = {
        name: {"mean": [float(value) for value in scaler.mean_],
               "scale": [float(value) for value in scaler.scale_]}
        for name, scaler in trainer.scalers.items()
    }
    (bundle / "gnn_scalers.json").write_text(json.dumps(scalers, indent=2) + "\n")
    metadata = {
        "model_version": model_dir.name,
        "architecture": "monthly_h3_graphsage_v1",
        "graph_hidden_dim": trainer.graph_hidden_dim,
        "head_hidden_dim": trainer.head_hidden_dim,
        "property_features": ["sqft", "sqft_lot", "beds", "water_proximity"],
        "time_features": ["time_trend", "annual_sin", "annual_cos"],
        "market_features": ["mortgage_rate", "unemployment_rate"],
        "node_feature_names": list(trainer.graph_data.node_feature_names),
        "reference_date": str(trainer.graph_data.dataframe["sale_date"].min().date()),
        "month_starts": list(trainer.graph_data.month_starts),
        "latest_snapshot_month": trainer.graph_data.month_starts[-1],
        "fallback_after_latest": "use latest causal monthly embedding; it becomes stale until retraining/export",
        "uncertainty": {
            "method": "held_out_absolute_log_residual_conformal",
            "calibration_sample_count": int(len(validation)),
            "absolute_log_residual_quantiles": conformal_intervals,
        },
        "max_verified_log_price_difference": difference,
    }
    (bundle / "gnn_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if args.deploy:
        atomic_deploy(bundle)
    print(f"Verified GNN price-head ONNX max difference: {difference:.8f}")
    print(f"Exported GNN artifacts to: {bundle}")
    return bundle


if __name__ == "__main__":
    export_gnn(parse_args())
