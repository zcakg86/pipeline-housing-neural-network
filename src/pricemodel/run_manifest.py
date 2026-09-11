"""Self-describing run manifests used for reproducible model ablations."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def _file_sha256(path: str | Path | None) -> str | None:
    if not path:
        return None
    source = Path(path)
    if not source.is_file():
        return None
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _indices_sha256(indices) -> str | None:
    if indices is None:
        return None
    values = np.asarray(indices, dtype=np.int64)
    return hashlib.sha256(values.tobytes()).hexdigest()


def _git_metadata() -> dict:
    try:
        root = Path(__file__).resolve().parents[2]
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip())
        return {"commit": commit, "working_tree_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "working_tree_dirty": None}


def build_run_manifest(
    *, run_id: str, model_family: str, architecture_version: int,
    architecture: dict, training_config: dict, metrics: dict,
    train_indices=None, validation_indices=None,
) -> dict:
    """Build comparable metadata without depending on a trainer implementation."""
    sales_csv = training_config.get("sales_csv")
    normalization = any(
        bool(value) for key, value in architecture.items() if "layer_norm" in key
    )
    residual = any(
        bool(value) for key, value in architecture.items() if "residual" in key
    )
    variant = "+".join(
        ["layer_norm" if normalization else "no_norm",
         "residual" if residual else "no_residual"]
    )
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "model_family": model_family,
        "architecture_version": architecture_version,
        "architecture_variant": variant,
        "architecture": architecture,
        "training_config": training_config,
        "random_seed": training_config.get("random_seed"),
        "dataset": {"path": sales_csv, "sha256": _file_sha256(sales_csv)},
        "split": {
            "strategy": "chronological" if training_config.get("temporal_split", True) else "random",
            "train_ratio": training_config.get("train_ratio"),
            "train_indices_sha256": _indices_sha256(train_indices),
            "validation_indices_sha256": _indices_sha256(validation_indices),
        },
        "git": _git_metadata(),
        "metrics": metrics,
    }


def write_run_manifest(directory: str | Path, manifest: dict) -> Path:
    """Atomically write ``run_manifest.json`` in a training-run directory."""
    destination = Path(directory) / "run_manifest.json"
    temporary = destination.with_name(destination.name + ".tmp")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
