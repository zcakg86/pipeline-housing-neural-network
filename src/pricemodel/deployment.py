"""Build and atomically install versioned Java model-artifact bundles.

Exporters write into an isolated staging directory.  Deployment validates the
complete bundle and swaps the Java resource directory only after every file is
present, preventing a running/rebuilding application from observing a mixture
of old and new model files.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

CANONICAL_JAVA_ARTIFACT_DIR = Path(
    "java-app/house-price-app/src/main/resources/model-artifacts"
)
REQUIRED_NEURAL_ARTIFACTS = {
    "model.onnx", "model_metadata.json", "scalers.json",
    "community_map.json",
    "h3_l8_neighbor_communities.json", "h3_l8_neighbor_cells.json",
    "local_market_snapshot.json", "synthetic_h3_l8_grid.json",
    "feature_contract.json",
}
REQUIRED_LIGHTGBM_ARTIFACTS = {
    "lightgbm.onnx", "lightgbm_metadata.json", "lightgbm_model.txt",
    "feature_contract.json",
}
REQUIRED_GNN_ARTIFACTS = {
    "gnn_price_head.onnx", "gnn_monthly_embeddings.bin.gz",
    "gnn_scalers.json", "gnn_metadata.json",
}
RETIRED_ARTIFACTS = {"year_vocab.json", "week_vocab.json"}


def create_staging_directory(root="outputs/deployment", version=None, *, seed=True):
    """Create a versioned bundle, optionally seeded from the live deployment."""
    version = version or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    destination = Path(root) / version
    if destination.exists():
        raise FileExistsError(f"Deployment staging directory already exists: {destination}")
    if seed and CANONICAL_JAVA_ARTIFACT_DIR.exists():
        shutil.copytree(CANONICAL_JAVA_ARTIFACT_DIR, destination)
    else:
        destination.mkdir(parents=True)
    return destination


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(bundle_dir, *, sources=None):
    """Record immutable hashes for every deployable artifact in a bundle."""
    bundle_dir = Path(bundle_dir)
    files = {
        path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
        for path in sorted(bundle_dir.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": sources or {},
        "files": files,
    }
    temporary = bundle_dir / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, bundle_dir / "manifest.json")
    return manifest


def validate_bundle(bundle_dir, *, require_neural=True, require_lightgbm=True,
                    require_gnn=True):
    """Reject incomplete bundles before they can replace Java resources."""
    bundle_dir = Path(bundle_dir)
    required = {"king_county_water.geojson", "fred_indicators.csv"}
    if require_neural:
        required |= REQUIRED_NEURAL_ARTIFACTS
    if require_lightgbm:
        required |= REQUIRED_LIGHTGBM_ARTIFACTS
    if require_gnn:
        required |= REQUIRED_GNN_ARTIFACTS
    missing = sorted(name for name in required if not (bundle_dir / name).is_file())
    if missing:
        raise FileNotFoundError("Incomplete deployment bundle; missing: " + ", ".join(missing))
    retired = sorted(name for name in RETIRED_ARTIFACTS if (bundle_dir / name).exists())
    if retired:
        raise ValueError(
            "Deployment bundle contains retired categorical-time artifacts: "
            + ", ".join(retired)
        )
    return required


def atomic_deploy(bundle_dir, destination=CANONICAL_JAVA_ARTIFACT_DIR):
    """Install a validated bundle with a same-filesystem directory swap."""
    bundle_dir, destination = Path(bundle_dir), Path(destination)
    validate_bundle(bundle_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    incoming = destination.with_name(destination.name + ".incoming")
    previous = destination.with_name(destination.name + ".previous")
    if incoming.exists() or previous.exists():
        raise RuntimeError(
            f"Refusing deployment while stale swap directories exist: {incoming}, {previous}"
        )
    shutil.copytree(bundle_dir, incoming)
    try:
        if destination.exists():
            os.replace(destination, previous)
        os.replace(incoming, destination)
        if previous.exists():
            shutil.rmtree(previous)
    except Exception:
        if not destination.exists() and previous.exists():
            os.replace(previous, destination)
        raise
    finally:
        if incoming.exists():
            shutil.rmtree(incoming)
    return destination
