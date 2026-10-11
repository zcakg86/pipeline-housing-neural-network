"""Replace only the GNN artifacts in the installed Java model bundle."""
from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path

from export_gnn_for_java import export_gnn
from pricemodel.deployment import (
    CANONICAL_JAVA_ARTIFACT_DIR,
    REQUIRED_GNN_ARTIFACTS,
    atomic_deploy,
    create_staging_directory,
    validate_bundle,
    verify_manifest,
    write_manifest,
)


def latest_gnn_model_dir(root: str | Path = "outputs/gnn") -> Path:
    """Select the newest timestamped run with a saved GNN checkpoint."""
    candidates = sorted(Path(root).glob("*/gnn_model.pth"))
    if not candidates:
        raise FileNotFoundError("No GNN checkpoint found; pass --gnn-model-dir or train a GNN")
    return candidates[-1].parent


def deploy_gnn_only(
    model_dir: str | Path,
    *,
    deployed_dir: str | Path = CANONICAL_JAVA_ARTIFACT_DIR,
    staging_root: str | Path = "outputs/deployment",
) -> Path:
    """Preserve non-GNN artifacts while atomically installing one GNN run."""
    model_dir = Path(model_dir)
    deployed_dir = Path(deployed_dir)
    if not (model_dir / "gnn_model.pth").is_file():
        raise FileNotFoundError(f"GNN checkpoint not found: {model_dir / 'gnn_model.pth'}")
    validate_bundle(deployed_dir)
    previous_manifest = verify_manifest(deployed_dir)

    staged = create_staging_directory(
        root=staging_root, seed=True, source_dir=deployed_dir
    )
    export_gnn(Namespace(model_dir=str(model_dir), bundle_dir=str(staged), deploy=False))
    sources = dict(previous_manifest.get("sources", {}))
    sources["gnn_model_dir"] = str(model_dir)
    manifest = write_manifest(staged, sources=sources)
    validate_bundle(staged)

    # A GNN-only deployment must not silently change any other model or
    # runtime artifact copied from the installed bundle.
    for name, details in previous_manifest["files"].items():
        if name not in REQUIRED_GNN_ARTIFACTS:
            if manifest["files"].get(name) != details:
                raise ValueError(f"Non-GNN artifact changed during staging: {name}")

    destination = atomic_deploy(staged, destination=deployed_dir)
    print(f"Deployed GNN run {model_dir.name} to {destination}")
    print(f"Versioned bundle retained at {staged}")
    return destination


def main(argv=None) -> Path:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gnn-model-dir", help="GNN run directory (default: latest saved run)"
    )
    args = parser.parse_args(argv)
    model_dir = Path(args.gnn_model_dir) if args.gnn_model_dir else latest_gnn_model_dir()
    return deploy_gnn_only(model_dir)


if __name__ == "__main__":
    main()
