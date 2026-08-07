"""Export both trained models and atomically deploy one verified Java bundle."""
from argparse import ArgumentParser, Namespace

from export_lightgbm_for_java import export_lightgbm
from export_model_for_java import export_neural
from export_gnn_for_java import export_gnn
from pricemodel.deployment import atomic_deploy, create_staging_directory, write_manifest


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--neural-model-dir")
    parser.add_argument("--lightgbm-model-dir")
    parser.add_argument("--gnn-model-dir", help="Monthly GraphSAGE run directory")
    parser.add_argument("--sales", default="data/sales_2020_25.csv")
    parser.add_argument(
        "--historical-predictions", default="data/sales_2020_25_with_predictions.csv"
    )
    args = parser.parse_args(argv)

    # Both exporters target the same isolated directory. Deployment happens
    # once, after both parity checks and the complete-bundle validation pass.
    # Start with non-model runtime artifacts (synthetic grid, transport fields,
    # and RentCast rolling snapshot) so replacing the model bundle never drops
    # them. Exporters overwrite only the artifacts they own.
    bundle = create_staging_directory(seed=True)
    export_neural(Namespace(model_dir=args.neural_model_dir, bundle_dir=str(bundle), deploy=False))
    export_lightgbm(Namespace(
        model_dir=args.lightgbm_model_dir,
        sales=args.sales,
        historical_predictions=args.historical_predictions,
        # The Java historical layer reads these stored, row-causal predictions.
        # Refresh them with the same LightGBM model installed in this bundle.
        update_historical_predictions=True,
        bundle_dir=str(bundle),
        deploy=False,
    ))
    gnn_dir = args.gnn_model_dir
    if gnn_dir is None:
        candidates = sorted(__import__("pathlib").Path("outputs/gnn").glob("*/gnn_model.pth"))
        if not candidates:
            raise FileNotFoundError("No GNN model found in outputs/gnn; run make train-gnn first")
        gnn_dir = str(candidates[-1].parent)
    export_gnn(Namespace(model_dir=gnn_dir, bundle_dir=str(bundle), deploy=False))
    write_manifest(bundle, sources={
        "neural_model_dir": args.neural_model_dir or "latest",
        "lightgbm_model_dir": args.lightgbm_model_dir or "latest",
        "gnn_model_dir": gnn_dir,
    })
    destination = atomic_deploy(bundle)
    print(f"Atomically deployed complete model bundle to {destination}")


if __name__ == "__main__":
    main()
