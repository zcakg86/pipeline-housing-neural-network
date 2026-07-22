"""Export both trained models and atomically deploy one verified Java bundle."""
from argparse import ArgumentParser, Namespace

from export_lightgbm_for_java import export_lightgbm
from export_model_for_java import export_neural
from pricemodel.deployment import atomic_deploy, create_staging_directory, write_manifest


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--neural-model-dir")
    parser.add_argument("--lightgbm-model-dir")
    parser.add_argument("--sales", default="data/sales_2020_25.csv")
    parser.add_argument(
        "--historical-predictions", default="data/sales_2020_25_with_predictions.csv"
    )
    args = parser.parse_args(argv)

    # Both exporters target the same isolated directory. Deployment happens
    # once, after both parity checks and the complete-bundle validation pass.
    bundle = create_staging_directory(seed=False)
    export_neural(Namespace(model_dir=args.neural_model_dir, bundle_dir=str(bundle), deploy=False))
    export_lightgbm(Namespace(
        model_dir=args.lightgbm_model_dir,
        sales=args.sales,
        historical_predictions=args.historical_predictions,
        update_historical_predictions=False,
        bundle_dir=str(bundle),
        deploy=False,
    ))
    write_manifest(bundle, sources={
        "neural_model_dir": args.neural_model_dir or "latest",
        "lightgbm_model_dir": args.lightgbm_model_dir or "latest",
    })
    destination = atomic_deploy(bundle)
    print(f"Atomically deployed complete model bundle to {destination}")


if __name__ == "__main__":
    main()
