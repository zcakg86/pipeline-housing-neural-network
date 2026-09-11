"""Retrain a saved neural model with the current leakage-safe workflow."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from pricemodel.data_pipeline import DatasetBuilder
from pricemodel.model_manager import ModelManager
from pricemodel.training_config import TrainingConfig


def find_latest_model(root=Path("outputs/models")):
    """Return the newest checkpoint directory by its timestamped path."""
    checkpoints = sorted(root.glob("*/model.pth"))
    if not checkpoints:
        raise FileNotFoundError(f"No trained model found below {root}")
    return checkpoints[-1].parent


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--data", default="data/sales_2020_25.csv")
    parser.add_argument(
        "--indicators",
        default="data/market_indicators/fred_indicators.csv",
        help="Frozen dated indicator CSV; this command never refreshes FRED.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-uncertainty",
        action="store_true",
        help="Train only the mean model; deployed uncertainty will be unreliable.",
    )
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(argv)


def main(argv=None):
    """Load architecture metadata, retrain chronologically, and save a new run."""
    args = parse_args(argv)
    source_model = args.model_dir or find_latest_model()
    print(f"Loading architecture from: {source_model}")

    manager = ModelManager()
    manager.load_model(source_model)
    output_dir = args.output_dir or (
        Path("outputs/models")
        / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_retrain"
    )
    manager.directory = output_dir

    data = DatasetBuilder()
    data._map_communities(pd.read_csv(args.data), output_dir="data")
    data._prepare_data(market_indicator_cache_path=args.indicators)
    manager.processor(data)
    manager.split_data(train_ratio=0.7, temporal_split=True)

    estimate_uncertainty = not args.skip_uncertainty
    if not estimate_uncertainty:
        print(
            "WARNING: uncertainty calibration is disabled; prediction intervals "
            "from this checkpoint must not be treated as reliable."
        )
    config = TrainingConfig(
        sales_csv=args.data,
        market_indicator_csv=args.indicators,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        estimate_uncertainty=estimate_uncertainty,
        pooling_strategy=manager.pooling_strategy,
        use_local_correction=manager.use_local_correction,
        attention_layer_norm=manager.attention_layer_norm,
        attention_residual=manager.attention_residual,
        uncertainty_calibration_epochs=10 if estimate_uncertainty else 0,
        random_seed=args.seed,
    )
    manager.results["training_config"] = config.as_dict()
    manager.train_model(**config.train_kwargs(len(manager._PROPERTY_FEATURES)))
    manager.save_and_reload_for_evaluation()
    print(f"Retrained checkpoint saved and verified at: {manager.directory}")
    return manager


if __name__ == "__main__":
    main()
