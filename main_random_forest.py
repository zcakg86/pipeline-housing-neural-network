"""Train a quick Random Forest baseline with the neural model's features."""

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from pricemodel.data_pipeline import DatasetBuilder
from pricemodel.random_forest_model import RandomForestPriceModel


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/sales_2020_25.csv")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=24)
    parser.add_argument("--min-samples-leaf", type=int, default=5)
    parser.add_argument("--max-features", default="sqrt")
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dir")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir or str(
        Path("outputs/random_forest") / datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    print("Preparing the same leakage-safe features as the neural model...")
    data = DatasetBuilder()
    data._map_communities(pd.read_csv(args.data))
    data._prepare_data(market_indicator_cache_path="data/market_indicators/fred_indicators.csv")

    model = RandomForestPriceModel(
        n_estimators=args.trees,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        n_jobs=args.jobs,
        random_state=args.random_state,
    )
    print(f"Training on the first {args.train_ratio:.0%} of chronologically sorted sales...")
    predictions = model.fit(data, train_ratio=args.train_ratio)
    model.save(output_dir, predictions)

    metrics = model.metrics
    print("Validation metrics")
    print(f"  MAPE:       {metrics['validation_mape']:.2f}%")
    print(f"  Median APE: {metrics['validation_median_ape']:.2f}%")
    print(f"  RMSE:       ${metrics['validation_rmse']:,.0f}")
    print(f"Saved model and validation predictions to {output_dir}")


if __name__ == "__main__":
    main()
