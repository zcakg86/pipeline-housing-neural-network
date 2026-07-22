"""Train LightGBM with the neural model's leakage-safe features."""

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from pricemodel.data_pipeline import DatasetBuilder
from pricemodel.lightgbm_model import LightGBMPriceModel


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/sales_2020_25.csv")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--max-depth", type=int, default=-1)
    parser.add_argument("--min-child-samples", type=int, default=100)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--reg-lambda", type=float, default=5.0)
    parser.add_argument("--selection-fraction", type=float, default=0.15)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dir")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir or str(
        Path("outputs/lightgbm") / datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    print("Preparing the same leakage-safe features as the neural model...")
    data = DatasetBuilder()
    data._map_communities(pd.read_csv(args.data))
    data._prepare_data(
        market_indicator_cache_path="data/market_indicators/fred_indicators.csv"
    )

    model = LightGBMPriceModel(
        n_estimators=args.iterations,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        max_depth=args.max_depth,
        min_child_samples=args.min_child_samples,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda,
        n_jobs=args.jobs,
        random_state=args.random_state,
    )
    print(
        f"Selecting boosting rounds within the first {args.train_ratio:.0%}, "
        "then refitting that full training period..."
    )
    predictions = model.fit(
        data,
        train_ratio=args.train_ratio,
        selection_fraction=args.selection_fraction,
        early_stopping_rounds=args.early_stopping_rounds,
    )
    model.save(output_dir, predictions)

    metrics = model.metrics
    print("Validation metrics")
    print(f"  Best iteration: {model.best_iteration}")
    print(f"  MAPE:           {metrics['validation_mape']:.2f}%")
    print(f"  Median APE:     {metrics['validation_median_ape']:.2f}%")
    print(f"  RMSE:           ${metrics['validation_rmse']:,.0f}")
    print(f"Saved model and validation predictions to {output_dir}")


if __name__ == "__main__":
    main()
