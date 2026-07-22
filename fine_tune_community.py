import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
from pricemodel.data_pipeline import DatasetBuilder
from pricemodel.model_manager import ModelManager


def parse_communities(value):
    communities = []
    for token in value:
        for part in str(token).split(','):
            part = part.strip()
            if not part:
                continue
            try:
                communities.append(int(part))
            except ValueError as exc:
                raise argparse.ArgumentTypeError(
                    f"Invalid community id '{part}': must be an integer"
                ) from exc
    if not communities:
        raise argparse.ArgumentTypeError("At least one community id must be provided")
    return sorted(set(communities))


def find_latest_model(ignore_finetuned=True):
    model_paths = sorted(Path('outputs/models').glob('*/model.pth'))
    if not model_paths:
        raise FileNotFoundError("No trained model checkpoint found in outputs/models/")

    model_dirs = [path.parent for path in model_paths]
    if ignore_finetuned:
        base_dirs = [d for d in model_dirs if 'finetuned' not in d.name.lower()]
        if base_dirs:
            return sorted(base_dirs)[-1]
        # fallback if only fine-tuned models exist
    return sorted(model_dirs)[-1]


def load_data(csv_path: Path):
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    data = DatasetBuilder()
    print(f"Loading data from {csv_path}")
    data._map_communities(df, output_dir='data')
    data._prepare_data(market_indicator_cache_path='data/market_indicators/fred_indicators.csv')
    return data


def filter_by_communities(data, communities):
    if 'community' not in data.dataframe.columns:
        raise ValueError("Dataframe does not contain a 'community' column after mapping.")

    filtered = data.dataframe[data.dataframe['community'].isin(communities)].copy()
    if filtered.empty:
        raise ValueError(
            f"No rows matched community ids: {communities}. "
            "Verify the community values are present in the mapped dataset."
        )

    print(f"Filtering to {len(filtered)} rows from {len(data.dataframe)} total rows")
    print(f"Communities retained: {sorted(filtered['community'].unique().tolist())}")

    data.dataframe = filtered.reset_index(drop=True)
    data.length = len(filtered)
    return data


def build_save_directory(model_dir: Path, communities, save_dir: str | None):
    if save_dir:
        directory = Path(save_dir)
    else:
        suffix = datetime.now().strftime('%Y%m%d_%H%M%S')
        community_part = '_'.join(str(c) for c in communities)
        directory = Path('outputs/models') / f"{model_dir.name}_finetuned_{community_part}_{suffix}"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune a saved property price model on one or more communities."
    )
    parser.add_argument(
        '--model-dir', type=Path,
        help='Path to an existing model directory containing model.pth. If omitted, the latest model is used.'
    )
    parser.add_argument(
        '--communities', '-c', required=True, nargs='+', type=str,
        help='Community id or comma-separated list of ids to fine-tune on. Example: 12 34 or 12,34'
    )
    parser.add_argument(
        '--data-path', type=Path, default=Path('data/sales_2020_25.csv'),
        help='Path to the sales CSV file used for fine-tuning.'
    )
    parser.add_argument('--epochs', type=int, default=10, help='Number of fine-tuning epochs.')
    parser.add_argument('--batch', type=int, default=256, help='Batch size for training.')
    parser.add_argument('--learning-rate', type=float, default=3e-4, help='Learning rate for fine-tuning.')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience.')
    parser.add_argument('--save-dir', type=Path, help='Optional directory to save the fine-tuned model.')
    parser.add_argument('--train-ratio', type=float, default=0.7, help='Train/validation split ratio.')
    parser.add_argument('--temporal-split', action='store_true', help='Use a chronological train/validation split.')
    args = parser.parse_args()

    community_ids = parse_communities(args.communities)
    model_dir = args.model_dir or find_latest_model()

    if args.model_dir is None:
        print(f"Defaulting to latest base model (ignoring fine-tuned models): {model_dir}")
    else:
        print(f"Loading model from: {model_dir}")

    manager = ModelManager()
    manager.load_model(model_dir)

    optimizer_class = manager.predictor.optimizer.__class__.__name__
    learning_rates = [g['lr'] for g in manager.predictor.optimizer.param_groups]
    print(f"Loaded optimizer: {optimizer_class}")
    print(f"Optimizer learning rates: {learning_rates}")

    data = load_data(args.data_path)
    data = filter_by_communities(data, community_ids)

    print("Processing filtered dataset with loaded model scalers...")
    manager.processor(data)
    manager.split_data(train_ratio=args.train_ratio, temporal_split=args.temporal_split)

    save_directory = build_save_directory(model_dir, community_ids, args.save_dir)
    manager.directory = str(save_directory)

    print("Starting fine-tuning...")
    manager.train_model(
        embedding_dim=manager.embedding_dim,
        hidden_dim=manager.hidden_dim,
        property_dim=manager.property_dim,
        continuous_time_dim=manager.continuous_time_dim,
        market_dim=manager.market_dim,
        epochs=args.epochs,
        batch=args.batch,
        learning_rate=args.learning_rate,
        dropout_rate=manager.dropout_rate,
        pooling_strategy=manager.pooling_strategy,
        patience=args.patience
    )

    print(f"Saving fine-tuned model to: {manager.directory}")
    manager.save_model()
    print("Fine-tuning complete.")

    return manager


if __name__ == '__main__':
    main()
