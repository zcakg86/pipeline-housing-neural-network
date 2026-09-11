"""Train the standalone, monthly-snapshot H3 GraphSAGE price baseline."""
from __future__ import annotations

import argparse
from dataclasses import replace

import pandas as pd

from pricemodel.gnn_data import build_monthly_graph_data, prepare_gnn_sales_dataframe
from pricemodel.gnn_trainer import GNNBaselineTrainer
from pricemodel.gnn_training_config import GNNTrainingConfig


DEFAULT_CONFIG = GNNTrainingConfig()


def parse_args(argv=None):
    """Keep command-line overrides deliberately small and reproducible."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.random_seed)
    parser.add_argument(
        "--graph-layer-norm",
        action="store_true",
        default=DEFAULT_CONFIG.graph_layer_norm,
        help="Apply per-node LayerNorm after each GraphSAGE layer",
    )
    parser.add_argument(
        "--graph-residual",
        action="store_true",
        default=DEFAULT_CONFIG.graph_residual,
        help="Add a residual connection around the second GraphSAGE layer",
    )
    return parser.parse_args(argv)


def main(config: GNNTrainingConfig = DEFAULT_CONFIG) -> GNNBaselineTrainer:
    """Prepare causal monthly graphs, train GraphSAGE, and save its checkpoint."""
    print("Preparing standalone monthly H3 GNN baseline...")
    sales = pd.read_csv(config.sales_csv)
    frame = prepare_gnn_sales_dataframe(
        sales,
        market_indicator_cache_path=config.market_indicator_csv,
    )
    graph_data = build_monthly_graph_data(frame)
    print(
        f"Graph: {len(graph_data.cell_ids):,} cells, "
        f"{graph_data.edge_index.shape[1]:,} directed one-ring edges, "
        f"{len(graph_data.month_starts)} monthly causal snapshots"
    )
    print(f"Graph LayerNorm: {config.graph_layer_norm}")
    print(f"Graph residual: {config.graph_residual}")
    trainer = GNNBaselineTrainer(
        graph_data,
        graph_hidden_dim=config.graph_hidden_dim,
        head_hidden_dim=config.head_hidden_dim,
        dropout_rate=config.dropout_rate,
        learning_rate=config.learning_rate,
        random_seed=config.random_seed,
        graph_layer_norm=config.graph_layer_norm,
        graph_residual=config.graph_residual,
    ).fit(
        train_ratio=config.train_ratio,
        epochs=config.epochs,
        patience=config.patience,
        lr_plateau_factor=config.lr_plateau_factor,
        lr_plateau_patience=config.lr_plateau_patience,
        min_learning_rate=config.min_learning_rate,
    )
    trainer.add_predictions_and_metrics()
    trainer.results["training_config"] = config.as_dict()
    directory = trainer.save()
    print(f"Saved GNN baseline: {directory}")
    return trainer


if __name__ == "__main__":
    arguments = parse_args()
    main(replace(
        DEFAULT_CONFIG,
        random_seed=arguments.seed,
        graph_layer_norm=arguments.graph_layer_norm,
        graph_residual=arguments.graph_residual,
    ))
