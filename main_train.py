"""
Train Model with H3 L8 Neighborhood-Aware Community Embeddings

Uses:
- Explicit historical sales CSV and frozen economic-indicator CSV
- H3 Level 8 hexagons
- Neighborhood pooling: each location represented by center + 6 neighbors
- Three pooling strategies: mean, center_weighted, learnable
"""
import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h3
from pricemodel.data_pipeline import DatasetBuilder
from pricemodel.model_manager import ModelManager
from pricemodel.training_config import TrainingConfig


# All model/data defaults are defined in src/pricemodel/training_config.py.
# Keep one visible entry-point instance so IDE navigation and programmatic
# overrides both lead to the exact configuration used by this script.
DEFAULT_CONFIG = TrainingConfig()


def parse_args(argv=None):
    """Parse intentionally small entry-point overrides; other defaults stay typed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_CONFIG.random_seed,
        help="Python/NumPy/PyTorch/DataLoader seed (default: 42)",
    )
    return parser.parse_args(argv)


def main(config=None):
    """Train, verify, evaluate, and report one reproducible model run."""
    config = config or DEFAULT_CONFIG
    print("=" * 70)
    print(f"Configuration: src/pricemodel/training_config.py (seed={config.random_seed})")
    print("Training Model with H3 L Neighborhood-Aware Embeddings")
    print("=" * 70)
    
    # Load and prepare data
    df = pd.read_csv(config.sales_csv)
    data = DatasetBuilder()
    print("\n1. Checking h3 index and mapping communities...")
    data._map_communities(df)

    print("\n1. Preparing dataset features...")
    data._prepare_data(
        market_indicator_cache_path=config.market_indicator_csv
    )
    
    # Check neighborhood mapping
    if 'community_neighbors' in data.dataframe.columns:
        mapped = data.dataframe['community_neighbors'].notna().sum()
        print(f"   ✓ Neighborhood mapping: {mapped}/{len(data.dataframe)} properties")
        if mapped < len(data.dataframe):
            print(f"   Warning: {len(data.dataframe) - mapped} properties have no mapping")
    else:
        print("   ✗ No neighborhood mapping found!")
        return

    # Initialize model manager
    print("\n4. Initializing model manager...")
    manager = ModelManager()
    manager.processor(data, scale_mode="fit")

    print(f"   Device: {manager.device}")
    print(f"   Neighborhood pooling: {manager.use_neighborhood_pooling}")
    print(f"   Communities: {manager.n_communities}")

    # Split data
    print("\n5. Splitting data chronologically (70/30 train/val)...")
    manager.split_data(
        train_ratio=config.train_ratio,
        temporal_split=config.temporal_split,
    )
    print(f"   Train: {len(manager.train_dataset)} samples")
    print(f"   Val:   {len(manager.val_dataset)} samples")

    # Train model
    print("\n6. Training model...")
    print("   Architecture:")
    print(f"   - Attention/token dim: {config.embedding_dim}")
    print(f"   - Community embedding dim: {config.community_embedding_dim}")
    print(f"   - Hidden dim: {config.hidden_dim}")
    print("   - Continuous time: time_trend + annual_sin + annual_cos (dim=3)")
    print(f"   - Dropout: {config.dropout_rate}")
    print(f"   - Learning rate: {config.learning_rate}")
    print(f"   - Mean stage: up to {config.epochs} MSE epochs "
          f"(patience={config.patience}; best epoch restored)")
    print(f"   - Uncertainty stage: up to {config.uncertainty_calibration_epochs} "
          f"NLL epochs (patience={config.uncertainty_patience}; price weights frozen)")
    print("   - Scheduler: ReduceLROnPlateau "
          f"(factor={config.lr_plateau_factor}, "
          f"patience={config.lr_plateau_patience}, "
          f"min LR={config.min_learning_rate:g})")
    print(f"   - Global auxiliary weight: {config.global_aux_weight}")
    print(f"   - Local residual penalty: {config.residual_penalty}")
    print("   - Training: MSE price fitting, then frozen-head NLL calibration")

    manager.results['training_config'] = config.as_dict()
    manager.train_model(**config.train_kwargs(len(manager._PROPERTY_FEATURES)))
    
    
    # Persist the restored best state once, then immediately prove that the
    # serialized tensors are byte-for-byte equivalent before any reporting.
    print("\n7. Saving and verifying the best checkpoint...")
    manager.save_and_reload_for_evaluation()

    # Plot training curves
    print("\n8. Plotting training curves...")
    history = manager.results['diagnostic_history']
    phases = [row['phase'] for row in history]
    x = np.arange(1, len(history) + 1)
    metrics = [
        ('prediction_mse', 'Prediction MSE'),
        ('nll', 'Gaussian NLL'),
        ('mean_log_var', 'Mean log variance'),
        ('coverage_95', '95% interval coverage'),
        ('global_head_mse', 'Global-head MSE'),
        ('mean_abs_residual_adjustment', 'Mean |local adjustment|'),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(13, 12))
    phase_boundary = phases.count('mean') + 0.5
    for ax, (key, title) in zip(axes.flat, metrics):
        train_values = [row['train'][key] for row in history]
        val_values = [row['validation'][key] for row in history]
        if key == 'coverage_95':
            train_values = np.asarray(train_values) * 100
            val_values = np.asarray(val_values) * 100
            ax.axhline(95, color='gray', linestyle=':', linewidth=1)
            ax.set_ylabel('%')
        ax.plot(x, train_values, label='Train', linewidth=1.8)
        ax.plot(x, val_values, label='Validation', linewidth=1.8)
        if 'uncertainty' in phases:
            ax.axvline(phase_boundary, color='black', linestyle='--', alpha=0.5)
        ax.set_title(title)
        ax.set_xlabel('Recorded epoch')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle('Two-stage training diagnostics (dashed line = uncertainty calibration)')
    plt.tight_layout()
    training_curves_path = Path(manager.directory) / 'training_curves.png'
    plt.savefig(training_curves_path, dpi=150)
    print(f"   Saved: {training_curves_path}")
    
    # Generate predictions
    print("\n9. Generating predictions for whole dataset, with uncertainty calculation...")
    manager.add_predictions_to_data(return_uncertainty=True)
    
    # Calculate metrics
    mape = manager.dataframe['pct_error'].abs().mean()
    median_ape = manager.dataframe['pct_error'].abs().median()
    rmse = np.sqrt(((manager.dataframe['predicted_price'] - manager.dataframe['sale_price'])**2).mean())
    
    print(f"\n   Performance Metrics:")
    print(f"   - MAPE:       {mape:.2f}%")
    print(f"   - Median APE: {median_ape:.2f}%")
    print(f"   - RMSE:       ${rmse:,.0f}")
    
    # Metrics change after evaluation, but verified weights do not. Persist the
    # results document separately to avoid rewriting the checkpoint.
    print("\n10. Saving evaluation metrics...")
    manager.save_results()
    print(f"   Model directory: {manager.directory}")
    
    # Save predictions
    output_path = 'data/sales_2020_25_with_predictions.csv'
    manager.dataframe.to_csv(output_path, index=False)
    print(f"   Predictions saved: {output_path}")
    print(f"   Total records: {len(manager.dataframe)}")
    
    # Report data sources
    if 'data_source' in manager.dataframe.columns:
        source_counts = manager.dataframe['data_source'].value_counts()
        print(f"   Data sources:")
        for source, count in source_counts.items():
            print(f"     - {source}: {count} records")
    else:
        print(f"   Note: Includes main sales data + RentCast data (if available)")
    
    # Create summary report
    print("\n11. Creating summary report...")
    summary = {
        'model_version': 'current',
        'h3_level': 8,
        'neighborhood_pooling': True,
        'pooling_strategy': manager.pooling_strategy,
        'estimate_uncertainty': manager.estimate_uncertainty,
        'num_communities': manager.n_communities,
        'num_samples': len(manager.dataframe),
        'mape': f"{mape:.2f}%",
        'median_ape': f"{median_ape:.2f}%",
        'rmse': f"${rmse:,.0f}",
        'embedding_dim': manager.embedding_dim,
        'community_embedding_dim': manager.community_embedding_dim,
        'hidden_dim': manager.hidden_dim,
        'epochs_trained': len(manager.results['train_losses']),
        'best_epoch': manager.results['best_epoch'],
        'best_uncertainty_epoch': manager.results.get('best_uncertainty_epoch'),
        'best_validation_nll': manager.results.get('best_val_nll'),
        'saved_model_val_loss': f"{manager.results['best_val_loss']:.4f}",
        'evaluation_checkpoint_reloaded': manager.results.get(
            'evaluation_checkpoint_reloaded', False
        ),
        'last_observed_train_loss': f"{manager.results['train_losses'][-1]:.4f}",
        'last_observed_val_loss': f"{manager.results['val_losses'][-1]:.4f}",
        'model_directory': str(manager.directory)
    }
    
    summary_path = Path(manager.directory) / 'model_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("Model Training Summary\n")
        f.write("=" * 50 + "\n\n")
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")
    
    print(f"   Summary saved: {summary_path}")
    
    print("\n" + "=" * 70)
    print("✓ Model Training Complete!")
    print("=" * 70)
    
    return manager


if __name__ == "__main__":
    arguments = parse_args()
    manager = main(replace(DEFAULT_CONFIG, random_seed=arguments.seed))
