"""
Train Model with H3 L9 Neighborhood-Aware Community Embeddings

Uses:
- Combined dataset: sales_2020_25.csv + rentcast_recent_house_sales.csv
- H3 Level 8 hexagons
- Neighborhood pooling: each location represented by center + 6 neighbors
- Three pooling strategies: mean, center_weighted, learnable
"""
import sys
sys.path.append('src')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h3
from pricemodel.embedding_model import dataset
from pricemodel.model_manager import modelmanager


def main():
    print("=" * 70)
    print("Training Model with H3 L Neighborhood-Aware Embeddings")
    print("=" * 70)
    
    # Load and prepare data
    df = pd.read_csv('data/sales_2020_25.csv')
    data = dataset()
    print("\n1. Checking h3 index and mapping communities...")
    data._map_communities(df)

    print("\n1. Preparing dataset features...")
    data._prepare_data(
        market_indicator_cache_path='data/market_indicators/fred_indicators.csv'
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
    manager = modelmanager()
    manager.processor(data, scale_mode="fit")

    print(f"   Device: {manager.device}")
    print(f"   Neighborhood pooling: {manager.use_neighborhood_pooling}")
    print(f"   Communities: {manager.n_communities}")
    print(f"   Year vocab size: {manager.year_length}")

    # Split data
    print("\n5. Splitting data (80/20 train/val)...")
    manager.split_data(train_ratio=0.7, temporal_split=False)
    print(f"   Train: {len(manager.train_dataset)} samples")
    print(f"   Val:   {len(manager.val_dataset)} samples")

    # Train model
    print("\n6. Training model...")
    print("   Architecture:")
    print("   - Embedding dim: 128")
    print("   - Hidden dim: 256")
    print("   - Continuous time: time_trend only (dim=1)")
    print("   - Dropout: 0.2")
    print("   - Learning rate: 0.0003")
    print("   - Epochs: 50 (early stopping patience=15)")
    print("   - Loss: MSE (stable, comparable train/val)")

    manager.train_model(
        embedding_dim=128,
        hidden_dim=256,
        property_dim=3,
        continuous_time_dim=1,
        market_dim=2,
        epochs=50,
        batch=256,
        learning_rate=0.0003,
        dropout_rate=0.2,
        estimate_uncertainty=False, 
        patience=20
    )
    
    
    # Plot training curves
    print("\n7. Plotting training curves...")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(manager.results['train_losses'], label='Train Loss', linewidth=2)
    ax.plot(manager.results['val_losses'], label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('V4 Model Training (H3 L9 Neighborhood Pooling)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(manager.directory + '/training_curves.png', dpi=150)
    print("   Saved: " + manager.directory + "/training_curves.png")
    
    # Generate predictions
    print("\n8. Generating predictions for whole dataset, with uncertainty calculation...")
    manager.add_predictions_to_data(return_uncertainty=True)
    
    # Calculate metrics
    mape = manager.dataframe['pct_error'].abs().mean()
    median_ape = manager.dataframe['pct_error'].abs().median()
    rmse = np.sqrt(((manager.dataframe['predicted_price'] - manager.dataframe['sale_price'])**2).mean())
    
    print(f"\n   Performance Metrics:")
    print(f"   - MAPE:       {mape:.2f}%")
    print(f"   - Median APE: {median_ape:.2f}%")
    print(f"   - RMSE:       ${rmse:,.0f}")
    
    # Save model
    print("\n9. Saving model...")
    manager.save_model()
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
    print("\n10. Creating summary report...")
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
        'hidden_dim': manager.hidden_dim,
        'epochs_trained': len(manager.results['train_losses']),
        'final_train_loss': f"{manager.results['train_losses'][-1]:.4f}",
        'final_val_loss': f"{manager.results['val_losses'][-1]:.4f}",
        'model_directory': str(manager.directory)
    }
    
    with open(manager.directory + '/model_summary.txt', 'w') as f:
        f.write("Model Training Summary\n")
        f.write("=" * 50 + "\n\n")
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")
    
    print("   Summary saved: " + manager.directory + "/model_summary.txt")
    
    print("\n" + "=" * 70)
    print("✓ Model Training Complete!")
    print("=" * 70)
    
    return manager


if __name__ == "__main__":
    manager = main()
