"""
Train V4 Model with H3 L9 Neighborhood-Aware Community Embeddings

This version uses:
- Combined dataset: sales_2020_25.csv + rentcast_recent_house_sales.csv
- H3 Level 9 hexagons (higher resolution than L7)
- Neighborhood pooling: each location represented by center + 6 neighbors
- Three pooling strategies available: mean, center_weighted, learnable
"""
import sys
sys.path.append('src')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h3
from pricemodel.embedding_model_v2 import dataset
from pricemodel.model_manager_v2 import modelmanager

def load_and_prepare_data():
    """Load and combine sales data with RentCast data, apply H3 L9 indexing"""
    
    print("\n1. Loading and combining datasets...")
    
    # Load main sales data
    print("   Loading data/sales_2020_25.csv...")
    df_main = pd.read_csv('data/sales_2020_25.csv')
    print(f"   - Main dataset: {len(df_main)} records")
    
    # Load RentCast data
    rentcast_path = 'data/rentcast_recent_house_sales.csv'
    try:
        print(f"   Loading {rentcast_path}...")
        df_rentcast = pd.read_csv(rentcast_path)
        print(f"   - RentCast dataset: {len(df_rentcast)} records")
        
        # Combine datasets
        df = pd.concat([df_main, df_rentcast], ignore_index=True)
        print(f"   ✓ Combined dataset: {len(df)} records")
    except FileNotFoundError:
        print(f"   Warning: {rentcast_path} not found, using main dataset only")
        df = df_main
    
    # Apply H3 Level 9 indexing
    print("\n   Applying H3 Level 9 indexing...")
    if 'h3_09' not in df.columns or df['h3_09'].isna().any():
        print("   Generating H3 L9 indices from lat/lng...")
        df['h3_09'] = df.apply(
            lambda row: h3.latlng_to_cell(row['lat'], row['lng'], 9) 
            if pd.notna(row['lat']) and pd.notna(row['lng']) 
            else None,
            axis=1
        )
        print(f"   ✓ Generated H3 L9 indices for {df['h3_09'].notna().sum()} properties")
    else:
        print(f"   ✓ H3 L9 indices already present: {df['h3_09'].notna().sum()} properties")
    
    # Show date range
    df['sale_date'] = pd.to_datetime(df['sale_date'])
    print(f"\n   Date range: {df['sale_date'].min()} to {df['sale_date'].max()}")
    
    return df


def main():
    print("=" * 70)
    print("Training V4 Model with H3 L9 Neighborhood-Aware Embeddings")
    print("=" * 70)
    
    # Load and prepare data
    df = load_and_prepare_data()
    
    # Check for h3_09 column
    if 'h3_09' not in df.columns:
        print("\n   ERROR: h3_09 column not found after preparation!")
        return
    
    h3_count = df['h3_09'].notna().sum()
    print(f"   H3 L9 indices: {h3_count}/{len(df)} properties")
    
    # Prepare dataset (will auto-compute neighbor mappings if needed)
    print("\n2. Preparing dataset with H3 L9 neighborhood mapping...")
    data = dataset()
    data._prepare_data(df, include_market_indicators=True, future_year_buffer=5)
    
    # Check neighborhood mapping
    if 'community_neighbors' in data.dataframe.columns:
        mapped = data.dataframe['community_neighbors'].notna().sum()
        print(f"   ✓ Neighborhood mapping: {mapped}/{len(data.dataframe)} properties")
        if mapped < len(data.dataframe):
            print(f"   Warning: {len(data.dataframe) - mapped} properties have no mapping")
    else:
        print("   ✗ No neighborhood mapping found!")
        print("   The model will automatically compute it from the data.")
        return
    
    # Initialize model manager
    print("\n3. Initializing model manager...")
    manager = modelmanager()
    manager.processor(data, scale_mode="fit")
    
    print(f"   Device: {manager.device}")
    print(f"   Neighborhood pooling: {manager.use_neighborhood_pooling}")
    print(f"   Community tensor shape: {manager.tensors.tensors[0].shape}")
    print(f"   Communities: {manager.n_communities}")
    print(f"   Year vocab size: {manager.year_length}")
    
    # Split data
    print("\n4. Splitting data (80/20 train/val)...")
    manager.split_data(train_ratio=0.8, temporal_split=False)
    print(f"   Train: {len(manager.train_dataset)} samples")
    print(f"   Val:   {len(manager.val_dataset)} samples")
    
    # Train model
    print("\n5. Training V4 model...")
    print("   Architecture:")
    print("   - Embedding dim: 128")
    print("   - Hidden dim: 256")
    print("   - Pooling strategy: mean (center + 6 neighbors)")
    print("   - Dropout: 0.2")
    print("   - Learning rate: 0.001")
    print("   - Epochs: 50 (with early stopping, patience=15)")
    
    manager.train_model(
        embedding_dim=128,
        hidden_dim=256,
        property_dim=3,
        continuous_time_dim=5,
        market_dim=2,
        epochs=50,
        batch=256,
        learning_rate=0.001,
        dropout_rate=0.2,
        estimate_uncertainty=True,
        pooling_strategy='mean',  # Options: 'mean', 'center_weighted', 'learnable'
        patience=15
    )
    
    # Plot training curves
    print("\n6. Plotting training curves...")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(manager.results['train_losses'], label='Train Loss', linewidth=2)
    ax.plot(manager.results['val_losses'], label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('V4 Model Training (H3 L9 Neighborhood Pooling)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/v4_training_curves.png', dpi=150)
    print("   Saved: outputs/v4_training_curves.png")
    
    # Generate predictions
    print("\n7. Generating predictions...")
    manager.add_predictions_to_data(return_uncertainty=True)
    
    # Calculate metrics
    mape = manager.dataframe['pct_error'].abs().mean()
    median_ape = manager.dataframe['pct_error'].abs().median()
    rmse = np.sqrt(((manager.dataframe['predicted_price'] - manager.dataframe['sale_price'])**2).mean())
    
    print(f"\n   Performance Metrics:")
    print(f"   - MAPE:       {mape:.2f}%")
    print(f"   - Median APE: {median_ape:.2f}%")
    print(f"   - RMSE:       ${rmse:,.0f}")
    
    # Attention weights
    if 'cls_attn_community' in manager.dataframe.columns:
        print(f"\n   Attention Weights (average):")
        print(f"   - Community: {manager.dataframe['cls_attn_community'].mean():.3f}")
        print(f"   - Year:      {manager.dataframe['cls_attn_year'].mean():.3f}")
        print(f"   - Week:      {manager.dataframe['cls_attn_week'].mean():.3f}")
        print(f"   - Property:  {manager.dataframe['cls_attn_property'].mean():.3f}")
        print(f"   - Time:      {manager.dataframe['cls_attn_time'].mean():.3f}")
        print(f"   - Market:    {manager.dataframe['cls_attn_market'].mean():.3f}")
    
    # Save model
    print("\n8. Saving model...")
    manager.save_model()
    print(f"   Model directory: {manager.directory}")
    
    # Save predictions
    output_path = 'data/sales_2020_25_with_predictions_v4.csv'
    manager.dataframe.to_csv(output_path, index=False)
    print(f"   Predictions saved: {output_path}")
    
    # Create summary report
    print("\n9. Creating summary report...")
    summary = {
        'model_version': 'V4',
        'h3_level': 9,
        'neighborhood_pooling': True,
        'pooling_strategy': manager.pooling_strategy,
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
    
    with open('outputs/v4_model_summary.txt', 'w') as f:
        f.write("V4 Model Training Summary\n")
        f.write("=" * 50 + "\n\n")
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")
    
    print("   Summary saved: outputs/v4_model_summary.txt")
    
    print("\n" + "=" * 70)
    print("✓ V4 Model Training Complete!")
    print("=" * 70)
    print(f"\nKey improvements over V3:")
    print(f"  - Higher resolution: H3 L9 (vs L7)")
    print(f"  - Neighborhood context: 7 communities per location")
    print(f"  - Smoother spatial predictions")
    print(f"  - Better boundary handling")
    
    return manager


if __name__ == "__main__":
    manager = main()
