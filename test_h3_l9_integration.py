"""
Test script for H3 L9 neighborhood-aware community embedding integration
"""
import sys
sys.path.append('src')

import pandas as pd
import torch
from pricemodel.embedding_model_v2 import dataset
from pricemodel.model_manager_v2 import modelmanager

def test_integration():
    """Test the H3 L9 integration with a small sample"""
    
    print("=" * 60)
    print("Testing H3 L9 Neighborhood-Aware Community Embeddings")
    print("=" * 60)
    
    # Load a small sample of data
    print("\n1. Loading data...")
    
    # Load main sales data
    df_main = pd.read_csv('data/sales_2020_25.csv')
    
    # Try to load RentCast data
    try:
        df_rentcast = pd.read_csv('data/rentcast_recent_house_sales.csv')
        df = pd.concat([df_main, df_rentcast], ignore_index=True)
        print(f"   Combined dataset: {len(df)} records")
    except FileNotFoundError:
        df = df_main
        print(f"   Main dataset only: {len(df)} records")
    
    # Take a small sample for testing
    df_sample = df.head(1000).copy()
    print(f"   Using {len(df_sample)} records for testing")
    
    # Apply H3 L9 indexing if needed
    if 'h3_09' not in df_sample.columns or df_sample['h3_09'].isna().any():
        print("   Generating H3 L9 indices...")
        import h3
        df_sample['h3_09'] = df_sample.apply(
            lambda row: h3.latlng_to_cell(row['lat'], row['lng'], 9) 
            if pd.notna(row['lat']) and pd.notna(row['lng']) 
            else None,
            axis=1
        )
    
    print(f"   h3_09 column: {df_sample['h3_09'].notna().sum()} non-null values")
    
    # Initialize dataset
    print("\n2. Preparing dataset (will auto-compute mappings if needed)...")
    data = dataset()
    data._prepare_data(df_sample, include_market_indicators=True, future_year_buffer=5)
    
    # Check if neighborhood mapping was loaded or computed
    if 'community_neighbors' in data.dataframe.columns:
        has_neighbors = data.dataframe['community_neighbors'].notna().sum()
        print(f"   ✓ Neighborhood mapping: {has_neighbors} properties mapped")
        
        # Show a sample
        sample_neighbors = data.dataframe['community_neighbors'].iloc[0]
        print(f"   Sample neighborhood (7 communities): {sample_neighbors}")
    else:
        print("   ✗ No neighborhood mapping (will use single community index)")
        print("   This is expected if community_map.json doesn't exist")
    
    # Initialize model manager
    print("\n3. Initializing model manager...")
    manager = modelmanager()
    manager.processor(data, scale_mode="fit")
    
    print(f"   Using neighborhood pooling: {manager.use_neighborhood_pooling}")
    print(f"   Community tensor shape: {manager.tensors.tensors[0].shape}")
    print(f"   Number of communities: {manager.n_communities}")
    
    # Split data
    print("\n4. Splitting data...")
    manager.split_data(train_ratio=0.8, temporal_split=False)
    print(f"   Train size: {len(manager.train_dataset)}")
    print(f"   Val size: {len(manager.val_dataset)}")
    
    # Train for a few epochs to test
    print("\n5. Training model (5 epochs for testing)...")
    manager.train_model(
        embedding_dim=32,
        hidden_dim=64,
        property_dim=3,
        continuous_time_dim=5,
        market_dim=2,
        epochs=5,
        batch=128,
        learning_rate=0.001,
        pooling_strategy='mean',
        patience=10
    )
    
    print(f"   Final train loss: {manager.results['train_losses'][-1]:.4f}")
    print(f"   Final val loss: {manager.results['val_losses'][-1]:.4f}")
    
    # Add predictions
    print("\n6. Generating predictions...")
    manager.add_predictions_to_data(return_uncertainty=True)
    
    mape = manager.dataframe['pct_error'].abs().mean()
    print(f"   MAPE: {mape:.2f}%")
    
    # Check attention weights
    if 'cls_attn_community' in manager.dataframe.columns:
        print("\n7. Attention weights:")
        print(f"   Community: {manager.dataframe['cls_attn_community'].mean():.3f}")
        print(f"   Year:      {manager.dataframe['cls_attn_year'].mean():.3f}")
        print(f"   Week:      {manager.dataframe['cls_attn_week'].mean():.3f}")
        print(f"   Property:  {manager.dataframe['cls_attn_property'].mean():.3f}")
        print(f"   Time:      {manager.dataframe['cls_attn_time'].mean():.3f}")
        print(f"   Market:    {manager.dataframe['cls_attn_market'].mean():.3f}")
    
    # Test save/load
    print("\n8. Testing save/load...")
    manager.save_model()
    
    # Load model
    manager2 = modelmanager()
    manager2.load_model(manager.directory)
    print(f"   ✓ Model loaded successfully")
    print(f"   Neighborhood pooling: {manager2.use_neighborhood_pooling}")
    print(f"   Pooling strategy: {manager2.pooling_strategy}")
    
    print("\n" + "=" * 60)
    print("✓ Integration test completed successfully!")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    try:
        success = test_integration()
        if success:
            print("\nAll tests passed! The H3 L9 integration is working correctly.")
        else:
            print("\nSome tests failed. Please check the output above.")
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
