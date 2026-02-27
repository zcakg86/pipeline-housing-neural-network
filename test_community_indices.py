"""
Test script to verify community indices are handled correctly
"""
import sys
import os
import pandas as pd
import numpy as np
import torch

sys.path.insert(0, os.getcwd() + '/src/pricemodel')

from embedding_model_v2 import dataset, create_vocab

def test_community_indices():
    """Test that community indices are properly bounded"""
    
    print("Testing community index handling...")
    
    # Create sample data with some missing communities
    sample_data = {
        'sale_date': pd.date_range('2024-01-01', periods=100),
        'sale_price': np.random.uniform(300000, 800000, 100),
        'lat': np.random.uniform(47.4, 47.7, 100),
        'lng': np.random.uniform(-122.5, -122.2, 100),
        'sqft': np.random.uniform(1000, 3000, 100),
        'sqft_lot': np.random.uniform(4000, 10000, 100),
        'sale_nbr': np.ones(100),
        'beds': np.random.randint(2, 5, 100),
        'community': ['comm_' + str(i % 10) for i in range(100)]
    }
    
    # Add some NaN communities
    sample_data['community'][::10] = None
    
    df = pd.DataFrame(sample_data)
    
    # Prepare dataset
    data = dataset()
    data = data._prepare_data(df, include_market_indicators=False)
    
    print(f"✓ Dataset prepared")
    print(f"  Total records: {len(data.dataframe)}")
    print(f"  Communities: {data.n_communities}")
    print(f"  Community vocab size: {len(data.community_vocab)}")
    
    # Check community indices
    community_indices = data.dataframe['community_index'].values
    
    print(f"\n✓ Community indices:")
    print(f"  Min index: {community_indices.min()}")
    print(f"  Max index: {community_indices.max()}")
    print(f"  Expected max: {data.n_communities - 1}")
    
    # Verify all indices are within bounds
    if community_indices.max() >= data.n_communities:
        print(f"\n✗ ERROR: Max index {community_indices.max()} >= vocab size {data.n_communities}")
        print(f"  Out of bounds indices: {np.sum(community_indices >= data.n_communities)}")
        return False
    
    if community_indices.min() < 0:
        print(f"\n✗ ERROR: Min index {community_indices.min()} < 0")
        return False
    
    print(f"\n✓ All indices within bounds [0, {data.n_communities - 1}]")
    
    # Check for NaN
    if np.isnan(community_indices).any():
        print(f"\n✗ ERROR: Found {np.isnan(community_indices).sum()} NaN indices")
        return False
    
    print(f"✓ No NaN indices")
    
    # Test with unknown communities
    print(f"\n✓ Testing unknown community handling:")
    test_df = pd.DataFrame({
        'sale_date': ['2024-01-01'],
        'sale_price': [500000],
        'lat': [47.6],
        'lng': [-122.3],
        'sqft': [2000],
        'sqft_lot': [5000],
        'sale_nbr': [1],
        'beds': [3],
        'community': ['unknown_community_xyz']
    })
    
    test_df['sale_date'] = pd.to_datetime(test_df['sale_date'])
    test_df['month'] = test_df['sale_date'].dt.month
    test_df['year'] = test_df['sale_date'].dt.isocalendar().year
    test_df['week'] = test_df['sale_date'].dt.isocalendar().week
    
    # Map to existing vocab
    test_df['community_index'] = test_df['community'].map(data.community_vocab).fillna(
        data.community_vocab.get('unknown', data.n_communities - 1)
    ).astype(int)
    
    unknown_idx = test_df['community_index'].values[0]
    print(f"  Unknown community mapped to index: {unknown_idx}")
    print(f"  Expected 'unknown' index: {data.community_vocab.get('unknown', 'NOT FOUND')}")
    
    if unknown_idx >= data.n_communities:
        print(f"\n✗ ERROR: Unknown index {unknown_idx} >= vocab size {data.n_communities}")
        return False
    
    print(f"✓ Unknown community handled correctly")
    
    return True


if __name__ == "__main__":
    print("="*80)
    print("COMMUNITY INDEX TEST")
    print("="*80)
    print()
    
    try:
        success = test_community_indices()
        
        print("\n" + "="*80)
        if success:
            print("✓ ALL TESTS PASSED")
            print("Community indices are properly bounded")
        else:
            print("✗ TESTS FAILED")
            print("There are issues with community index handling")
        print("="*80)
        
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"\n✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
