"""
Enhanced Training Script V3 - RentCast Data
Train V3 model on recent RentCast property sales data
"""
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import json

# Add paths
sys.path.insert(0, os.getcwd() + '/src/pricemodel')

from market_indicators import MarketIndicatorFetcher, TimeFeatureEngineer
from embedding_model_v2 import dataset
from model_manager_v2 import modelmanager

def main():
    print("=" * 80)
    print("Enhanced Real Estate Price Model V3 - Fine-tuning on RentCast Data")
    print("Load existing V3 model and continue training on new data")
    print("=" * 80)
    
    # --- Step 1: Load Data ---
    print("\n[1/6] Loading RentCast sales data...")
    df = pd.read_csv('data/rentcast_recent_house_sales.csv')
    print(f"Loaded {len(df)} records from {df['sale_date'].min()} to {df['sale_date'].max()}")
    
    # Convert sale_nbr to numeric (it's a string in RentCast data)
    # Create a numeric ID by hashing the string or using row index
    if df['sale_nbr'].dtype == 'object':
        print(f"  Converting sale_nbr from string to numeric...")
        # Use row index as numeric sale_nbr
        df['sale_nbr'] = range(1, len(df) + 1)
    
    # Check data quality
    print(f"\nData Quality Check:")
    print(f"  Missing sale_price: {df['sale_price'].isna().sum()}")
    print(f"  Missing lat/lng: {df[['lat', 'lng']].isna().any(axis=1).sum()}")
    print(f"  Missing sqft: {df['sqft'].isna().sum()}")
    print(f"  Missing beds: {df['beds'].isna().sum()}")
    print(f"  Missing baths: {df['baths'].isna().sum()}")
    
    # Fill missing values for beds/baths with median
    if df['beds'].isna().sum() > 0:
        df['beds'] = df['beds'].fillna(df['beds'].median())
        print(f"  Filled missing beds with median: {df['beds'].median()}")
    
    if df['baths'].isna().sum() > 0:
        df['baths'] = df['baths'].fillna(df['baths'].median())
        print(f"  Filled missing baths with median: {df['baths'].median()}")
    
    # Fill missing sqft_lot with median
    if 'sqft_lot' in df.columns and df['sqft_lot'].isna().sum() > 0:
        df['sqft_lot'] = df['sqft_lot'].fillna(df['sqft_lot'].median())
        print(f"  Filled missing sqft_lot with median: {df['sqft_lot'].median()}")
    
    # Fill missing year_built with median
    if 'year_built' in df.columns and df['year_built'].isna().sum() > 0:
        df['year_built'] = df['year_built'].fillna(df['year_built'].median())
        print(f"  Filled missing year_built with median: {df['year_built'].median()}")
    
    # --- Step 2: Add Market Indicators ---
    print("\n[2/6] Fetching market indicators...")
    market_fetcher = MarketIndicatorFetcher()
    df = market_fetcher.merge_indicators_to_sales(df, date_column='sale_date')
    
    # Add local inventory proxy (or load from file if available)
    inventory_file = 'data/local_inventory.csv'  # Optional
    df = market_fetcher.add_local_inventory(df, inventory_file if os.path.exists(inventory_file) else None)
    
    print(f"Added market indicators: mortgage_rate, unemployment_rate")
    
    # --- Step 3: Add Time Features ---
    print("\n[3/6] Engineering time features...")
    time_engineer = TimeFeatureEngineer()
    df = time_engineer.add_continuous_time_features(df, date_column='sale_date')
    df = time_engineer.add_market_momentum_features(df, date_column='sale_date', price_column='sale_price')
    
    print(f"Added continuous time features and market momentum indicators")
    
    # --- Step 4: Prepare Dataset with Community Mapping ---
    print("\n[4/6] Preparing dataset with community mapping...")
    
    # Check if h3_07 exists, if not create it
    if 'h3_07' not in df.columns:
        print("  Creating H3 level 7 indices...")
        import h3
        df['h3_07'] = df.apply(lambda row: h3.latlng_to_cell(row['lat'], row['lng'], 7) 
                               if pd.notna(row['lat']) and pd.notna(row['lng']) else None, axis=1)
    
    # Load community mapping
    community_map_file = 'data/community_map.json'
    if os.path.exists(community_map_file):
        print(f"  Loading community mapping from {community_map_file}")
        with open(community_map_file, 'r') as f:
            community_map = json.load(f)
        
        df['community'] = df['h3_07'].map(community_map).fillna(-1).astype(int)
        
        print(f"  Mapped {len(df[df['community'] != -1])} records to communities")
        print(f"  Unmapped records: {len(df[df['community'] == -1])}")
        print(f"  Unique communities: {df['community'].nunique()}")
        
        # If too many unmapped, warn user
        unmapped_pct = (df['community'] == -1).sum() / len(df) * 100
        if unmapped_pct > 50:
            print(f"\n  ⚠️  Warning: {unmapped_pct:.1f}% of records are unmapped")
            print(f"  This may affect model performance. Consider:")
            print(f"    1. Updating community_map.json with new H3 cells")
            print(f"    2. Or training will assign them to 'unknown' community")
    else:
        print(f"  ⚠️  Community map not found at {community_map_file}")
        print(f"  Using default community assignment from data")
        if 'community' not in df.columns:
            df['community'] = -1
    
    # Prepare data
    data = dataset()
    
    # Check if we'll be loading an existing model
    model_dir = 'outputs/models'
    v3_models = []
    existing_model_path = None
    
    if os.path.exists(model_dir):
        for folder in os.listdir(model_dir):
            folder_path = os.path.join(model_dir, folder)
            if os.path.isdir(folder_path):
                model_file = os.path.join(folder_path, 'model.pth')
                if os.path.exists(model_file):
                    results_file = os.path.join(folder_path, 'results.json')
                    if os.path.exists(results_file):
                        v3_models.append(folder_path)
    
    if v3_models:
        # Sort by timestamp and get most recent
        v3_models.sort(reverse=True)
        existing_model_path = v3_models[0]
        print(f"\n  Found existing V3 model: {existing_model_path}")
        print(f"  Loading vocabularies from existing model...")
        
        # Load existing vocabularies
        with open(os.path.join(existing_model_path, 'community_vocab.json'), 'r') as f:
            existing_community_vocab = json.load(f)
        with open(os.path.join(existing_model_path, 'year_vocab.json'), 'r') as f:
            existing_year_vocab = json.load(f)
        with open(os.path.join(existing_model_path, 'week_vocab.json'), 'r') as f:
            existing_week_vocab = json.load(f)
        
        print(f"    Existing year vocab: {min([int(k) for k in existing_year_vocab.keys() if k != 'unknown'])} to {max([int(k) for k in existing_year_vocab.keys() if k != 'unknown'])}")
        print(f"    Existing communities: {len(existing_community_vocab)}")
        
        # Prepare data WITHOUT creating new vocabularies
        data = data._prepare_data(df, include_market_indicators=True, future_year_buffer=5)
        
        # Override with existing vocabularies
        print(f"  Overriding with existing model vocabularies...")
        data.community_vocab = existing_community_vocab
        data.year_vocab = existing_year_vocab
        data.week_vocab = existing_week_vocab
        data.n_communities = len(existing_community_vocab)
        data.year_length = len(existing_year_vocab)
        data.week_length = len(existing_week_vocab) - 1  # Exclude unknown
        
        # Remap dataframe columns to use existing vocabularies
        data.dataframe['community_index'] = data.dataframe['community'].map(existing_community_vocab).fillna(existing_community_vocab.get("unknown", len(existing_community_vocab)-1)).astype(int)
        
        print(f"    Using existing vocabularies:")
        print(f"      Communities: {data.n_communities}")
        print(f"      Years: {data.year_length}")
        print(f"      Weeks: {data.week_length}")
    else:
        print(f"\n  No existing model found, creating new vocabularies...")
        data = data._prepare_data(df, include_market_indicators=True, future_year_buffer=5)
    
    print(f"\nDataset prepared: {data.length} records, {data.n_communities} communities")
    print(f"Date range: {data.dataframe['sale_date'].min()} to {data.dataframe['sale_date'].max()}")
    print(f"Reference date: {data.reference_date}")
    
    # Report year distribution
    print(f"\nYear Distribution in Data:")
    year_counts = data.dataframe['year'].value_counts().sort_index()
    for year, count in year_counts.items():
        print(f"  {year}: {count} records")
    
    # Report year vocabulary
    print(f"\nYear Vocabulary Mapping:")
    print(f"  Total vocab size: {len(data.year_vocab)}")
    year_vocab_sorted = sorted([(k, v) for k, v in data.year_vocab.items() if k != 'unknown'], 
                               key=lambda x: x[1])
    for year, idx in year_vocab_sorted:
        count = len(data.dataframe[data.dataframe['year'] == int(year)]) if year != 'unknown' else 0
        print(f"  Year {year} -> index {idx} ({count} records)")
    print(f"  'unknown' -> index {data.year_vocab['unknown']}")
    
    # Check for years in data not in vocab
    data_years = set(data.dataframe['year'].unique())
    vocab_years = set([int(k) for k in data.year_vocab.keys() if k != 'unknown'])
    missing_in_vocab = data_years - vocab_years
    if missing_in_vocab:
        print(f"\n  ⚠️  Years in data but NOT in vocab: {sorted(missing_in_vocab)}")
        print(f"     These will be mapped to 'unknown' index")
    else:
        print(f"\n  ✓ All years in data are covered by vocabulary")
    
    # --- Step 5: Load Existing V3 Model and Continue Training ---
    print("\n[5/6] Loading existing V3 model and fine-tuning on RentCast data...")
    
    if existing_model_path:
        print(f"  Loading model from {existing_model_path}...")
        
        # Load the existing model
        model = modelmanager(model_name="property_model_v3_rentcast_finetuned")
        model.load_model(existing_model_path)
        
        # Process new data with the same scalers and vocabularies
        print(f"  Processing new data with existing model's parameters...")
        model.processor(data, scale_mode="use_existing")  # Use existing scalers
        model.split_data(train_ratio=0.8, temporal_split=False)
        
        print(f"  Fine-tuning model on {data.length} new records...")
        print(f"  Using existing architecture:")
        print(f"    Embedding dim: {model.embedding_dim}")
        print(f"    Hidden dim: {model.hidden_dim}")
        print(f"    Communities: {model.n_communities}")
        
        # Continue training with lower learning rate for fine-tuning
        model.train_model(
            embedding_dim=model.embedding_dim,
            hidden_dim=model.hidden_dim,
            property_dim=model.property_dim,
            continuous_time_dim=model.continuous_time_dim,
            market_dim=model.market_dim,
            epochs=20,  # Fewer epochs for fine-tuning
            batch=256,
            learning_rate=0.0001,  # Lower learning rate for fine-tuning
            dropout_rate=0.1,
            estimate_uncertainty=True,
            patience=10
        )
        
        print(f"  ✓ Fine-tuning complete")
        
    else:
        print(f"  Training new model from scratch...")
        
        # Train new model from scratch
        model = modelmanager(model_name="property_model_v3_rentcast")
        model.processor(data)
        model.split_data(train_ratio=0.8, temporal_split=False)
        
        model.train_model(
            embedding_dim=16,
            hidden_dim=32,
            property_dim=3,
            continuous_time_dim=5,
            market_dim=2,
            epochs=20,
            batch=256,
            learning_rate=0.001,
            dropout_rate=0.1,
            estimate_uncertainty=True,
            patience=20
        )

    # --- Step 6: Evaluate and Save ---
    print("\n[6/6] Generating predictions and saving model...")
    model.add_predictions_to_data(return_uncertainty=True)
    
    # Save predictions
    output_file = 'data/rentcast_with_predictions_v3_finetuned.csv'
    model.dataframe.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")
    
    # Save model
    model.save_model()
    
    # --- Performance Summary ---
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY - V3 Fine-tuned on RentCast Data")
    print("=" * 80)
    
    # Overall metrics
    print(f"\nOverall Performance:")
    print(f"  Mean Absolute % Error: {model.dataframe['pct_error'].abs().mean():.2f}%")
    print(f"  Median Absolute % Error: {model.dataframe['pct_error'].abs().median():.2f}%")
    
    if 'price_lower_95' in model.dataframe.columns:
        in_ci = ((model.dataframe['sale_price'] >= model.dataframe['price_lower_95']) & 
                (model.dataframe['sale_price'] <= model.dataframe['price_upper_95']))
        print(f"  95% Confidence Interval Coverage: {in_ci.mean()*100:.1f}%")
    
    # CLS Attention
    if 'cls_attn_community' in model.dataframe.columns:
        print(f"\nCLS Attention Weights:")
        print(f"  Community: {model.dataframe['cls_attn_community'].mean():.3f}")
        print(f"  Year:      {model.dataframe['cls_attn_year'].mean():.3f}")
        print(f"  Week:      {model.dataframe['cls_attn_week'].mean():.3f}")
        print(f"  Property:  {model.dataframe['cls_attn_property'].mean():.3f}")
        print(f"  Time:      {model.dataframe['cls_attn_time'].mean():.3f}")
        print(f"  Market:    {model.dataframe['cls_attn_market'].mean():.3f}")
    
    # Performance by year
    if 'year' in model.dataframe.columns:
        print(f"\nPerformance by Year:")
        yearly_performance = model.dataframe.groupby('year').agg({
            'pct_error': lambda x: x.abs().mean(),
            'sale_price': 'count'
        }).round(2)
        yearly_performance.columns = ['Mean Abs % Error', 'Count']
        print(yearly_performance)
    
    # Top communities
    print(f"\nTop 10 Communities by Volume:")
    top_communities = model.dataframe.groupby('community').agg({
        'pct_error': lambda x: x.abs().mean(),
        'sale_price': 'count'
    }).sort_values('sale_price', ascending=False).head(10).round(2)
    top_communities.columns = ['Mean Abs % Error', 'Count']
    print(top_communities)
    
    # Price range analysis
    print(f"\nPerformance by Price Range:")
    price_bins = [
        ('Low (<$500k)', 0, 500000),
        ('Mid ($500k-$1M)', 500000, 1000000),
        ('High ($1M-$2M)', 1000000, 2000000),
        ('Luxury (>$2M)', 2000000, float('inf'))
    ]
    
    for label, min_price, max_price in price_bins:
        subset = model.dataframe[(model.dataframe['sale_price'] >= min_price) & 
                                 (model.dataframe['sale_price'] < max_price)]
        if len(subset) > 0:
            print(f"  {label}: {len(subset)} records, {subset['pct_error'].abs().mean():.2f}% MAE")
    
    print("\n" + "=" * 80)
    print(f"Model saved to: {model.directory}")
    print("Training complete!")
    print("=" * 80)
    
    print("\nNext Steps:")
    print("1. Review the predictions in:", output_file)
    print("2. Compare with previous V3 model trained on historical data")
    print("3. Use this model for predictions on new listings")
    
    return model


if __name__ == "__main__":
    model = main()
