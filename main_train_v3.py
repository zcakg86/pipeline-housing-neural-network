"""
Enhanced Training Script V3
Reintroduces community mapping from H3 level 7
Trains for 20 epochs to compare with V2
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
    print("Enhanced Real Estate Price Model V3 - Training")
    print("With Community Mapping from H3 Level 7")
    print("=" * 80)
    
    # --- Step 1: Load Data ---
    print("\n[1/6] Loading sales data...")
    df = pd.read_csv('data/sales_202025.csv')
    print(f"Loaded {len(df)} records from {df['sale_date'].min()} to {df['sale_date'].max()}")
    
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
    else:
        print(f"  ⚠️  Community map not found at {community_map_file}")
        print(f"  Using default community assignment from data")
    
    # Prepare data
    data = dataset()
    data = data._prepare_data(df, include_market_indicators=True, future_year_buffer=5)
    
    print(f"Dataset prepared: {data.length} records, {data.n_communities} communities")
    print(f"Date range: {data.dataframe['sale_date'].min()} to {data.dataframe['sale_date'].max()}")
    print(f"Reference date: {data.reference_date}")
    
    # --- Step 5: Train Model ---
    print("\n[5/6] Training enhanced model V3...")
    model = modelmanager(model_name="property_model_v3")
    model.processor(data)
    model.split_data(train_ratio=0.8, temporal_split=False)
    
    # Train with same architecture as V2 but for exactly 20 epochs
    model.train_model(
        embedding_dim=16,
        hidden_dim=32,
        property_dim=3,  # sqft, sqft_lot, beds
        continuous_time_dim=5,  # time_trend, sin_day, cos_day, sin_month, cos_month
        market_dim=2,  # mortgage_rate, unemployment_rate
        epochs=20,
        batch=256,
        learning_rate=0.001,
        dropout_rate=0.1,
        estimate_uncertainty=True,
        patience=20  # Set patience to 20 to ensure full 20 epochs
    )
    
    # --- Step 6: Evaluate and Save ---
    print("\n[6/6] Generating predictions and saving model...")
    model.add_predictions_to_data(return_uncertainty=True)
    
    # Save predictions
    output_file = 'data/sales_2020_25_with_predictions_v3.csv'
    model.dataframe.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")
    
    # Save model
    model.save_model()
    
    # --- Performance Summary ---
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY - V3")
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
    print(f"\nPerformance by Year:")
    yearly_performance = model.dataframe.groupby('year').agg({
        'pct_error': lambda x: x.abs().mean(),
        'sale_price': 'count'
    }).round(2)
    yearly_performance.columns = ['Mean Abs % Error', 'Count']
    print(yearly_performance)
    
    # Performance on recent data (2024-2025)
    recent_data = model.dataframe[model.dataframe['year'] >= 2024]
    if len(recent_data) > 0:
        print(f"\nPerformance on Recent Data (2024-2025):")
        print(f"  Records: {len(recent_data)}")
        print(f"  Mean Absolute % Error: {recent_data['pct_error'].abs().mean():.2f}%")
        print(f"  Median Absolute % Error: {recent_data['pct_error'].abs().median():.2f}%")
    
    # Top communities
    print(f"\nTop 10 Communities by Volume:")
    top_communities = model.dataframe.groupby('community').agg({
        'pct_error': lambda x: x.abs().mean(),
        'sale_price': 'count'
    }).sort_values('sale_price', ascending=False).head(10).round(2)
    top_communities.columns = ['Mean Abs % Error', 'Count']
    print(top_communities)
    
    print("\n" + "=" * 80)
    print(f"Model saved to: {model.directory}")
    print("Training complete!")
    print("=" * 80)
    
    return model


if __name__ == "__main__":
    model = main()
