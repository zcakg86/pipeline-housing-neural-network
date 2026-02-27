"""
Enhanced Training Script V2
Trains the improved model with market indicators and continuous time features
"""
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add paths
sys.path.insert(0, os.getcwd() + '/src/pricemodel')

from market_indicators import MarketIndicatorFetcher, TimeFeatureEngineer
from embedding_model_v2 import dataset
from model_manager_v2 import modelmanager

def main():
    print("=" * 80)
    print("Enhanced Real Estate Price Model V2 - Training")
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
    
    # --- Step 4: Prepare Dataset ---
    print("\n[4/6] Preparing dataset...")
    # # Load community mapping
    # import json
    # with open('data/community_map.json', 'r') as f:
    #     community_map = json.load(f)
    
    # df['community'] = df['h3_07'].map(community_map).astype(str)
    
    # Prepare data
    data = dataset()
    data = data._prepare_data(df, include_market_indicators=True, future_year_buffer=5)
    
    print(f"Dataset prepared: {data.length} records, {data.n_communities} communities")
    print(f"Date range: {data.dataframe['sale_date'].min()} to {data.dataframe['sale_date'].max()}")
    print(f"Reference date: {data.reference_date}")
    
    # --- Step 5: Train Model ---
    print("\n[5/6] Training enhanced model...")
    model = modelmanager()
    model.processor(data)
    model.split_data(train_ratio=0.8, temporal_split=False)  # Use temporal_split=True for chronological split
    
    # Train with enhanced architecture
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
        patience=10
    )
    
    # --- Step 6: Evaluate and Save ---
    print("\n[6/6] Generating predictions and saving model...")
    model.add_predictions_to_data(return_uncertainty=True)
    
    # Save predictions
    output_file = 'data/sales_2020_25_with_predictions_v2.csv'
    model.dataframe.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")
    
    # Save model
    model.save_model()
    
    # --- Performance Summary ---
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    
    # Overall metrics
    print(f"\nOverall Performance:")
    print(f"  Mean Absolute % Error: {model.dataframe['pct_error'].abs().mean():.2f}%")
    print(f"  Median Absolute % Error: {model.dataframe['pct_error'].abs().median():.2f}%")
    
    if 'price_lower_95' in model.dataframe.columns:
        in_ci = ((model.dataframe['sale_price'] >= model.dataframe['price_lower_95']) & 
                (model.dataframe['sale_price'] <= model.dataframe['price_upper_95']))
        print(f"  95% Confidence Interval Coverage: {in_ci.mean()*100:.1f}%")
    
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
