"""
Load trained Pytorch Model and retrain and save.
"""
import sys
sys.path.append('src')

import os
import json
import joblib
import torch
import numpy as np
import pandas as pd
import h3
from pathlib import Path
from glob import glob

# ── 1. Find latest model ──────────────────────────────────────────────────────
def find_latest_model():
    dirs = sorted(glob('outputs/models/*/model.pth'))
    if not dirs:
        raise FileNotFoundError("No trained model found in outputs/models/")
    return Path(dirs[-1]).parent

model_dir = find_latest_model()
print(f"Latest model loaded from: {model_dir}")


# ── 2. Load model via model manager ──────────────────────────────────────────
from pricemodel.model_manager import modelmanager
from pricemodel.embedding_model import dataset
manager = modelmanager()
manager.load_model(model_dir)

print(f"  Neighborhood pooling: {manager.use_neighborhood_pooling}")
print(f"  Pooling strategy:     {manager.pooling_strategy}")
print(f"  Communities:          {manager.n_communities}")

# Load and prepare data
def load_and_prepare_data():
    """Load and combine sales data with RentCast data, apply H3 L9 indexing"""
    
    print("\n1. Loading and combining datasets...")
    
    # Load main sales data
    print("   Loading data/sales_2020_25.csv...")
    df = pd.read_csv('data/sales_2020_25.csv')
    
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

df = load_and_prepare_data()
data = dataset()
data._map_communities()
data._prepare_data()

manager.processor(data)
manager.split_data(train_ratio=0.7, temporal_split=False)

manager.train_model(
    embedding_dim=128,
    hidden_dim=256,
    property_dim=3,
    continuous_time_dim=1,
    market_dim=2,
    epochs=5,
    batch=256,
    learning_rate=0.0003,
    dropout_rate=0.2,
    estimate_uncertainty=False,   # MSE for training — uncertainty head used at inference only
    pooling_strategy='mean',
    patience=20
)

manager.save_model()
