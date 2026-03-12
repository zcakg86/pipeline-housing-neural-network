"""
Generate price predictions for Zillow listings using the latest trained model
"""
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.getcwd() + '/src/pricemodel')

from model_manager_v2 import modelmanager
from embedding_model_v2 import dataset


def find_latest_model():
    """Find the most recent model in outputs/models/"""
    models_dir = Path('outputs/models')
    
    if not models_dir.exists():
        raise FileNotFoundError("No models directory found at outputs/models/")
    
    # Get all subdirectories (model timestamps)
    model_dirs = [d for d in models_dir.iterdir() if d.is_dir()]
    
    if not model_dirs:
        raise FileNotFoundError("No trained models found in outputs/models/")
    
    # Sort by directory name (timestamp format YYYYMMDD_HHMMSS)
    latest_model = sorted(model_dirs, key=lambda x: x.name)[-1]
    
    return latest_model


def load_and_predict():
    """Load latest model and generate predictions for Zillow data"""
    
    print("="*80)
    print("ZILLOW PRICE PREDICTIONS USING LATEST MODEL")
    print("="*80)
    
    # Find and load latest model
    print(f"\n[1/5] Finding latest model...")
    model_dir = find_latest_model()
    print(f"  Found: {model_dir}")
    
    # Check if it's a V4 model (has neighborhood pooling)
    model_pth = model_dir / 'model.pth'
    if model_pth.exists():
        import torch
        checkpoint = torch.load(model_pth, map_location='cpu')
        is_v4 = checkpoint.get('use_neighborhood_pooling', False)
        pooling_strategy = checkpoint.get('pooling_strategy', 'N/A')
        model_version = 'V4' if is_v4 else 'V3'
        print(f"  Model version: {model_version}")
        if is_v4:
            print(f"  Pooling strategy: {pooling_strategy}")
    else:
        model_version = 'Unknown'
    
    print(f"\n  Loading model from {model_dir}...")
    
    model = modelmanager()
    model.load_model(model_dir)
    
    print(f"  ✓ Model loaded successfully")
    print(f"  Communities: {model.n_communities}")
    print(f"  Year vocab size: {model.year_length}")
    print(f"  Week vocab size: {model.week_length}")
    if hasattr(model, 'use_neighborhood_pooling'):
        print(f"  Neighborhood pooling: {model.use_neighborhood_pooling}")
    
    # Load prepared Zillow data
    zillow_file = 'data/zillow_prepared_for_prediction.csv'
    print(f"\n[2/5] Loading prepared Zillow data from {zillow_file}...")
    
    df = pd.read_csv(zillow_file)
    print(f"  ✓ Loaded {len(df)} properties")
    
    # Convert sale_date to datetime
    df['sale_date'] = pd.to_datetime(df['sale_date'])
    
    # Data quality check
    print(f"\nData Quality Check:")
    print(f"  Missing sale_price: {df['sale_price'].isna().sum()}")
    print(f"  Missing lat/lng: {df[['lat', 'lng']].isna().any(axis=1).sum()}")
    print(f"  Missing sqft: {df['sqft'].isna().sum()}")
    print(f"  Missing beds: {df['beds'].isna().sum()}")
    print(f"  Missing baths: {df['baths'].isna().sum()}")
    if 'community' in df.columns:
        print(f"  Unmapped communities: {(df['community'] == -1).sum()}")
    
    # Prepare dataset using model's existing vocabularies
    print(f"\n[3/5] Preparing dataset with model's vocabularies...")
    
    # Create dataset object with existing vocabularies
    data = dataset()
    
    # Set reference date from model
    data.reference_date = model.reference_date
    
    # Load existing vocabularies from model
    data.community_vocab = model.community_vocab
    data.year_vocab = model.year_vocab
    data.week_vocab = model.week_vocab
    
    # Calculate vocab sizes
    data.n_communities = len(model.community_vocab)
    data.year_length = len(model.year_vocab)
    data.week_length = len(model.week_vocab)
    
    print(f"  Using existing vocabularies:")
    print(f"    Communities: {data.n_communities}")
    print(f"    Years: {data.year_length} (range: {min(model.year_vocab.keys())}-{max(model.year_vocab.keys())})")
    print(f"    Weeks: {data.week_length}")
    
    # Prepare data (this will map to existing vocabularies)
    data = data._prepare_data(df, include_market_indicators=True, future_year_buffer=5)
    
    # Report year distribution
    print(f"\n  Year distribution in Zillow data:")
    year_counts = data.dataframe['year'].value_counts().sort_index()
    for year, count in year_counts.items():
        year_actual = [k for k, v in model.year_vocab.items() if v == year]
        year_str = year_actual[0] if year_actual else f"index_{year}"
        print(f"    {year_str}: {count} properties")
    
    print(f"  ✓ Prepared {len(data.dataframe)} properties for prediction")
    
    # Process data through model
    print(f"\n[4/5] Processing data through model...")
    
    model.processor(data, scale_mode="transform")
    
    print(f"  ✓ Data processed and scaled")
    
    # Generate predictions
    print(f"\n[5/5] Generating predictions...")
    
    model.add_predictions_to_data(return_uncertainty=True)
    
    print(f"  ✓ Predictions generated")
    
    # Analysis
    print("\n" + "="*80)
    print("PREDICTION RESULTS")
    print("="*80)
    
    results_df = model.dataframe[[
        'address', 'city', 'home_type', 'sqft', 'beds', 'baths',
        'sale_price', 'predicted_price', 'price_error', 'pct_error',
        'prediction_std_price', 'price_lower_95', 'price_upper_95',
        'days_on_zillow', 'zestimate'
    ]].copy()
    
    # Calculate zestimate error if available
    results_df['has_zestimate'] = results_df['zestimate'].notna()
    results_df['zestimate_error'] = results_df['zestimate'] - results_df['sale_price']
    results_df['zestimate_pct_error'] = 100 * (results_df['zestimate_error'] / results_df['sale_price'])
    
    # Summary statistics
    print(f"\nOverall Statistics:")
    print(f"  Mean Absolute Error: ${results_df['price_error'].abs().mean():,.0f}")
    print(f"  Mean Absolute % Error: {results_df['pct_error'].abs().mean():.2f}%")
    print(f"  Median Absolute % Error: {results_df['pct_error'].abs().median():.2f}%")
    print(f"  RMSE: ${np.sqrt((results_df['price_error']**2).mean()):,.0f}")
    
    # Zestimate comparison
    with_zestimate = results_df[results_df['has_zestimate']]
    if len(with_zestimate) > 0:
        print(f"\nZestimate Comparison ({len(with_zestimate)} properties with Zestimate):")
        print(f"  Zestimate MAE: ${with_zestimate['zestimate_error'].abs().mean():,.0f}")
        print(f"  Zestimate MAPE: {with_zestimate['zestimate_pct_error'].abs().mean():.2f}%")
        print(f"  {model_version} Model MAE: ${with_zestimate['price_error'].abs().mean():,.0f}")
        print(f"  {model_version} Model MAPE: {with_zestimate['pct_error'].abs().mean():.2f}%")
        
        # Which is better?
        model_better = (with_zestimate['pct_error'].abs() < with_zestimate['zestimate_pct_error'].abs()).sum()
        print(f"  {model_version} Model more accurate: {model_better}/{len(with_zestimate)} ({model_better/len(with_zestimate)*100:.1f}%)")
    
    # Prediction vs listing price comparison
    print(f"\nPrediction vs Listing Price:")
    print(f"  Avg Listing Price: ${results_df['sale_price'].mean():,.0f}")
    print(f"  Avg Predicted Price: ${results_df['predicted_price'].mean():,.0f}")
    print(f"  Difference: ${(results_df['predicted_price'].mean() - results_df['sale_price'].mean()):,.0f}")
    
    # Overpriced vs underpriced
    overpriced = results_df['pct_error'] < -5  # Model predicts >5% lower
    underpriced = results_df['pct_error'] > 5  # Model predicts >5% higher
    fair = (~overpriced) & (~underpriced)
    
    print(f"\nMarket Assessment (±5% threshold):")
    print(f"  Potentially Overpriced: {overpriced.sum()} ({overpriced.sum()/len(results_df)*100:.1f}%)")
    print(f"  Fairly Priced: {fair.sum()} ({fair.sum()/len(results_df)*100:.1f}%)")
    print(f"  Potentially Underpriced: {underpriced.sum()} ({underpriced.sum()/len(results_df)*100:.1f}%)")
    
    # Top 10 most overpriced (listing > prediction)
    print(f"\nTop 10 Most Overpriced (Listing > Model Prediction):")
    overpriced_sorted = results_df.sort_values('pct_error', ascending=True).head(10)
    for idx, row in overpriced_sorted.iterrows():
        zest_str = f"Zest: ${row['zestimate']:>10,.0f}" if pd.notna(row['zestimate']) else "Zest:        N/A"
        print(f"  {row['address'][:50]:50s} | List: ${row['sale_price']:>10,.0f} | Pred: ${row['predicted_price']:>10,.0f} | {zest_str} | Diff: {row['pct_error']:>6.1f}%")
    
    # Top 10 most underpriced (listing < prediction)
    print(f"\nTop 10 Most Underpriced (Listing < Model Prediction):")
    underpriced_sorted = results_df.sort_values('pct_error', ascending=False).head(10)
    for idx, row in underpriced_sorted.iterrows():
        zest_str = f"Zest: ${row['zestimate']:>10,.0f}" if pd.notna(row['zestimate']) else "Zest:        N/A"
        print(f"  {row['address'][:50]:50s} | List: ${row['sale_price']:>10,.0f} | Pred: ${row['predicted_price']:>10,.0f} | {zest_str} | Diff: {row['pct_error']:>6.1f}%")
    
    # Save results
    output_file = 'data/zillow_with_predictions.csv'
    model.dataframe.to_csv(output_file, index=False)
    
    print(f"\n{'='*80}")
    print(f"✓ Saved predictions to {output_file}")
    print(f"{'='*80}")
    
    # Create summary report
    summary_file = 'outputs/zillow_prediction_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("ZILLOW PRICE PREDICTIONS - SUMMARY REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Model: {model_version} ({model_dir})\n")
        f.write(f"Properties Analyzed: {len(results_df)}\n")
        f.write(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("OVERALL STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Mean Absolute Error: ${results_df['price_error'].abs().mean():,.0f}\n")
        f.write(f"Mean Absolute % Error: {results_df['pct_error'].abs().mean():.2f}%\n")
        f.write(f"Median Absolute % Error: {results_df['pct_error'].abs().median():.2f}%\n")
        f.write(f"RMSE: ${np.sqrt((results_df['price_error']**2).mean()):,.0f}\n\n")
        
        f.write("MARKET ASSESSMENT (±5% threshold)\n")
        f.write("-"*80 + "\n")
        f.write(f"Potentially Overpriced: {overpriced.sum()} ({overpriced.sum()/len(results_df)*100:.1f}%)\n")
        f.write(f"Fairly Priced: {fair.sum()} ({fair.sum()/len(results_df)*100:.1f}%)\n")
        f.write(f"Potentially Underpriced: {underpriced.sum()} ({underpriced.sum()/len(results_df)*100:.1f}%)\n\n")
        
        f.write("TOP 10 MOST OVERPRICED\n")
        f.write("-"*80 + "\n")
        for idx, row in overpriced_sorted.iterrows():
            f.write(f"{row['address']}\n")
            f.write(f"  Listing: ${row['sale_price']:,.0f} | Predicted: ${row['predicted_price']:,.0f} | Difference: {row['pct_error']:.1f}%\n")
            f.write(f"  {row['sqft']:.0f} sqft, {row['beds']:.0f} bed, {row['baths']:.1f} bath\n\n")
        
        f.write("\nTOP 10 MOST UNDERPRICED\n")
        f.write("-"*80 + "\n")
        for idx, row in underpriced_sorted.iterrows():
            f.write(f"{row['address']}\n")
            f.write(f"  Listing: ${row['sale_price']:,.0f} | Predicted: ${row['predicted_price']:,.0f} | Difference: {row['pct_error']:.1f}%\n")
            f.write(f"  {row['sqft']:.0f} sqft, {row['beds']:.0f} bed, {row['baths']:.1f} bath\n\n")
    
    print(f"✓ Saved summary report to {summary_file}")
    
    return model.dataframe


if __name__ == "__main__":
    df = load_and_predict()
