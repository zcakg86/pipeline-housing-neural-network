"""
Prediction Script for New Listings
Fetches listings from API and generates price predictions with confidence intervals
"""
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import json
import requests
from typing import Optional, Dict, List

sys.path.insert(0, os.getcwd() + '/src/pricemodel')

from market_indicators import MarketIndicatorFetcher, TimeFeatureEngineer
from embedding_model_v2 import dataset
from retraining_pipeline import RetrainingPipeline


class ListingPredictor:
    """
    Handles prediction for new listings from API
    """
    
    def __init__(self, api_endpoint: Optional[str] = None):
        self.api_endpoint = api_endpoint
        self.pipeline = RetrainingPipeline()
        self.model = None
        self.market_fetcher = MarketIndicatorFetcher()
        self.time_engineer = TimeFeatureEngineer()
        
    def load_model(self):
        """Load the active model"""
        if self.model is None:
            self.model = self.pipeline.get_active_model()
        return self.model
    
    def fetch_listings_from_api(self, api_endpoint: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch current listings from API
        
        Expected API response format:
        [
            {
                "listing_id": "123",
                "lat": 47.6062,
                "lng": -122.3321,
                "sqft": 2000,
                "sqft_lot": 5000,
                "beds": 3,
                "year_built": 1990,
                "h3_07": "8728d5cd3ffffff",
                "list_date": "2025-01-15",
                ...
            }
        ]
        """
        if api_endpoint is None:
            api_endpoint = self.api_endpoint
        
        if api_endpoint is None:
            raise ValueError("API endpoint not provided")
        
        print(f"Fetching listings from {api_endpoint}...")
        
        try:
            response = requests.get(api_endpoint, timeout=30)
            response.raise_for_status()
            listings = response.json()
            
            df = pd.DataFrame(listings)
            print(f"Fetched {len(df)} listings from API")
            
            return df
        
        except Exception as e:
            print(f"Error fetching from API: {e}")
            print("Using sample data for demonstration...")
            return self._create_sample_listings()
    
    def _create_sample_listings(self) -> pd.DataFrame:
        """Create sample listings for testing"""
        sample_data = {
            'listing_id': ['L001', 'L002', 'L003', 'L004', 'L005'],
            'lat': [47.6062, 47.6205, 47.5569, 47.6740, 47.4801],
            'lng': [-122.3321, -122.3493, -122.3789, -122.3215, -122.2079],
            'sqft': [1800, 2200, 1500, 2800, 1900],
            'sqft_lot': [5000, 6500, 4200, 7200, 8500],
            'beds': [3, 4, 2, 4, 3],
            'year_built': [1995, 2005, 1980, 2010, 1998],
            'h3_07': ['8728d5cd3ffffff', '8728d540bffffff', '8728d5576ffffff', 
                     '8728d540dffffff', '8728d5532ffffff'],
            'list_date': ['2025-02-26'] * 5
        }
        
        return pd.DataFrame(sample_data)
    
    def load_listings_from_file(self, file_path: str) -> pd.DataFrame:
        """Load listings from CSV file"""
        print(f"Loading listings from {file_path}...")
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} listings")
        return df
    
    def prepare_listings_for_prediction(self, listings_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare listing data for model prediction
        Adds all required features: market indicators, time features, etc.
        """
        df = listings_df.copy()
        
        # Ensure date column
        if 'list_date' in df.columns:
            df['sale_date'] = pd.to_datetime(df['list_date'])
        else:
            df['sale_date'] = pd.Timestamp.now()
        
        # Add required columns with defaults if missing
        required_cols = {
            'sale_nbr': 1.0,  # Dummy value for listings
            'sale_price': 0.0  # Unknown (what we're predicting)
        }
        
        for col, default_val in required_cols.items():
            if col not in df.columns:
                df[col] = default_val
        
        # Add market indicators
        print("Adding market indicators...")
        df = self.market_fetcher.merge_indicators_to_sales(df, date_column='sale_date')
        df = self.market_fetcher.add_local_inventory(df)
        
        # Add time features
        print("Adding time features...")
        # Use model's reference date if available
        reference_date = self.model.reference_date if self.model else None
        df = self.time_engineer.add_continuous_time_features(df, date_column='sale_date', 
                                                             reference_date=reference_date)
        
        # For momentum features, we can't calculate from listing data alone
        # Use defaults or fetch from recent sales
        df['price_ma_3m'] = 0.0
        df['price_volatility_3m'] = 0.0
        df['yoy_price_change'] = 0.0
        
        # Add community mapping
        with open('data/community_map.json', 'r') as f:
            community_map = json.load(f)
        
        df['community'] = df['h3_07'].map(community_map).astype(str)
        
        # Handle unknown communities
        df['community'] = df['community'].fillna('unknown')
        
        return df
    
    def predict(self, listings_df: pd.DataFrame, return_details=True) -> pd.DataFrame:
        """
        Generate predictions for listings
        """
        # Load model if not already loaded
        self.load_model()
        
        # Prepare data
        df = self.prepare_listings_for_prediction(listings_df)
        
        # Create dataset object
        data = dataset()
        data.reference_date = self.model.reference_date
        data.community_vocab = self.model.community_vocab
        data.year_vocab = self.model.year_vocab
        data.week_vocab = self.model.week_vocab
        
        # Prepare data (without fitting new scalers)
        df['log_price'] = 0.0  # Dummy target
        data.dataframe = df
        
        # Add derived features
        df['price_per_sqft'] = 0.0
        df['month'] = df['sale_date'].dt.month
        df['year'] = df['sale_date'].dt.isocalendar().year
        df['week'] = df['sale_date'].dt.isocalendar().week
        df['day_of_year'] = df['sale_date'].dt.dayofyear
        df['quarter'] = df['sale_date'].dt.quarter
        
        # Map communities
        df['community_index'] = df['community'].map(self.model.community_vocab).fillna(
            self.model.community_vocab.get('unknown', 0)
        )
        
        data.dataframe = df
        data.length = len(df)
        data.n_communities = self.model.n_communities
        data.year_length = self.model.year_length
        data.week_length = self.model.week_length
        
        # Process with existing scalers
        self.model.processor(data, scale_mode="transform")
        
        # Generate predictions
        print("Generating predictions...")
        self.model.add_predictions_to_data(return_uncertainty=True)
        
        # Prepare output
        result_df = self.model.dataframe.copy()
        
        if return_details:
            # Include all relevant columns
            output_cols = [
                'listing_id', 'lat', 'lng', 'sqft', 'sqft_lot', 'beds',
                'community', 'sale_date',
                'predicted_price', 'prediction_std_price',
                'price_lower_95', 'price_upper_95',
                'mortgage_rate', 'unemployment_rate'
            ]
            
            # Only include columns that exist
            output_cols = [col for col in output_cols if col in result_df.columns]
            result_df = result_df[output_cols]
        else:
            # Minimal output
            result_df = result_df[['listing_id', 'predicted_price', 'price_lower_95', 'price_upper_95']]
        
        print(f"\nPredictions generated for {len(result_df)} listings")
        print(f"Average predicted price: ${result_df['predicted_price'].mean():,.0f}")
        print(f"Price range: ${result_df['predicted_price'].min():,.0f} - ${result_df['predicted_price'].max():,.0f}")
        
        return result_df
    
    def predict_and_save(self, listings_source, output_path='data/listing_predictions.csv'):
        """
        Generate predictions and save to file
        
        Args:
            listings_source: Either API endpoint (str), file path (str), or DataFrame
            output_path: Where to save predictions
        """
        # Load listings
        if isinstance(listings_source, pd.DataFrame):
            listings_df = listings_source
        elif isinstance(listings_source, str):
            if listings_source.startswith('http'):
                listings_df = self.fetch_listings_from_api(listings_source)
            else:
                listings_df = self.load_listings_from_file(listings_source)
        else:
            raise ValueError("listings_source must be DataFrame, file path, or API endpoint")
        
        # Generate predictions
        predictions_df = self.predict(listings_df, return_details=True)
        
        # Save
        predictions_df.to_csv(output_path, index=False)
        print(f"\nPredictions saved to {output_path}")
        
        return predictions_df


def main():
    """
    Example usage
    """
    print("=" * 80)
    print("Real Estate Listing Price Predictor")
    print("=" * 80)
    
    predictor = ListingPredictor()
    
    # Option 1: Fetch from API
    # predictions = predictor.predict_and_save('https://your-api.com/listings')
    
    # Option 2: Load from file
    # predictions = predictor.predict_and_save('data/new_listings.csv')
    
    # Option 3: Use sample data
    print("\nUsing sample listings for demonstration...")
    sample_listings = predictor._create_sample_listings()
    predictions = predictor.predict(sample_listings)
    
    print("\n" + "=" * 80)
    print("PREDICTION RESULTS")
    print("=" * 80)
    print(predictions.to_string(index=False))
    
    # Save predictions
    predictions.to_csv('data/listing_predictions.csv', index=False)
    print(f"\nPredictions saved to data/listing_predictions.csv")
    
    return predictor, predictions


if __name__ == "__main__":
    predictor, predictions = main()
