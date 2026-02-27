"""
Market Indicators Module
Fetches and manages economic indicators for real estate price modeling
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from typing import Optional, Dict
import json
import os


class MarketIndicatorFetcher:
    """Fetches and caches market indicators from various sources"""
    
    def __init__(self, cache_dir='data/market_indicators'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_file = os.path.join(cache_dir, 'indicators_cache.json')
        self.data = self._load_cache()
    
    def _load_cache(self) -> Dict:
        """Load cached indicator data"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Warning: Cache file corrupted ({e}), creating new cache")
                # Remove corrupted cache file
                try:
                    os.remove(self.cache_file)
                except:
                    pass
                return {}
        return {}
    
    def _save_cache(self):
        """Save indicator data to cache"""
        try:
            # Write to temporary file first
            temp_file = self.cache_file + '.tmp'
            with open(temp_file, 'w') as f:
                json.dump(self.data, f, indent=2)
            
            # Only replace original if write was successful
            if os.path.exists(temp_file):
                os.replace(temp_file, self.cache_file)
        except Exception as e:
            print(f"Warning: Failed to save cache ({e})")
    
    def fetch_mortgage_rates(self, start_date='2020-01-01', end_date=None) -> pd.DataFrame:
        """
        Fetch 30-year mortgage rates from FRED API
        Series: MORTGAGE30US (weekly data)
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        cache_key = f'mortgage_rates_{start_date}_{end_date}'
        
        if cache_key in self.data:
            print(f"Loading mortgage rates from cache")
            df = pd.DataFrame(self.data[cache_key])
            df['date'] = pd.to_datetime(df['date'])
            return df
        
        try:
            # FRED API endpoint (no API key needed for public data)
            url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US&cosd={start_date}&coed={end_date}'
            df = pd.read_csv(url)
            df.columns = ['date', 'mortgage_rate']
            df['date'] = pd.to_datetime(df['date'])
            df['mortgage_rate'] = pd.to_numeric(df['mortgage_rate'], errors='coerce')
            df = df.dropna()
            
            # Cache the data
            self.data[cache_key] = df.to_dict('records')
            self._save_cache()
            
            print(f"Fetched {len(df)} mortgage rate records from FRED")
            return df
        except Exception as e:
            print(f"Error fetching mortgage rates: {e}")
            return self._create_fallback_mortgage_rates(start_date, end_date)
    
    def _create_fallback_mortgage_rates(self, start_date, end_date) -> pd.DataFrame:
        """Create synthetic mortgage rate data if API fails"""
        print("Using fallback synthetic mortgage rates")
        dates = pd.date_range(start=start_date, end=end_date, freq='W')
        # Approximate historical rates with some variation
        base_rates = {
            2020: 3.1, 2021: 3.0, 2022: 5.3, 2023: 6.8, 2024: 6.9, 2025: 6.7, 2026: 6.5
        }
        rates = []
        for date in dates:
            base = base_rates.get(date.year, 6.5)
            noise = np.random.normal(0, 0.2)
            rates.append(base + noise)
        
        return pd.DataFrame({'date': dates, 'mortgage_rate': rates})
    
    def fetch_unemployment_rate(self, start_date='2020-01-01', end_date=None) -> pd.DataFrame:
        """
        Fetch unemployment rate from FRED
        Series: UNRATE (monthly data)
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        cache_key = f'unemployment_{start_date}_{end_date}'
        
        if cache_key in self.data:
            print(f"Loading unemployment data from cache")
            df = pd.DataFrame(self.data[cache_key])
            df['date'] = pd.to_datetime(df['date'])
            return df
        
        try:
            url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id=UNRATE&cosd={start_date}&coed={end_date}'
            df = pd.read_csv(url)
            df.columns = ['date', 'unemployment_rate']
            df['date'] = pd.to_datetime(df['date'])
            df['unemployment_rate'] = pd.to_numeric(df['unemployment_rate'], errors='coerce')
            df = df.dropna()
            
            self.data[cache_key] = df.to_dict('records')
            self._save_cache()
            
            print(f"Fetched {len(df)} unemployment records from FRED")
            return df
        except Exception as e:
            print(f"Error fetching unemployment: {e}")
            return self._create_fallback_unemployment(start_date, end_date)
    
    def _create_fallback_unemployment(self, start_date, end_date) -> pd.DataFrame:
        """Create synthetic unemployment data if API fails"""
        print("Using fallback synthetic unemployment rates")
        dates = pd.date_range(start=start_date, end=end_date, freq='MS')
        base_rates = {
            2020: 8.1, 2021: 5.4, 2022: 3.6, 2023: 3.6, 2024: 4.0, 2025: 4.2, 2026: 4.1
        }
        rates = [base_rates.get(date.year, 4.0) + np.random.normal(0, 0.3) for date in dates]
        return pd.DataFrame({'date': dates, 'unemployment_rate': rates})
    
    def merge_indicators_to_sales(self, sales_df: pd.DataFrame, 
                                   date_column='sale_date') -> pd.DataFrame:
        """
        Merge all market indicators to sales dataframe based on date
        Uses forward-fill for weekly/monthly data
        """
        df = sales_df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        
        # Fetch indicators
        start_date = df[date_column].min().strftime('%Y-%m-%d')
        end_date = df[date_column].max().strftime('%Y-%m-%d')
        
        mortgage_df = self.fetch_mortgage_rates(start_date, end_date)
        unemployment_df = self.fetch_unemployment_rate(start_date, end_date)
        
        # Merge with forward fill
        df = df.sort_values(date_column)
        df = pd.merge_asof(df, mortgage_df, left_on=date_column, right_on='date', direction='backward')
        df = pd.merge_asof(df, unemployment_df, left_on=date_column, right_on='date', direction='backward')
        
        # Fill any remaining NaNs
        df['mortgage_rate'] = df['mortgage_rate'].fillna(df['mortgage_rate'].mean())
        df['unemployment_rate'] = df['unemployment_rate'].fillna(df['unemployment_rate'].mean())
        
        return df
    
    def add_local_inventory(self, sales_df: pd.DataFrame, 
                           inventory_file: Optional[str] = None) -> pd.DataFrame:
        """
        Add local market inventory data if available
        Expected format: CSV with columns [date, inventory_count, months_supply]
        """
        if inventory_file and os.path.exists(inventory_file):
            inventory_df = pd.read_csv(inventory_file)
            inventory_df['date'] = pd.to_datetime(inventory_df['date'])
            sales_df = pd.merge_asof(
                sales_df.sort_values('sale_date'),
                inventory_df,
                left_on='sale_date',
                right_on='date',
                direction='backward'
            )
            print(f"Added local inventory data from {inventory_file}")
        else:
            # Create synthetic inventory features based on sales volume
            print("No inventory file provided, creating synthetic inventory indicators")
            sales_df = sales_df.sort_values('sale_date')
            sales_df['rolling_sales_volume'] = sales_df.groupby('community')['sale_price'].transform(
                lambda x: x.rolling(window=30, min_periods=1).count()
            )
            # Normalize by community
            sales_df['inventory_proxy'] = sales_df.groupby('community')['rolling_sales_volume'].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-6)
            )
        
        return sales_df


class TimeFeatureEngineer:
    """Creates continuous and cyclical time features"""
    
    @staticmethod
    def add_continuous_time_features(df: pd.DataFrame, 
                                     date_column='sale_date',
                                     reference_date=None) -> pd.DataFrame:
        """Add continuous time features for trend modeling"""
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        
        if reference_date is None:
            reference_date = df[date_column].min()
        else:
            reference_date = pd.to_datetime(reference_date)
        
        # Days since reference (normalized to years)
        df['time_trend'] = (df[date_column] - reference_date).dt.days / 365.25
        
        # Cyclical features for seasonality
        df['day_of_year'] = df[date_column].dt.dayofyear
        df['sin_day'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        df['cos_day'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
        
        # Month cyclical
        df['sin_month'] = np.sin(2 * np.pi * df[date_column].dt.month / 12)
        df['cos_month'] = np.cos(2 * np.pi * df[date_column].dt.month / 12)
        
        # Quarter
        df['quarter'] = df[date_column].dt.quarter
        
        return df
    
    @staticmethod
    def add_market_momentum_features(df: pd.DataFrame,
                                     date_column='sale_date',
                                     price_column='sale_price') -> pd.DataFrame:
        """Add features capturing market momentum and trends"""
        df = df.copy()
        df = df.sort_values(date_column)
        
        # Price momentum by community (3-month rolling average)
        df['price_ma_3m'] = df.groupby('community')[price_column].transform(
            lambda x: x.rolling(window=90, min_periods=10).mean()
        )
        
        # Price volatility (3-month rolling std)
        df['price_volatility_3m'] = df.groupby('community')[price_column].transform(
            lambda x: x.rolling(window=90, min_periods=10).std()
        )
        
        # Year-over-year price change
        df['yoy_price_change'] = df.groupby('community')[price_column].transform(
            lambda x: x.pct_change(periods=365)
        )
        
        # Fill NaNs with 0 for early records
        df['price_ma_3m'] = df['price_ma_3m'].fillna(df[price_column])
        df['price_volatility_3m'] = df['price_volatility_3m'].fillna(0)
        df['yoy_price_change'] = df['yoy_price_change'].fillna(0)
        
        return df
