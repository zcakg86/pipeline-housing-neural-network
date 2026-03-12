"""
Transform Zillow JSON data to dataset ready for model predictions
"""
import json
import pandas as pd
import numpy as np
import h3
from datetime import datetime

def load_zillow_data(file_path='data/zillow_seattle_listings.json'):
    """Load Zillow JSON data"""
    print("="*80)
    print("ZILLOW DATA PREPARATION FOR PREDICTIONS")
    print("="*80)
    
    print(f"\nLoading data from {file_path}...")
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    properties = data.get('properties', [])
    print(f"Found {len(properties)} properties")
    
    # Show total results available
    if 'searchInformation' in data and 'totalResults' in data['searchInformation']:
        print(f"Total results available: {data['searchInformation']['totalResults']}")
    
    return properties


def transform_zillow_to_model_format(properties):
    """
    Transform Zillow properties to model input format
    
    Required fields for model:
        - sale_date: Use current date (for prediction)
        - sale_price: Use listing price
        - sale_nbr: Unique identifier
        - lat: Latitude
        - lng: Longitude
        - sqft: Square footage
        - sqft_lot: Lot size
        - year_built: Year built
        - year_reno: Year renovated (optional)
        - beds: Bedrooms
        - baths: Bathrooms
        - community: Will be mapped from H3
        - h3_07: H3 index level 7
    """
    
    print("\nTransforming Zillow data to model format...")
    
    records = []
    current_date = datetime.now().strftime('%Y-%m-%d')
    
    for idx, prop in enumerate(properties):
        try:
            # Extract address info
            address = prop.get('address', {})
            
            record = {
                # Use current date as "sale_date" for prediction
                'sale_date': current_date,
                
                # Use listing price as "sale_price" (what we want to predict)
                'sale_price': prop.get('price'),
                
                # Unique identifier
                'sale_nbr': idx + 1,  # Sequential number
                
                # Location
                'lat': prop.get('latitude'),
                'lng': prop.get('longitude'),
                
                # Property characteristics
                'sqft': prop.get('area') or prop.get('livingArea'),
                'sqft_lot': prop.get('lotAreaValue'),
                'year_built': prop.get('yearBuilt'),
                'year_reno': None,  # Not typically in Zillow data
                'beds': prop.get('beds'),
                'baths': prop.get('baths'),
                
                # Additional fields for reference
                'zillow_id': prop.get('id'),
                'address': prop.get('addressRaw') or f"{address.get('street', '')}, {address.get('city', '')}, {address.get('state', '')} {address.get('zipcode', '')}",
                'city': address.get('city'),
                'state': address.get('state'),
                'zipcode': address.get('zipcode'),
                'home_type': prop.get('homeType'),
                'status': prop.get('status'),
                'days_on_zillow': prop.get('daysOnZillow'),
                'url': prop.get('url'),
                'zestimate': prop.get('zestimate'),  # Zillow's automated valuation
            }
            
            # Skip if missing critical fields
            if not all([record['lat'], record['lng'], record['sqft']]):
                continue
            
            # Generate H3 index at resolution 7
            if record['lat'] and record['lng']:
                try:
                    record['h3_07'] = h3.latlng_to_cell(record['lat'], record['lng'], 7)
                except:
                    record['h3_07'] = None
            else:
                record['h3_07'] = None
            
            # Convert numeric fields
            for field in ['sale_price', 'sqft', 'sqft_lot', 'year_built', 'beds', 'baths']:
                if record[field] is not None:
                    try:
                        record[field] = float(record[field])
                    except:
                        record[field] = None
            
            records.append(record)
            
        except Exception as e:
            print(f"  Error processing property {idx}: {e}")
            continue
    
    df = pd.DataFrame(records)
    
    print(f"Transformed {len(df)} properties")
    
    return df


def add_community_mapping(df, community_map_file='data/community_map.json'):
    """Add community IDs using existing community mapping"""
    
    print(f"\nAdding community mapping from {community_map_file}...")
    
    try:
        with open(community_map_file, 'r') as f:
            community_map = json.load(f)
        
        df['community'] = df['h3_07'].map(community_map).fillna(-1).astype(int)
        
        mapped = df['community'] != -1
        print(f"  Mapped {mapped.sum()} properties to communities")
        print(f"  Unmapped: {(~mapped).sum()} properties")
        print(f"  Unique communities: {df[mapped]['community'].nunique()}")
        
    except FileNotFoundError:
        print(f"  Warning: Community map not found at {community_map_file}")
        print(f"  Setting all communities to -1")
        df['community'] = -1
    
    return df


def add_market_indicators(df):
    """Add market indicators and time features"""
    
    print("\nAdding market indicators and time features...")
    
    # Import here to avoid circular dependencies
    import sys
    import os
    sys.path.insert(0, os.getcwd() + '/src/pricemodel')
    
    from market_indicators import MarketIndicatorFetcher, TimeFeatureEngineer
    
    # Add market indicators
    market_fetcher = MarketIndicatorFetcher()
    df = market_fetcher.merge_indicators_to_sales(df, date_column='sale_date')
    
    # Add local inventory (optional)
    inventory_file = 'data/local_inventory.csv'
    df = market_fetcher.add_local_inventory(df, inventory_file if os.path.exists(inventory_file) else None)
    
    print(f"  Added market indicators")
    
    # Add time features
    time_engineer = TimeFeatureEngineer()
    df = time_engineer.add_continuous_time_features(df, date_column='sale_date')
    df = time_engineer.add_market_momentum_features(df, date_column='sale_date', price_column='sale_price')
    
    print(f"  Added time features")
    
    return df


def validate_for_prediction(df):
    """Validate dataset is ready for prediction"""
    
    print("\n" + "="*80)
    print("VALIDATION FOR PREDICTION")
    print("="*80)
    
    required_fields = ['sale_date', 'sale_price', 'sale_nbr', 'lat', 'lng', 
                      'sqft', 'beds', 'baths', 'community', 'h3_07']
    
    print(f"\nRequired fields check:")
    for field in required_fields:
        if field in df.columns:
            missing = df[field].isna().sum()
            print(f"  ✓ {field}: {len(df) - missing}/{len(df)} present ({missing} missing)")
        else:
            print(f"  ✗ {field}: MISSING COLUMN")
    
    # Fill missing values
    print(f"\nFilling missing values...")
    
    if 'beds' in df.columns and df['beds'].isna().sum() > 0:
        median_beds = df['beds'].median()
        df['beds'] = df['beds'].fillna(median_beds)
        print(f"  Filled {df['beds'].isna().sum()} missing beds with median: {median_beds}")
    
    if 'baths' in df.columns and df['baths'].isna().sum() > 0:
        median_baths = df['baths'].median()
        df['baths'] = df['baths'].fillna(median_baths)
        print(f"  Filled {df['baths'].isna().sum()} missing baths with median: {median_baths}")
    
    if 'sqft_lot' in df.columns and df['sqft_lot'].isna().sum() > 0:
        median_lot = df['sqft_lot'].median()
        df['sqft_lot'] = df['sqft_lot'].fillna(median_lot)
        print(f"  Filled {df['sqft_lot'].isna().sum()} missing sqft_lot with median: {median_lot}")
    
    if 'year_built' in df.columns and df['year_built'].isna().sum() > 0:
        median_year = df['year_built'].median()
        df['year_built'] = df['year_built'].fillna(median_year)
        print(f"  Filled {df['year_built'].isna().sum()} missing year_built with median: {median_year}")
    
    # Summary
    print(f"\nDataset Summary:")
    print(f"  Total properties: {len(df)}")
    print(f"  Price range: ${df['sale_price'].min():,.0f} to ${df['sale_price'].max():,.0f}")
    print(f"  Median price: ${df['sale_price'].median():,.0f}")
    print(f"  Sqft range: {df['sqft'].min():.0f} to {df['sqft'].max():.0f}")
    print(f"  Beds range: {df['beds'].min():.0f} to {df['beds'].max():.0f}")
    print(f"  Baths range: {df['baths'].min():.0f} to {df['baths'].max():.0f}")
    
    return df


def main():
    """Main function to prepare Zillow data for predictions"""
    
    # Load Zillow data
    properties = load_zillow_data('data/zillow_seattle_listings.json')
    
    if not properties:
        print("\n❌ No properties found in zillow.json")
        return
    
    # Transform to model format
    df = transform_zillow_to_model_format(properties)
    
    if len(df) == 0:
        print("\n❌ No valid properties after transformation")
        return
    
    # Add community mapping
    df = add_community_mapping(df)
    
    # Add market indicators and time features
    df = add_market_indicators(df)
    
    # Validate and fill missing values
    df = validate_for_prediction(df)
    
    # Save prepared dataset
    output_file = 'data/zillow_prepared_for_prediction.csv'
    df.to_csv(output_file, index=False)
    
    print(f"\n{'='*80}")
    print(f"✓ Saved prepared dataset to {output_file}")
    print(f"{'='*80}")
    
    print("\nSample properties:")
    display_cols = ['address', 'sale_price', 'sqft', 'beds', 'baths', 'city', 'community', 'days_on_zillow', 'zestimate']
    available_cols = [col for col in display_cols if col in df.columns]
    print(df[available_cols].head(10))
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\n1. Load your trained V3 model")
    print("2. Load this prepared dataset")
    print("3. Run model.add_predictions_to_data() to get price predictions")
    print("\nExample code:")
    print("  from model_manager_v2 import modelmanager")
    print("  model = modelmanager()")
    print("  model.load_model('outputs/models/[your_model_timestamp]')")
    print("  # Load and process zillow data with model's processor")
    print("  # Then run: model.add_predictions_to_data()")
    
    return df


if __name__ == "__main__":
    df = main()
