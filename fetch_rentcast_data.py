"""
Fetch recent property sales from RentCast API
Query sales based on location in the last 6 months
Format data to match existing model training dataset
"""
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time
import h3

# RentCast API configuration
RENTCAST_API_KEY = "c07c11bb6ddf4284be66c55cdb0aba92"
RENTCAST_BASE_URL = "https://api.rentcast.io/v1"


def get_recent_sales(api_key, lat = 47.60, lng = -122.33,radius = 10, months_back=6, limit=500):
    """
    Query RentCast API for recent property sales
    
    Parameters:
        api_key: RentCast API key
        months_back: Number of months to look back (default: 6)
        limit: Max records per request (default: 500, max allowed)
    
    Returns:
        List of property records
    """
    
    # Calculate date range
    end_date = datetime.now()
    days = months_back * 30
    start_date = end_date - timedelta(days=days)
    
    print(f"Fetching sales from {start_date.date()} to {end_date.date()}")
    print(f"Location: ({lat}, {lng}), Radius: {radius} miles")
    
    # API endpoint
    url = f"{RENTCAST_BASE_URL}/properties"
    
    headers = {
        "X-Api-Key": api_key,
        "accept": "application/json"
    }
    # Query parameters
    params = {
        "latitude": lat,
        "longitude": lng,
        "radius": radius,   
        "limit": limit,
        "saleDateRange": f'0:{days}',
    }
    
    all_properties = []
    1
    # Single API call (no pagination)
    print(f"\nFetching up to {limit} records...")
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        # Debug: print response structure
        print(f"  API Response keys: {data.keys() if isinstance(data, dict) else type(data)}")
        
        # Handle different response structures
        if isinstance(data, list):
            # API returns a list directly
            properties = data
        elif isinstance(data, dict):
            # API returns a dict with properties key
            properties = data.get("properties", data.get("results", []))
        else:
            print(f"  Unexpected response type: {type(data)}")
            print(f"  Response: {data}")
            return []
        
        if not properties:
            print("No properties found")
            return []
        
        print(f"  Found {len(properties)} properties")
        all_properties.extend(properties)
        
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        print(f"Response status: {response.status_code}")
        print(f"Response text: {response.text[:500]}")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        print(f"Response text: {response.text[:500]}")
    
    print(f"\nTotal properties fetched: {len(all_properties)}")
    return all_properties


def transform_to_model_format(properties):
    """
    Transform RentCast property data to match existing model dataset format
    
    Expected columns in existing dataset:
        - sale_date: Date of sale
        - sale_price: Sale price
        - sale_nbr: Unique sale identifier
        - lat: Latitude
        - lng: Longitude
        - sqft: Square footage
        - sqft_lot: Lot size in square feet
        - year_built: Year property was built
        - year_reno: Year of last renovation (if any)
        - beds: Number of bedrooms
        - baths: Number of bathrooms
        - community: Community identifier (will be derived from H3)
        - h3_07: H3 index at resolution 7
    """
    
    print("\nTransforming data to model format...")
    
    records = []
    
    for prop in properties:
        try:
            # Extract required fields
            record = {
                # Sale information
                'sale_date': pd.to_datetime(prop.get('lastSaleDate')).strftime('%Y-%m-%d') if prop.get('lastSaleDate') else None,
                'sale_price': prop.get('lastSalePrice'),
                'sale_nbr': prop.get('id') or prop.get('formattedAddress', '').replace(' ', '_'),
                
                # Location
                'lat': prop.get('latitude'),
                'lng': prop.get('longitude'),
                
                # Property characteristics
                'sqft': prop.get('squareFootage') or prop.get('livingSquareFootage'),
                'sqft_lot': prop.get('lotSize'),
                'year_built': prop.get('yearBuilt'),
                'year_reno': prop.get('lastRenovationDate'),  # May need parsing
                'beds': prop.get('bedrooms'),
                'baths': prop.get('bathrooms'),
                
                # Additional useful fields
                'address': prop.get('formattedAddress'),
                'city': prop.get('city'),
                'zipcode': prop.get('zipCode'),
                'property_type': prop.get('propertyType')
            }
            
            # Skip if missing critical fields
            if not all([record['sale_date'], record['sale_price'], 
                       record['lat'], record['lng'], record['sqft']]):
                continue
            
            # Generate H3 index at resolution 7
            if record['lat'] and record['lng']:
                try:
                    record['h3_07'] = h3.latlng_to_cell(record['lat'], record['lng'], 7)
                except:
                    record['h3_07'] = None
            else:
                record['h3_07'] = None
            
            # Handle year_reno - extract year if it's a date
            if record['year_reno']:
                try:
                    if isinstance(record['year_reno'], str):
                        record['year_reno'] = pd.to_datetime(record['year_reno']).year
                except:
                    record['year_reno'] = None
            
            # Convert numeric fields
            for field in ['sale_price', 'sqft', 'sqft_lot', 'year_built', 'beds', 'baths']:
                if record[field] is not None:
                    try:
                        record[field] = float(record[field])
                    except:
                        record[field] = None
            
            records.append(record)
            
        except Exception as e:
            print(f"Error processing property: {e}")
            continue
    
    df = pd.DataFrame(records)
    
    print(f"Transformed {len(df)} records")
    print(f"\nData summary:")
    print(f"  Date range: {df['sale_date'].min()} to {df['sale_date'].max()}")
    print(f"  Price range: ${df['sale_price'].min():,.0f} to ${df['sale_price'].max():,.0f}")
    print(f"  Median price: ${df['sale_price'].median():,.0f}")
    print(f"  Properties with H3: {df['h3_07'].notna().sum()}")
    
    return df


def add_community_mapping(df, community_map_file='data/community_map.json'):
    """
    Add community IDs using existing community mapping
    
    Parameters:
        df: DataFrame with h3_07 column
        community_map_file: Path to community mapping JSON file
    
    Returns:
        DataFrame with community column added
    """
    
    print(f"\nAdding community mapping from {community_map_file}...")
    
    try:
        with open(community_map_file, 'r') as f:
            community_map = json.load(f)
        
        df['community'] = df['h3_07'].map(community_map).fillna(-1).astype(int)
        
        mapped = df['community'] != -1
        print(f"  Mapped {mapped.sum()} records to communities")
        print(f"  Unmapped: {(~mapped).sum()} records")
        print(f"  Unique communities: {df[mapped]['community'].nunique()}")
        
    except FileNotFoundError:
        print(f"  Warning: Community map not found at {community_map_file}")
        print(f"  Setting all communities to -1 (will need to be mapped)")
        df['community'] = -1
    
    return df


def validate_dataset(df, reference_file='data/sales_202025.csv'):
    """
    Validate the new dataset against the reference dataset structure
    
    Parameters:
        df: New dataset
        reference_file: Path to reference dataset
    """
    
    print("\n" + "="*80)
    print("DATASET VALIDATION")
    print("="*80)
    
    try:
        df_ref = pd.read_csv(reference_file, nrows=100)
        ref_columns = set(df_ref.columns)
        new_columns = set(df.columns)
        
        print(f"\nReference dataset columns: {len(ref_columns)}")
        print(f"New dataset columns: {len(new_columns)}")
        
        # Check for missing columns
        missing = ref_columns - new_columns
        if missing:
            print(f"\n⚠️  Missing columns: {missing}")
        else:
            print(f"\n✓ All reference columns present")
        
        # Check for extra columns
        extra = new_columns - ref_columns
        if extra:
            print(f"\n✓ Extra columns (will be kept): {extra}")
        
    except FileNotFoundError:
        print(f"\n⚠️  Reference file not found: {reference_file}")
        print(f"Cannot validate against reference dataset")
    
    # Data quality checks
    print(f"\nData Quality:")
    print(f"  Total records: {len(df)}")
    print(f"  Missing sale_price: {df['sale_price'].isna().sum()}")
    print(f"  Missing sale_date: {df['sale_date'].isna().sum()}")
    print(f"  Missing lat/lng: {df[['lat', 'lng']].isna().any(axis=1).sum()}")
    print(f"  Missing sqft: {df['sqft'].isna().sum()}")
    print(f"  Missing beds: {df['beds'].isna().sum()}")
    print(f"  Missing baths: {df['baths'].isna().sum()}")
    
    # Value ranges
    print(f"\nValue Ranges:")
    print(f"  Sale price: ${df['sale_price'].min():,.0f} to ${df['sale_price'].max():,.0f}")
    print(f"  Sqft: {df['sqft'].min():.0f} to {df['sqft'].max():.0f}")
    print(f"  Beds: {df['beds'].min():.0f} to {df['beds'].max():.0f}")
    print(f"  Baths: {df['baths'].min():.0f} to {df['baths'].max():.0f}")
    print(f"  Year built: {df['year_built'].min():.0f} to {df['year_built'].max():.0f}")


def main():
    """
    Main function to fetch and process RentCast data
    """
    
    print("="*80)
    print("RENTCAST DATA FETCHER")
    print("Fetch recent property sales for model training")
    print("="*80)
    
    # Check if API key is set
    if RENTCAST_API_KEY == "YOUR_API_KEY_HERE":
        print("\n⚠️  ERROR: Please set your RentCast API key in the script")
        print("Get your API key from: https://app.rentcast.io/app/api-settings")
        print("\nUpdate the RENTCAST_API_KEY variable at the top of this script")
        return
    
    # Fetch data from RentCast API
    # Seattle coordinates: 47.6062, -122.3321
    # Adjust lat/lng and radius to cover King County
    properties = get_recent_sales(
        api_key=RENTCAST_API_KEY,
        lat=47.60,           # Seattle latitude
        lng=-122.33,         # Seattle longitude  
        radius=10,           # 10 mile radius to cover seattle area.
        months_back=6,
        limit=500
    )
    
    if not properties:
        print("\n❌ No properties fetched. Check your API key and parameters.")
        return
    
    # Transform to model format
    df = transform_to_model_format(properties)
    
    if len(df) == 0:
        print("\n❌ No valid records after transformation")
        return
    
    # Add community mapping
    df = add_community_mapping(df)
    
    # Validate dataset
    validate_dataset(df)
    
    # Save to CSV
    output_file = 'data/rentcast_recent_sales.csv'
    df.to_csv(output_file, index=False)
    print(f"\n{'='*80}")
    print(f"✓ Saved {len(df)} records to {output_file}")
    print(f"{'='*80}")
    
    # Print sample
    print("\nSample records:")
    print(df[['sale_date', 'sale_price', 'sqft', 'beds', 'baths', 'city', 'community']].head(10))
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\n1. Review the data quality and coverage")
    print("2. If communities are unmapped (-1), you may need to:")
    print("   - Update the community_map.json with new H3 cells")
    print("   - Or retrain with the new data to create new communities")
    print("3. Combine with existing training data if desired:")
    print("   df_old = pd.read_csv('data/sales_202025.csv')")
    print("   df_new = pd.read_csv('data/rentcast_recent_sales.csv')")
    print("   df_combined = pd.concat([df_old, df_new], ignore_index=True)")
    print("4. Train V3 model with the new data:")
    print("   python main_train_v3.py")


if __name__ == "__main__":
    main()
