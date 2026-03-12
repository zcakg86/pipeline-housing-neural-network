# RentCast API Integration Guide

## Overview
This script fetches recent property sales data from the RentCast API and formats it to match your existing model training dataset.

## Setup

### 1. Get RentCast API Key
1. Sign up at https://www.rentcast.io/
2. Navigate to API Settings: https://app.rentcast.io/app/api-settings
3. Copy your API key

### 2. Configure the Script
Open `fetch_rentcast_data.py` and replace the API key:

```python
RENTCAST_API_KEY = "your_actual_api_key_here"
```

### 3. Install Dependencies (if needed)
```bash
pip3 install requests pandas h3 --user
```

## Usage

### Basic Usage
Fetch last 6 months of sales in King County, WA:
```bash
python3 fetch_rentcast_data.py
```

### Customize Parameters
Edit the `main()` function in the script:

```python
properties = get_recent_sales(
    api_key=RENTCAST_API_KEY,
    county="King",        # Change county
    state="WA",           # Change state
    months_back=6,        # Change time range
    limit=500             # Records per API call (max 500)
)
```

## Output

The script creates: `data/rentcast_recent_sales.csv`

### Output Columns
Matches your existing training data format:

**Required for Model:**
- `sale_date` - Date of sale
- `sale_price` - Sale price in USD
- `sale_nbr` - Unique sale identifier
- `lat` - Latitude
- `lng` - Longitude
- `sqft` - Living square footage
- `sqft_lot` - Lot size in square feet
- `year_built` - Year property was built
- `year_reno` - Year of last renovation
- `beds` - Number of bedrooms
- `baths` - Number of bathrooms
- `community` - Community ID (mapped from H3)
- `h3_07` - H3 index at resolution 7

**Additional Fields:**
- `address` - Full address
- `city` - City name
- `zipcode` - ZIP code
- `property_type` - Type of property
- `county` - County name
- `state` - State abbreviation

## RentCast API Field Mapping

| RentCast Field | Model Field | Notes |
|----------------|-------------|-------|
| `lastSaleDate` | `sale_date` | Converted to datetime |
| `lastSalePrice` | `sale_price` | In USD |
| `id` or `formattedAddress` | `sale_nbr` | Unique identifier |
| `latitude` | `lat` | Decimal degrees |
| `longitude` | `lng` | Decimal degrees |
| `squareFootage` or `livingSquareFootage` | `sqft` | Living area |
| `lotSize` | `sqft_lot` | Lot size |
| `yearBuilt` | `year_built` | Year constructed |
| `lastRenovationDate` | `year_reno` | Year of renovation |
| `bedrooms` | `beds` | Number of bedrooms |
| `bathrooms` | `baths` | Number of bathrooms |

## Community Mapping

The script uses your existing `data/community_map.json` to map H3 cells to community IDs.

**If communities are unmapped (-1):**
1. New H3 cells not in your existing map
2. Options:
   - Update `community_map.json` with new mappings
   - Retrain model to create new community clusters
   - Use the data as-is (model will assign to "unknown" community)

## Combining with Existing Data

```python
import pandas as pd

# Load existing training data
df_old = pd.read_csv('data/sales_202025.csv')

# Load new RentCast data
df_new = pd.read_csv('data/rentcast_recent_sales.csv')

# Combine
df_combined = pd.concat([df_old, df_new], ignore_index=True)

# Remove duplicates (if any)
df_combined = df_combined.drop_duplicates(subset=['sale_nbr'])

# Save
df_combined.to_csv('data/sales_combined.csv', index=False)

print(f"Combined dataset: {len(df_combined)} records")
```

## Training with New Data

### Option 1: Train on New Data Only
```bash
# Update main_train_v3.py to use new file
# Change: df = pd.read_csv('data/sales_202025.csv')
# To:     df = pd.read_csv('data/rentcast_recent_sales.csv')

python3 main_train_v3.py
```

### Option 2: Train on Combined Data
```bash
# First combine the datasets (see above)
# Then update main_train_v3.py to use combined file
python3 main_train_v3.py
```

## API Rate Limits

RentCast API limits:
- **Free tier**: 100 requests/month
- **Paid tiers**: Higher limits

The script includes:
- 0.5 second delay between requests
- Pagination support (500 records per request)
- Error handling for rate limits

## Troubleshooting

### No properties fetched
- Check API key is correct
- Verify county/state spelling
- Check date range (may be no sales in period)
- Review API quota/limits

### Missing fields
- Some properties may not have all fields
- Script skips records missing critical fields (price, location, sqft)
- Check data quality summary in output

### Community mapping issues
- Unmapped communities show as -1
- Update `community_map.json` or retrain model
- Model can handle unknown communities

## Example Output

```
================================================================================
RENTCAST DATA FETCHER
Fetch recent property sales for model training
================================================================================

Fetching sales from 2025-08-26 to 2026-02-26
County: King, State: WA

Fetching records 0 to 500...
  Found 500 properties, 342 within date range

Total properties fetched: 342

Transforming data to model format...
Transformed 338 records

Data summary:
  Date range: 2025-08-27 to 2026-02-25
  Price range: $425,000 to $2,850,000
  Median price: $785,000
  Properties with H3: 338

Adding community mapping from data/community_map.json...
  Mapped 312 records to communities
  Unmapped: 26 records
  Unique communities: 45

✓ Saved 338 records to data/rentcast_recent_sales.csv
```

## API Documentation

Full RentCast API documentation:
https://developers.rentcast.io/reference/property-data-schema

## Support

For issues with:
- **RentCast API**: https://www.rentcast.io/support
- **This script**: Check the validation output and error messages
