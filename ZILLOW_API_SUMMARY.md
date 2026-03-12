# Zillow API Integration Summary

## Overview
Successfully integrated Zillow API to fetch current Seattle listings and compare model predictions against both listing prices and Zestimate values.

## Files Created

### 1. Data Fetching
- **`fetch_zillow_listings.py`**: Fetches all Seattle listings from Zillow API
  - Uses HasData API with provided credentials
  - Fetches 41 properties from 1,933 total available
  - Saves raw response to `data/zillow_seattle_listings.json`

### 2. Data Preparation
- **`prepare_zillow_for_prediction.py`** (updated): Transforms Zillow data for model
  - Now includes: `url`, `days_on_zillow`, `zestimate`
  - Maps properties to H3 communities
  - Adds market indicators and time features
  - Outputs: `data/zillow_prepared_for_prediction.csv`

### 3. Predictions
- **`predict_zillow_prices.py`** (updated): Generates predictions with Zestimate comparison
  - Loads V3 model
  - Processes 39 properties
  - Compares V3 predictions vs Zestimate vs Listing prices
  - Outputs: `data/zillow_with_predictions.csv`

### 4. Visualizations
- **`visualize_zillow_predictions.py`**: Creates analysis charts
  - 6-panel analysis (scatter, error distribution, by price/size, uncertainty, attention)
  - Geographic maps (error distribution, price distribution)
  - Outputs: `outputs/zillow_predictions_analysis.png`, `outputs/zillow_predictions_geographic.png`

## Key Results

### Model Performance Comparison (26 properties with Zestimate)

| Metric | Zestimate | V3 Model |
|--------|-----------|----------|
| Mean Absolute Error | $38,991 | $531,050 |
| Mean Absolute % Error | 2.61% | 39.67% |
| More Accurate | 26/26 (100%) | 0/26 (0%) |

### Overall Statistics (39 properties)

- **V3 Model MAPE**: 41.28%
- **Median APE**: 40.95%
- **RMSE**: $857,923
- **Average Listing**: $1,142,874
- **Average Prediction**: $627,937
- **Difference**: -$514,936 (model undervalues by 45%)

### Market Assessment (±5% threshold)

- **Potentially Overpriced**: 30 properties (76.9%)
- **Fairly Priced**: 2 properties (5.1%)
- **Potentially Underpriced**: 7 properties (17.9%)

### Model Attention Weights

- **Property Features**: 31.1% (most important)
- **Community**: 19.7%
- **Time Features**: 16.5%
- **Year**: 10.3%
- **Market**: 10.3%
- **Week**: 5.8%

## Key Findings

### 1. Systematic Undervaluation
The V3 model consistently predicts lower prices than both listing prices and Zestimate. This suggests:
- Model trained on historical data (2020-2025) doesn't capture current market appreciation
- Missing features that capture luxury/premium characteristics
- Need for market adjustment factor or retraining on recent data

### 2. Zestimate Accuracy
Zestimate is remarkably accurate (2.61% MAPE), demonstrating:
- Access to comprehensive data sources
- Sophisticated feature engineering
- Regular model updates with current market data
- Likely includes features like: recent comps, neighborhood trends, property condition, etc.

### 3. Price Range Sensitivity
- Model performs worse on high-end properties (>$1M)
- Better accuracy on smaller condos and lower-priced properties
- Suggests need for price-stratified models or non-linear adjustments

### 4. Geographic Patterns
- Certain neighborhoods (Magnolia, Queen Anne) show higher listing premiums
- Model may not capture neighborhood prestige/desirability factors
- Community embeddings capture some but not all location value

## Recommendations

### Short-term Improvements
1. **Market Adjustment Factor**: Apply a 1.45x multiplier to predictions to align with current market
2. **Zestimate Integration**: Use Zestimate as an additional feature when available
3. **Price Stratification**: Train separate models for different price ranges

### Long-term Improvements
1. **Retrain on Recent Data**: Include 2025-2026 sales data
2. **Additional Features**: 
   - Property condition/quality scores
   - Recent neighborhood sales trends
   - School ratings
   - Walkability scores
   - Days on market patterns
3. **Ensemble Approach**: Combine V3 model with Zestimate and other signals
4. **Regular Updates**: Implement monthly retraining pipeline

## API Details

### Endpoint
```
GET https://api.hasdata.com/scrape/zillow/listing
```

### Parameters
- `keyword`: "Seattle, WA"
- `type`: "forSale"
- `sort`: "newest"

### Headers
- `x-api-key`: a33bc008-3e10-4785-a44f-3924e2391e0d
- `Content-Type`: application/json

### Response
- Returns 41 properties per request
- Total available: 1,933 properties
- Includes: price, beds, baths, sqft, location, zestimate, daysOnZillow, etc.

## Files Generated

### Data Files
- `data/zillow_seattle_listings.json` - Raw API response (41 properties)
- `data/zillow_prepared_for_prediction.csv` - Processed data ready for model (39 properties)
- `data/zillow_with_predictions.csv` - Full predictions with all fields

### Output Files
- `outputs/zillow_prediction_summary.txt` - Text summary report
- `outputs/zillow_predictions_analysis.png` - 6-panel visualization
- `outputs/zillow_predictions_geographic.png` - Geographic maps

## Usage

### Fetch New Listings
```bash
python3 fetch_zillow_listings.py
```

### Prepare Data
```bash
python3 prepare_zillow_for_prediction.py
```

### Generate Predictions
```bash
python3 predict_zillow_prices.py
```

### Create Visualizations
```bash
python3 visualize_zillow_predictions.py
```

## Next Steps

1. Fetch more properties (API returns 41 of 1,933 available)
2. Implement pagination to get all Seattle listings
3. Fine-tune model on recent RentCast data (619 properties from 2025-2026)
4. Compare fine-tuned model performance vs current V3
5. Explore ensemble methods combining V3 + Zestimate
