# Future Year Handling - Design Document

## Problem

When predicting on future data (e.g., 2026) with a model trained on historical data (2020-2025), we need to decide how to handle year embeddings for unseen years.

## Two Approaches

### Approach 1: Map All Future Years to "Unknown" ❌

**How it works:**
```python
year_vocab = {2020: 0, 2021: 1, 2022: 2, 2023: 3, 2024: 4, 2025: 5, "unknown": 6}
# 2026, 2027, 2028... all map to index 6
```

**Problems:**
- All future years share the same embedding
- 2026 and 2030 are treated identically
- Loses temporal granularity
- Model can't distinguish between near-future and far-future

### Approach 2: Pre-allocate Future Years ✅ (Implemented)

**How it works:**
```python
# Training data: 2020-2025
# Pre-allocate 5 future years
year_vocab = {
    2020: 0, 2021: 1, 2022: 2, 2023: 3, 2024: 4, 2025: 5,
    2026: 6, 2027: 7, 2028: 8, 2029: 9, 2030: 10,  # Future years
    "unknown": 11  # For years beyond 2030
}
```

**Benefits:**
- Each year gets its own embedding
- Model can learn year-specific patterns
- Better temporal granularity
- Smooth interpolation between known and future years

## Implementation

### Configuration

```python
# In dataset._prepare_data()
data = dataset()
data._prepare_data(
    df, 
    include_market_indicators=True,
    future_year_buffer=5  # Pre-allocate 5 future years
)
```

### How It Works

1. **Vocabulary Creation** (embedding_model_v2.py, line ~93):
```python
min_year = int(df['year'].min())  # e.g., 2020
max_year = int(df['year'].max())  # e.g., 2025

# Create vocab from min_year to max_year + buffer
year_range = range(min_year, max_year + future_year_buffer + 1)
# Result: 2020, 2021, ..., 2025, 2026, 2027, 2028, 2029, 2030

year_vocab = {int(year): idx for idx, year in enumerate(year_range)}
year_vocab["unknown"] = len(year_vocab)  # For years > 2030
```

2. **Embedding Layer**:
```python
# Model has embeddings for all years in vocab
self.year_embedding = nn.Embedding(
    num_embeddings=len(year_vocab),  # e.g., 12 (2020-2030 + unknown)
    embedding_dim=16
)
```

3. **Training**:
- Only years 2020-2025 have training data
- Embeddings for 2026-2030 are initialized randomly
- They get updated through backpropagation via:
  - Continuous time features (time_trend)
  - Shared patterns with nearby years
  - Market indicators

4. **Prediction on 2026**:
```python
# 2026 is in vocab, gets its own embedding
year_index = year_vocab[2026]  # Returns 6
year_embedding = model.year_embedding(year_index)  # Unique embedding for 2026
```

## Why This Works

### 1. Continuous Time Features Provide Signal

Even though 2026 has no training data, the model learns from:

```python
# Continuous features that work for any year
time_trend = 6.0  # 6 years from reference (2020)
sin_day = sin(2π × day_of_year / 365)
cos_day = cos(2π × day_of_year / 365)
```

These features provide temporal context that helps the model extrapolate.

### 2. Embedding Interpolation

Neural networks naturally interpolate between embeddings:

```
2024 embedding ──→ [learned from data]
2025 embedding ──→ [learned from data]
2026 embedding ──→ [initialized randomly, but close to 2025 in embedding space]
2027 embedding ──→ [initialized randomly, but close to 2026 in embedding space]
```

During training, the model learns that consecutive years should have similar embeddings.

### 3. Market Indicators

Current market conditions (mortgage rates, unemployment) provide real-time context:

```python
# 2026 prediction uses actual 2026 market data
mortgage_rate_2026 = 6.8%  # Fetched from FRED API
unemployment_2026 = 4.2%   # Fetched from FRED API
```

## Configuration Options

### Default (Recommended)
```python
future_year_buffer=5  # Pre-allocate 5 years ahead
```

**Use when:**
- Making predictions 1-5 years ahead
- Want good temporal granularity
- Have stable market conditions

### Conservative
```python
future_year_buffer=2  # Only 2 years ahead
```

**Use when:**
- Only need short-term predictions
- Want smaller model size
- Market is highly volatile

### Aggressive
```python
future_year_buffer=10  # Pre-allocate 10 years ahead
```

**Use when:**
- Need long-term projections
- Have stable long-term trends
- Willing to accept more uncertainty

## Example Usage

### Training Script

```python
from src.pricemodel.embedding_model_v2 import dataset
from src.pricemodel.model_manager_v2 import modelmanager

# Load data (2020-2025)
df = pd.read_csv('data/sales_202025.csv')

# Prepare with 5-year buffer
data = dataset()
data._prepare_data(
    df,
    include_market_indicators=True,
    future_year_buffer=5  # Allows predictions through 2030
)

# Train model
model = modelmanager()
model.processor(data)
model.train_model(...)
```

### Prediction on 2026 Data

```python
# Load 2026 listings
listings_2026 = pd.read_csv('listings_2026.csv')

# Predict (2026 is in vocab, gets its own embedding)
predictions = predictor.predict(listings_2026)

# Each 2026 listing uses:
# - year_embedding[2026] (unique to 2026)
# - time_trend = 6.0 (continuous)
# - mortgage_rate_2026 (current)
# - unemployment_2026 (current)
```

## Comparison: 2026 vs 2030 Predictions

### With Future Buffer = 5

```python
# 2026 prediction
year_vocab[2026] = 6  # Has its own embedding
time_trend = 6.0
confidence = "medium"  # 1 year beyond training

# 2030 prediction  
year_vocab[2030] = 10  # Has its own embedding
time_trend = 10.0
confidence = "low"  # 5 years beyond training
```

### Without Future Buffer (Old Approach)

```python
# 2026 prediction
year_vocab["unknown"] = 6  # Shared with all future years
time_trend = 6.0
confidence = "low"  # Grouped with all future

# 2030 prediction
year_vocab["unknown"] = 6  # SAME as 2026!
time_trend = 10.0
confidence = "low"  # Can't distinguish from 2026
```

## Monitoring & Retraining

### When to Retrain

As you collect 2026 data, retrain to:
1. Update 2026 embedding with real data
2. Extend buffer to include 2031
3. Improve accuracy on recent years

```python
# Retrain with 2026 data
df_updated = pd.read_csv('sales_2020_2026.csv')  # Now includes 2026

data = dataset()
data._prepare_data(
    df_updated,
    future_year_buffer=5  # Now covers 2027-2031
)

model.train_model(...)
```

### Monitoring Predictions

Track prediction quality by year:

```python
# Check prediction confidence by year
predictions['year'] = predictions['list_date'].dt.year

by_year = predictions.groupby('year').agg({
    'prediction_std_price': 'mean',  # Uncertainty
    'confidence': lambda x: (x == 'high').mean()  # % high confidence
})

print(by_year)
#      prediction_std_price  confidence
# 2025        $35,000         0.85  (85% high confidence)
# 2026        $48,000         0.62  (62% high confidence)
# 2027        $65,000         0.41  (41% high confidence)
```

## Best Practices

### 1. Choose Buffer Based on Use Case

```python
# Short-term predictions (1-2 years)
future_year_buffer=2

# Medium-term predictions (3-5 years)
future_year_buffer=5  # Recommended default

# Long-term projections (5-10 years)
future_year_buffer=10
```

### 2. Retrain Regularly

```python
# Retrain when:
# - New year's data becomes available
# - Prediction quality degrades
# - Market conditions change significantly

# Automated retraining (scheduler.py)
pipeline.should_retrain(
    time_threshold_days=30  # Retrain monthly
)
```

### 3. Monitor Uncertainty

```python
# Flag predictions with high uncertainty
high_uncertainty = predictions[
    predictions['prediction_std_price'] > 100000
]

# These likely need:
# - More recent training data
# - Extended future buffer
# - Manual review
```

### 4. Use Continuous Features

The future year embeddings work because of continuous features:

```python
# Always include these for temporal extrapolation
- time_trend (continuous years)
- sin_day, cos_day (seasonality)
- sin_month, cos_month (monthly patterns)
- mortgage_rate (current market)
- unemployment_rate (economic conditions)
```

## Summary

✅ **Pre-allocating future years is better because:**
- Each year gets unique embedding
- Better temporal granularity
- Smooth interpolation
- Works with continuous features
- Distinguishes near vs far future

✅ **Configuration:**
```python
future_year_buffer=5  # Default, covers 5 years ahead
```

✅ **Works for 2026 predictions:**
- 2026 gets its own embedding (not grouped with 2027+)
- Continuous time features enable extrapolation
- Market indicators provide current context
- Uncertainty estimates show confidence

✅ **Retrain as new data arrives:**
- Update embeddings with real data
- Extend buffer for next years
- Maintain prediction quality

This approach gives you the best of both worlds: structured year embeddings AND the ability to extrapolate to future years!
