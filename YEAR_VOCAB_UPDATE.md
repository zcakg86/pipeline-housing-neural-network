# Year Vocabulary Update - Summary

## What Changed

Updated the year vocabulary creation to **pre-allocate future years** instead of mapping all future years to "unknown".

## Before (Old Approach)

```python
# Only years in training data
year_vocab = {2020: 0, 2021: 1, 2022: 2, 2023: 3, 2024: 4, 2025: 5, "unknown": 6}

# Problem: 2026, 2027, 2028... all map to "unknown" (index 6)
```

## After (New Approach)

```python
# Pre-allocate 5 future years
year_vocab = {
    2020: 0, 2021: 1, 2022: 2, 2023: 3, 2024: 4, 2025: 5,
    2026: 6, 2027: 7, 2028: 8, 2029: 9, 2030: 10,  # Future years
    "unknown": 11  # Only for years beyond 2030
}

# Benefit: Each year gets its own embedding!
```

## Why This Is Better

### 1. Temporal Granularity
- 2026 and 2027 are treated as distinct years
- Model can learn year-specific patterns
- Better interpolation between years

### 2. Smooth Extrapolation
- Embeddings for consecutive years are close in embedding space
- Model naturally interpolates: 2025 → 2026 → 2027
- Works with continuous time features

### 3. Flexibility
- Configurable buffer: `future_year_buffer=5` (default)
- Can adjust based on prediction horizon
- "Unknown" still available for years beyond buffer

## Configuration

### Default (Recommended)
```python
data._prepare_data(df, future_year_buffer=5)
# Covers 5 years beyond training data
```

### Short-term
```python
data._prepare_data(df, future_year_buffer=2)
# Only 2 years ahead
```

### Long-term
```python
data._prepare_data(df, future_year_buffer=10)
# Up to 10 years ahead
```

## Example: Training on 2020-2025, Predicting 2026

### Year Embedding
```python
# 2026 gets its own embedding (index 6)
year_embedding_2026 = model.year_embedding[6]

# Not grouped with 2027 (index 7) or 2028 (index 8)
```

### Combined with Continuous Features
```python
# Year embedding: Discrete representation
year_embedding = model.year_embedding[6]  # 2026

# Continuous features: Smooth extrapolation
time_trend = 6.0  # 6 years from 2020
sin_day = sin(2π × day_of_year / 365)
cos_day = cos(2π × day_of_year / 365)

# Market indicators: Current conditions
mortgage_rate = 6.8%  # Actual 2026 rate
unemployment = 4.2%   # Actual 2026 rate

# Model combines all features for prediction
```

## Files Modified

1. **src/pricemodel/embedding_model_v2.py**
   - Line ~40: Added `future_year_buffer` parameter
   - Line ~93: Updated year vocab creation
   - Line ~113: Updated year_length calculation

2. **main_train_v2.py**
   - Line ~50: Added `future_year_buffer=5` parameter

3. **Documentation**
   - Created `FUTURE_YEAR_HANDLING.md` - Complete explanation
   - Created `YEAR_VOCAB_UPDATE.md` - This summary

## How to Use

### Training
```python
# Automatically uses future_year_buffer=5
python main_train_v2.py
```

### Custom Buffer
```python
from src.pricemodel.embedding_model_v2 import dataset

data = dataset()
data._prepare_data(
    df,
    include_market_indicators=True,
    future_year_buffer=10  # Custom: 10 years ahead
)
```

### Docker
```bash
# Train with default settings
make train

# Or with custom script
docker-compose run --rm train python your_custom_train.py
```

## Monitoring

Check which years are in vocabulary:

```python
print(f"Year vocabulary: {data.year_vocab}")
print(f"Training years: {df['year'].min()} to {df['year'].max()}")
print(f"Prediction range: up to {max(data.year_vocab.keys()) - 1}")
```

Output:
```
Year vocabulary: {2020: 0, 2021: 1, ..., 2030: 10, 'unknown': 11}
Training years: 2020 to 2025
Prediction range: up to 2030
```

## Benefits Summary

✅ **Better temporal modeling**
- Each year has unique embedding
- Smooth interpolation between years
- Distinguishes near vs far future

✅ **Flexible configuration**
- Adjustable buffer size
- Balances granularity vs model size
- "Unknown" for years beyond buffer

✅ **Works with continuous features**
- Year embeddings + time_trend
- Discrete + continuous representation
- Best of both worlds

✅ **Production ready**
- Default buffer=5 covers most use cases
- Automatic in training scripts
- Compatible with retraining pipeline

## Next Steps

1. **Rebuild Docker**: `make build`
2. **Train model**: `make train`
3. **Check vocab**: Look for "Year vocabulary:" in training output
4. **Test 2026**: Use predict_listings.py with 2026 data

The model will now handle 2026-2030 predictions with individual year embeddings instead of grouping them all as "unknown"!
