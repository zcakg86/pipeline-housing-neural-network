# Community Index Out of Bounds - Fix Documentation

## Problem

You encountered an error:
```
File "/app/src/pricemodel/embedding_model_v2.py", line 176, in forward
    community_embeddings = self.community_embedding(community_indices)
```

This error occurs when community indices exceed the embedding table size, causing an IndexError.

## Root Causes

1. **Missing communities mapped to NaN**: When a community doesn't exist in the vocabulary, `map()` returns NaN instead of the "unknown" index
2. **No bounds checking**: The forward pass didn't validate that indices were within valid range
3. **Vocab handling**: The vocab_replace_tensor function didn't properly handle edge cases

## Fixes Applied

### 1. Fixed Community Index Mapping (embedding_model_v2.py, line ~107)

**Before:**
```python
df['community_index'] = df['community'].map(self.community_vocab)
```

**After:**
```python
# Map communities, filling unknown with the "unknown" index
df['community_index'] = df['community'].map(self.community_vocab).fillna(
    self.community_vocab["unknown"]
).astype(int)
```

**Why**: This ensures that any missing or unknown communities are mapped to the "unknown" token index instead of NaN.

### 2. Added Safety Clamping in Forward Pass (embedding_model_v2.py, line ~173)

**Before:**
```python
def forward(self, community_indices, year, week, property_features, 
            time_features, market_features, return_uncertainty=False):
    # --- Embed Categorical Inputs ---
    community_embeddings = self.community_embedding(community_indices)
    year_embeddings = self.year_embedding(year)
    week_embeddings = self.week_embedding(week)
```

**After:**
```python
def forward(self, community_indices, year, week, property_features, 
            time_features, market_features, return_uncertainty=False):
    # --- Safety: Clamp indices to valid range ---
    community_indices = torch.clamp(community_indices, 0, self.community_embedding.num_embeddings - 1)
    year = torch.clamp(year, 0, self.year_embedding.num_embeddings - 1)
    week = torch.clamp(week, 0, self.week_embedding.num_embeddings - 1)
    
    # --- Embed Categorical Inputs ---
    community_embeddings = self.community_embedding(community_indices)
    year_embeddings = self.year_embedding(year)
    week_embeddings = self.week_embedding(week)
```

**Why**: This provides a safety net by clamping any out-of-bounds indices to the valid range [0, num_embeddings-1].

### 3. Improved vocab_replace_tensor Function (embedding_model_v2.py, line ~343)

**Before:**
```python
def vocab_replace_tensor(tensor, vocab):
    """Replaces values in a tensor with their indices from vocabulary"""
    replaced = [vocab.get(value.item(), vocab['unknown']) for value in tensor]
    return torch.tensor(replaced, dtype=torch.int)
```

**After:**
```python
def vocab_replace_tensor(tensor, vocab):
    """Replaces values in a tensor with their indices from vocabulary"""
    # Get the unknown index (should be the last index)
    unknown_idx = vocab.get('unknown', len(vocab) - 1)
    
    replaced = []
    for value in tensor:
        val = value.item()
        # Try to get the vocab index, use unknown if not found
        if val in vocab:
            replaced.append(vocab[val])
        else:
            replaced.append(unknown_idx)
    
    return torch.tensor(replaced, dtype=torch.int)
```

**Why**: This handles edge cases where the value might not be in the vocabulary, and properly retrieves the unknown index.

## Testing

A test script `test_community_indices.py` was created to verify the fixes:

```bash
# Run in Docker
docker-compose run --rm train python test_community_indices.py
```

The test verifies:
- ✅ All community indices are within bounds [0, n_communities-1]
- ✅ No NaN indices exist
- ✅ Unknown communities are properly mapped
- ✅ Min/max indices are correct

## How to Verify the Fix

### Option 1: Run the test script
```bash
make shell
python test_community_indices.py
```

### Option 2: Train the model
```bash
make train
```

If the training completes without the IndexError, the fix is working.

### Option 3: Check indices manually
```python
import pandas as pd
import sys
sys.path.insert(0, 'src/pricemodel')
from embedding_model_v2 import dataset

# Load your data
df = pd.read_csv('data/sales_202025.csv')
# ... add community mapping ...

# Prepare dataset
data = dataset()
data = data._prepare_data(df)

# Check indices
print(f"Community vocab size: {data.n_communities}")
print(f"Max community index: {data.dataframe['community_index'].max()}")
print(f"Min community index: {data.dataframe['community_index'].min()}")
print(f"Any NaN: {data.dataframe['community_index'].isna().any()}")

# Should print:
# Max index < vocab size
# Min index >= 0
# No NaN
```

## Prevention

To prevent this issue in the future:

1. **Always use .fillna()** when mapping categorical variables:
   ```python
   df['category_index'] = df['category'].map(vocab).fillna(vocab['unknown']).astype(int)
   ```

2. **Add bounds checking** in model forward passes:
   ```python
   indices = torch.clamp(indices, 0, embedding.num_embeddings - 1)
   ```

3. **Validate data** before training:
   ```python
   assert df['community_index'].max() < n_communities
   assert df['community_index'].min() >= 0
   assert not df['community_index'].isna().any()
   ```

4. **Include "unknown" token** in all vocabularies:
   ```python
   vocab = create_vocab(df, 'column')
   vocab['unknown'] = len(vocab)  # Always add unknown token
   ```

## Summary

The fixes ensure that:
- ✅ All community indices are properly bounded
- ✅ Unknown/missing communities are handled gracefully
- ✅ The model won't crash with IndexError
- ✅ Predictions work on new data with unseen communities

The model is now robust to:
- Missing community values
- New communities not in training data
- Edge cases in vocabulary mapping

## Rebuild Docker Container

After these fixes, rebuild your Docker container:

```bash
make build-clean
make train
```

The training should now complete successfully without the IndexError.
