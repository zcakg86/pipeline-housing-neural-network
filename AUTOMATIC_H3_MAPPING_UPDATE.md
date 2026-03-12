# Automatic H3 L9 Mapping Update

## Changes Made

### 1. New Module: `src/pricemodel/h3_neighbor_mapper.py`

Created a new module that handles automatic computation of H3 L9 neighbor mappings:

**Key Functions:**
- `compute_h3_l9_neighbor_mappings()` - Computes and caches mappings
- `ensure_h3_l9_mappings()` - Checks cache first, computes if needed

**Features:**
- Automatically checks if mapping files exist
- Computes mappings from dataframe if not found
- Caches results to avoid recomputation
- Supports force recompute option
- Progress reporting during computation

**Generated Files:**
- `data/h3_l9_to_community.json` - Direct hex → community mapping
- `data/h3_l9_neighbor_communities.json` - Hex → 7 neighbors mapping
- `data/community_vocab_l9.json` - Vocabulary metadata

### 2. Updated: `src/pricemodel/embedding_model_v2.py`

**Import Added:**
```python
from h3_neighbor_mapper import ensure_h3_l9_mappings
```

**Updated `_prepare_data()` Method:**
- Now automatically computes H3 L9 mappings if files don't exist
- Checks for existing files first (uses cache)
- Falls back gracefully if computation fails
- No longer requires manual precomputation step

**Behavior:**
```python
# Before: Required manual precomputation
# python3 precompute_h3_community_neighbors.py

# After: Automatic on first run
data._prepare_data(df)  # Computes mappings if needed
```

### 3. Updated: `main_train_v4_h3l9.py`

**New Data Loading:**
- Loads `data/sales_2020_25.csv` (main dataset)
- Loads `data/rentcast_recent_house_sales.csv` (RentCast data)
- Combines both datasets automatically
- Falls back to main dataset if RentCast not found

**H3 L9 Indexing:**
- Automatically generates H3 L9 indices from lat/lng
- Checks if h3_09 column exists first
- Only generates for missing values
- Reports progress and statistics

**New Function:**
```python
def load_and_prepare_data():
    """Load and combine sales data with RentCast data, apply H3 L9 indexing"""
    # Loads both datasets
    # Applies H3 L9 indexing
    # Returns combined dataframe
```

### 4. Updated: `test_h3_l9_integration.py`

**Updated to Match Training Script:**
- Loads combined dataset (main + RentCast)
- Applies H3 L9 indexing automatically
- Tests automatic mapping computation
- More robust error handling

## Workflow Comparison

### Before (Manual)

```bash
# Step 1: Manually run precomputation
python3 precompute_h3_community_neighbors.py

# Step 2: Train model
python3 main_train_v4_h3l9.py
```

### After (Automatic)

```bash
# Single step - everything automatic
python3 main_train_v4_h3l9.py
```

The script now:
1. Loads and combines datasets
2. Applies H3 L9 indexing
3. Computes neighbor mappings (if needed)
4. Trains the model

## Benefits

### 1. Simplified Workflow
- No manual precomputation step required
- Single command to train model
- Automatic data preparation

### 2. Intelligent Caching
- Checks for existing mappings first
- Only computes when necessary
- Saves computation time on subsequent runs

### 3. Robust Error Handling
- Graceful fallback if computation fails
- Clear error messages
- Continues with single community index if needed

### 4. Combined Dataset
- Automatically includes RentCast data
- More training data = better model
- Seamless integration

### 5. Automatic H3 Indexing
- Generates H3 L9 indices on the fly
- No need to pre-process data
- Works with any dataset with lat/lng

## File Structure

```
project/
├── src/
│   └── pricemodel/
│       ├── embedding_model_v2.py      (updated - auto-compute)
│       ├── h3_neighbor_mapper.py      (new - mapping logic)
│       ├── h3_community_embedding.py  (existing)
│       └── model_manager_v2.py        (existing)
├── data/
│   ├── sales_2020_25.csv              (main dataset)
│   ├── rentcast_recent_house_sales.csv (RentCast data)
│   ├── community_map.json             (required - L7 communities)
│   ├── h3_l9_to_community.json        (auto-generated)
│   ├── h3_l9_neighbor_communities.json (auto-generated)
│   └── community_vocab_l9.json        (auto-generated)
├── main_train_v4_h3l9.py              (updated - combined data)
└── test_h3_l9_integration.py          (updated - combined data)
```

## Usage

### Train V4 Model (One Command)

```bash
python3 main_train_v4_h3l9.py
```

**What Happens:**
1. Loads `data/sales_2020_25.csv`
2. Loads `data/rentcast_recent_house_sales.csv` (if exists)
3. Combines datasets
4. Generates H3 L9 indices (if needed)
5. Checks for neighbor mappings
6. Computes mappings if not found (first run only)
7. Trains model with neighborhood pooling

**First Run Output:**
```
Loading and combining datasets...
  - Main dataset: 84,497 records
  - RentCast dataset: 485 records
  ✓ Combined dataset: 84,982 records

Applying H3 Level 9 indexing...
  ✓ Generated H3 L9 indices for 84,982 properties

Preparing dataset with H3 L9 neighborhood mapping...
  H3 L9 neighbor mappings not found, computing automatically...
    Found 11,997 unique H3 L9 hexes
    Loaded 595 H3 L7 -> community mappings
    Mapped 11,997/11,997 L9 hexes
    Unique communities: 231
    Computing neighbor mappings for 11,997 hexes...
    ✓ Computed 11,997 neighbor mappings
    ✓ Saved mappings to data/
  ✓ H3 L9 neighbor mappings computed and cached
```

**Subsequent Runs:**
```
Loading and combining datasets...
  ✓ Combined dataset: 84,982 records

Applying H3 Level 9 indexing...
  ✓ H3 L9 indices already present: 84,982 properties

Preparing dataset with H3 L9 neighborhood mapping...
  H3 L9 neighbor mappings already exist, loading from cache...
  ✓ Loaded 11,997 L9 hex mappings from cache
```

### Test Integration

```bash
python3 test_h3_l9_integration.py
```

Tests with 1000 samples, including automatic mapping computation.

### Force Recompute Mappings

If you need to recompute mappings (e.g., after updating community detection):

```python
from pricemodel.h3_neighbor_mapper import compute_h3_l9_neighbor_mappings

# Force recompute
compute_h3_l9_neighbor_mappings(
    df,
    force_recompute=True
)
```

## Requirements

### Required Files
- `data/sales_2020_25.csv` - Main sales dataset
- `data/community_map.json` - H3 L7 → community mapping

### Optional Files
- `data/rentcast_recent_house_sales.csv` - Additional training data

### Auto-Generated Files (First Run)
- `data/h3_l9_to_community.json`
- `data/h3_l9_neighbor_communities.json`
- `data/community_vocab_l9.json`

## Backward Compatibility

The changes maintain full backward compatibility:

1. **Old precomputation script still works:**
   ```bash
   python3 precompute_h3_community_neighbors.py
   ```

2. **Existing cached files are used:**
   - If mappings exist, they're loaded from cache
   - No recomputation unless forced

3. **Fallback to single community:**
   - If computation fails, uses single community index
   - Model still trains, just without neighborhood pooling

## Performance

### First Run (Computing Mappings)
- ~11,997 hexes: ~5-10 seconds
- Progress reported every 2000 hexes
- Files cached for future use

### Subsequent Runs (Using Cache)
- Loading from cache: <1 second
- No computation overhead

## Troubleshooting

### "community_map.json not found"
**Cause:** H3 L7 community mapping doesn't exist
**Solution:** Run community detection first:
```bash
python3 main_community.py
```

### "Could not compute H3 L9 mappings"
**Cause:** Missing required columns or files
**Solution:** Check that data has lat/lng columns and community_map.json exists

### Mappings seem outdated
**Cause:** Community detection was rerun but mappings not updated
**Solution:** Delete old mapping files and rerun:
```bash
rm data/h3_l9*.json data/community_vocab_l9.json
python3 main_train_v4_h3l9.py
```

## Summary

The integration is now fully automatic:
- ✓ No manual precomputation required
- ✓ Automatic H3 L9 indexing
- ✓ Intelligent caching
- ✓ Combined dataset support
- ✓ Robust error handling
- ✓ Backward compatible

Just run `python3 main_train_v4_h3l9.py` and everything happens automatically!
