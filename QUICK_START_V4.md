# Quick Start: V4 Model with H3 L9 Neighborhood Embeddings

## TL;DR

The model now uses H3 Level 9 hexagons with neighborhood pooling. Each location is represented by 7 communities (center + 6 neighbors) instead of just 1.

**Everything is automatic** - just run one command!

## Train V4 Model (One Command)

```bash
python3 main_train_v4_h3l9.py
```

That's it! The script automatically:
- ✓ Loads and combines datasets (main + RentCast)
- ✓ Generates H3 L9 indices
- ✓ Computes neighbor mappings (first run only)
- ✓ Trains model with neighborhood pooling

## What You'll See

### First Run (Computing Mappings)

```
Training V4 Model with H3 L9 Neighborhood-Aware Embeddings
==================================================================

1. Loading and combining datasets...
   - Main dataset: 84,497 records
   - RentCast dataset: 485 records
   ✓ Combined dataset: 84,982 records

   Applying H3 Level 9 indexing...
   ✓ Generated H3 L9 indices for 84,982 properties

2. Preparing dataset with H3 L9 neighborhood mapping...
   H3 L9 neighbor mappings not found, computing automatically...
     Found 11,997 unique H3 L9 hexes
     Loaded 595 H3 L7 -> community mappings
     Mapped 11,997/11,997 L9 hexes
     Computing neighbor mappings for 11,997 hexes...
     ✓ Computed 11,997 neighbor mappings
     ✓ Saved mappings to data/
   ✓ H3 L9 neighbor mappings computed and cached

3. Training model...
   [Progress and results]
```

### Subsequent Runs (Using Cache)

```
1. Loading and combining datasets...
   ✓ Combined dataset: 84,982 records
   ✓ H3 L9 indices already present

2. Preparing dataset with H3 L9 neighborhood mapping...
   H3 L9 neighbor mappings already exist, loading from cache...
   ✓ Loaded 11,997 L9 hex mappings from cache

3. Training model...
   [Progress and results]
```

## Quick Test (30 seconds)

```bash
python3 test_h3_l9_integration.py
```

Tests with 1000 samples to verify everything works. Also tests automatic mapping computation.

## What Changed from V3

| Aspect | V3 | V4 |
|--------|----|----|
| H3 Level | 7 (~5 km²) | 9 (~0.1 km²) |
| Communities per location | 1 | 7 (center + neighbors) |
| Spatial context | None | Neighborhood pooling |
| Community tensor shape | (batch,) | (batch, 7) |

## Key Benefits

✓ Higher spatial resolution (50x smaller hexagons)
✓ Captures neighborhood patterns
✓ Smoother predictions across space
✓ Better boundary handling

## Pooling Strategies

Change in `main_train_v4_h3l9.py`:

```python
manager.train_model(
    ...
    pooling_strategy='mean'  # Options: 'mean', 'center_weighted', 'learnable'
)
```

- **mean**: Equal weight to all 7 (default, recommended)
- **center_weighted**: 50% center, 50% neighbors
- **learnable**: Model learns optimal weights

## Load a Trained V4 Model

```python
from pricemodel.model_manager_v2 import modelmanager

manager = modelmanager()
manager.load_model('outputs/models/20260311_123456')

# Automatically knows it uses neighborhood pooling
print(manager.use_neighborhood_pooling)  # True
print(manager.pooling_strategy)  # 'mean'
```

## Troubleshooting

### "community_map.json not found"
Run community detection first:
```bash
python3 main_community.py
```

### Want to recompute mappings?
Delete cached files and rerun:
```bash
rm data/h3_l9*.json data/community_vocab_l9.json
python3 main_train_v4_h3l9.py
```

### Model trains but uses single community
Check that:
1. `data/community_map.json` exists
2. Data has lat/lng columns
3. No errors during mapping computation

## Files You Need

**Required:**
- ✓ `data/sales_2020_25.csv` - Main sales dataset
- ✓ `data/community_map.json` - H3 L7 community mapping

**Optional:**
- `data/rentcast_recent_house_sales.csv` - Additional training data (auto-included if present)

**Auto-Generated (First Run):**
- `data/h3_l9_to_community.json` - Computed automatically
- `data/h3_l9_neighbor_communities.json` - Computed automatically
- `data/community_vocab_l9.json` - Computed automatically

## Compare V3 vs V4

After training V4:

```python
# Load both models
v3 = pd.read_csv('data/sales_2020_25_with_predictions_v3.csv')
v4 = pd.read_csv('data/sales_2020_25_with_predictions_v4.csv')

# Compare MAPE
print(f"V3 MAPE: {v3['pct_error'].abs().mean():.2f}%")
print(f"V4 MAPE: {v4['pct_error'].abs().mean():.2f}%")

# Compare attention weights
print(f"\nV3 Community Attention: {v3['cls_attn_community'].mean():.3f}")
print(f"V4 Community Attention: {v4['cls_attn_community'].mean():.3f}")
```

## Documentation

- **Full guide:** `H3_L9_MODEL_INTEGRATION_GUIDE.md`
- **Summary:** `V4_INTEGRATION_SUMMARY.md`
- **Original spec:** `H3_L9_INTEGRATION.md`

## That's It!

You're ready to train V4. Just run:

```bash
python3 main_train_v4_h3l9.py
```

The model will automatically use H3 L9 neighborhood pooling and save everything you need.
