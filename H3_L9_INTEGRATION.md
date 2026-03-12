# H3 Level 9 Community Embedding Integration Guide

## Overview
The model now uses H3 Level 9 hexagons with neighborhood-aware community embeddings.
Each location is represented by its community and its 6 neighbors' communities.

## Files Created

### 1. Precomputation Script
- **`precompute_h3_community_neighbors.py`**: Generates neighbor mappings
- Run once before training: `python3 precompute_h3_community_neighbors.py`

### 2. Generated Data Files
- **`data/h3_l9_to_community.json`**: Maps each H3 L9 hex to its community
- **`data/h3_l9_neighbor_communities.json`**: Maps each hex to [center + 6 neighbors]
- **`data/community_vocab_l9.json`**: Community vocabulary and metadata

### 3. Model Module
- **`src/pricemodel/h3_community_embedding.py`**: New embedding layer

## Integration Steps

### Step 1: Update Data Preparation
In `embedding_model_v2.py`, modify `_prepare_data()`:

```python
# Load neighbor mappings
with open('data/h3_l9_neighbor_communities.json', 'r') as f:
    h3_neighbor_map = json.load(f)

# Add neighbor community indices to dataframe
df['community_neighbors'] = df['h3_09'].map(h3_neighbor_map)
```

### Step 2: Update Dataset Class
Modify the dataset to return neighbor communities:

```python
# In dataset tensors, replace single community index with 7-element tensor
community_tensor = torch.tensor([row['community_neighbors'] for row in df], dtype=torch.long)
# Shape: (N, 7) instead of (N,)
```

### Step 3: Update Model Architecture
Replace the community embedding in `price_predictor`:

```python
from h3_community_embedding import H3CommunityEmbedding

# In __init__:
self.community_embedding = H3CommunityEmbedding(
    num_communities=n_communities,
    embedding_dim=embedding_dim,
    pooling_strategy='mean'  # or 'center_weighted' or 'learnable'
)

# In forward():
# Input shape: (batch, 7)
community_embed = self.community_embedding(community)
# Output shape: (batch, embedding_dim)
```

## Pooling Strategies

1. **'mean'**: Simple average (recommended for start)
2. **'center_weighted'**: 50% center, 50% neighbors
3. **'learnable'**: Model learns optimal weights

## Benefits

- Captures spatial context from neighboring areas
- More robust to boundary effects
- Better generalization to nearby unseen locations
- Smoother predictions across space

## Statistics

- H3 L9 hexes: 11,997
- Communities: 231
- Vocabulary size: 232 (including UNKNOWN)
- Each hex: 7 community indices (center + 6 neighbors)
