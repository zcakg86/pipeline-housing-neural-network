"""
Precompute H3 level 9 community mappings with neighbors
Creates a mapping of each H3 L9 hex to its community and its 6 neighbors' communities
"""
import pandas as pd
import h3
import json
import numpy as np
from collections import defaultdict

print("="*80)
print("PRECOMPUTING H3 L9 COMMUNITY NEIGHBOR MAPPINGS")
print("="*80)

# Load the sales data to get H3 indices
print("\n[1/5] Loading sales data...")
sales_df = pd.read_csv('data/sales_2020_25_with_predictions_v3.csv')
print(f"  Loaded {len(sales_df)} sales records")

# Get unique H3 level 9 indices
if 'h3_09' not in sales_df.columns:
    print("\n  Generating H3 level 9 indices from lat/lng...")
    sales_df['h3_09'] = sales_df.apply(
        lambda row: h3.latlng_to_cell(row['lat'], row['lng'], 9),
        axis=1
    )
    # Save back
    sales_df.to_csv('data/sales_2020_25_with_predictions_v3.csv', index=False)
    print(f"  ✓ Generated and saved H3 L9 indices")

active_l9_hexes = sales_df['h3_09'].unique()
print(f"  Found {len(active_l9_hexes)} unique H3 L9 hexes")

# Load community mapping
print("\n[2/5] Loading community mapping...")
with open('data/community_map.json', 'r') as f:
    community_map_l7 = json.load(f)

print(f"  Loaded {len(community_map_l7)} H3 L7 -> community mappings")

# Create L9 to community mapping by looking up parent L7
print("\n[3/5] Creating H3 L9 -> community mapping...")
l9_to_community = {}
unmapped_count = 0

for hex_l9 in active_l9_hexes:
    # Get parent L7 hex
    hex_l7 = h3.cell_to_parent(hex_l9, 7)
    
    # Look up community
    if hex_l7 in community_map_l7:
        l9_to_community[hex_l9] = community_map_l7[hex_l7]
    else:
        unmapped_count += 1
        # Assign to unknown community (will be handled later)
        l9_to_community[hex_l9] = -1

print(f"  Mapped {len(l9_to_community) - unmapped_count} L9 hexes to communities")
print(f"  Unmapped: {unmapped_count} L9 hexes")

# Get unique communities
communities = sorted([c for c in set(l9_to_community.values()) if c != -1])
print(f"  Unique communities: {len(communities)}")

# Create community vocabulary (community_id -> index)
community_vocab = {comm_id: idx for idx, comm_id in enumerate(communities)}
UNKNOWN_COMMUNITY_ID = len(communities)  # Index for unknown/padding
vocab_size = len(communities) + 1

print(f"  Vocabulary size: {vocab_size} (including UNKNOWN)")

# Precompute neighbor mappings
print("\n[4/5] Precomputing neighbor community mappings...")
l9_neighbor_map = {}
neighbor_stats = defaultdict(int)

for i, hex_id in enumerate(active_l9_hexes):
    if (i + 1) % 1000 == 0:
        print(f"  Processing {i+1}/{len(active_l9_hexes)}...", end='\r')
    
    # Get hex + 6 neighbors (grid_disk with k=1 returns center + ring)
    try:
        disk = list(h3.grid_disk(hex_id, 1))
    except:
        # Fallback for older h3 versions
        disk = list(h3.k_ring(hex_id, 1))
    
    neighbor_stats[len(disk)] += 1
    
    # Map them to their community IDs
    community_ids = []
    for n_id in disk:
        if n_id in l9_to_community:
            comm_id = l9_to_community[n_id]
            if comm_id == -1:
                community_ids.append(UNKNOWN_COMMUNITY_ID)
            else:
                # Convert to vocabulary index
                community_ids.append(community_vocab[comm_id])
        else:
            community_ids.append(UNKNOWN_COMMUNITY_ID)
    
    # Pad or truncate to exactly 7 elements (1 center + up to 6 neighbors)
    # H3 pentagons might have fewer neighbors
    community_ids = (community_ids + [UNKNOWN_COMMUNITY_ID] * 7)[:7]
    
    l9_neighbor_map[hex_id] = community_ids

print(f"\n  ✓ Processed {len(l9_neighbor_map)} L9 hexes")
print(f"\n  Neighbor count distribution:")
for count in sorted(neighbor_stats.keys()):
    print(f"    {count} neighbors: {neighbor_stats[count]} hexes")

# Save mappings
print("\n[5/5] Saving mappings...")

# Save L9 to community mapping
output_l9_community = 'data/h3_l9_to_community.json'
with open(output_l9_community, 'w') as f:
    json.dump(l9_to_community, f)
print(f"  ✓ Saved L9->community mapping to {output_l9_community}")

# Save L9 neighbor community mapping
output_l9_neighbors = 'data/h3_l9_neighbor_communities.json'
with open(output_l9_neighbors, 'w') as f:
    json.dump(l9_neighbor_map, f)
print(f"  ✓ Saved L9 neighbor mappings to {output_l9_neighbors}")

# Save community vocabulary
output_vocab = 'data/community_vocab_l9.json'
vocab_data = {
    'community_to_idx': community_vocab,
    'idx_to_community': {idx: comm_id for comm_id, idx in community_vocab.items()},
    'vocab_size': vocab_size,
    'unknown_idx': UNKNOWN_COMMUNITY_ID,
    'num_communities': len(communities)
}
with open(output_vocab, 'w') as f:
    json.dump(vocab_data, f, indent=2)
print(f"  ✓ Saved community vocabulary to {output_vocab}")

# Create summary statistics
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"\nH3 Level 9 Statistics:")
print(f"  Total L9 hexes: {len(active_l9_hexes)}")
print(f"  Mapped to communities: {len(l9_to_community) - unmapped_count}")
print(f"  Unmapped: {unmapped_count}")
print(f"\nCommunity Statistics:")
print(f"  Unique communities: {len(communities)}")
print(f"  Vocabulary size: {vocab_size} (including UNKNOWN)")
print(f"\nNeighbor Mapping:")
print(f"  Total mappings: {len(l9_neighbor_map)}")
print(f"  Each hex mapped to: 7 community indices (center + 6 neighbors)")

# Sample a few mappings
print(f"\nSample Mappings (first 5):")
for i, (hex_id, comm_ids) in enumerate(list(l9_neighbor_map.items())[:5]):
    center_comm = comm_ids[0]
    neighbor_comms = comm_ids[1:]
    print(f"  {hex_id}:")
    print(f"    Center community idx: {center_comm}")
    print(f"    Neighbor community idxs: {neighbor_comms}")

print("\n" + "="*80)
print("COMPLETE - Ready for model training!")
print("="*80)
print("\nGenerated files:")
print(f"  - {output_l9_community}")
print(f"  - {output_l9_neighbors}")
print(f"  - {output_vocab}")
