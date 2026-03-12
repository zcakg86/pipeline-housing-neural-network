"""
H3 L9 Neighbor Mapper
Automatically computes and caches H3 L9 community neighbor mappings
"""
import pandas as pd
import h3
import json
import os
from collections import defaultdict


def compute_h3_l9_neighbor_mappings(dataframe, community_map_path='data/community_map.json',
                                     output_dir='data', force_recompute=False):
    """
    Compute H3 L9 community neighbor mappings from a dataframe
    
    Args:
        dataframe: DataFrame with 'h3_09' column (or 'lat'/'lng' to generate it)
        community_map_path: Path to H3 L7 -> community mapping JSON
        output_dir: Directory to save output files
        force_recompute: If True, recompute even if files exist
    
    Returns:
        tuple: (l9_to_community, l9_neighbor_map, community_vocab)
    """
    
    # Check if files already exist
    output_l9_community = os.path.join(output_dir, 'h3_l9_to_community.json')
    output_l9_neighbors = os.path.join(output_dir, 'h3_l9_neighbor_communities.json')
    output_vocab = os.path.join(output_dir, 'community_vocab_l9.json')
    
    if not force_recompute and all(os.path.exists(f) for f in [output_l9_community, output_l9_neighbors, output_vocab]):
        print("  H3 L9 neighbor mappings already exist, loading from cache...")
        with open(output_l9_community, 'r') as f:
            l9_to_community = json.load(f)
        with open(output_l9_neighbors, 'r') as f:
            l9_neighbor_map = json.load(f)
        with open(output_vocab, 'r') as f:
            vocab_data = json.load(f)
        print(f"  ✓ Loaded {len(l9_neighbor_map)} L9 hex mappings from cache")
        return l9_to_community, l9_neighbor_map, vocab_data
    
    print("  Computing H3 L9 community neighbor mappings...")
    
    # Ensure h3_09 column exists
    df = dataframe.copy()
    if 'h3_09' not in df.columns:
        if 'lat' not in df.columns or 'lng' not in df.columns:
            raise ValueError("DataFrame must have either 'h3_09' or 'lat'/'lng' columns")
        print("    Generating H3 L9 indices from lat/lng...")
        df['h3_09'] = df.apply(
            lambda row: h3.latlng_to_cell(row['lat'], row['lng'], 9),
            axis=1
        )
    
    # Get unique H3 level 9 indices
    active_l9_hexes = df['h3_09'].dropna().unique()
    print(f"    Found {len(active_l9_hexes)} unique H3 L9 hexes")
    
    # Load community mapping (H3 L7 -> community)
    if not os.path.exists(community_map_path):
        raise FileNotFoundError(f"Community map not found: {community_map_path}")
    
    with open(community_map_path, 'r') as f:
        community_map_l7 = json.load(f)
    print(f"    Loaded {len(community_map_l7)} H3 L7 -> community mappings")
    
    # Create L9 to community mapping by looking up parent L7
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
            l9_to_community[hex_l9] = -1  # Unknown
    
    print(f"    Mapped {len(l9_to_community) - unmapped_count}/{len(active_l9_hexes)} L9 hexes")
    
    # Get unique communities
    communities = sorted([c for c in set(l9_to_community.values()) if c != -1])
    print(f"    Unique communities: {len(communities)}")
    
    # Create community vocabulary (community_id -> index)
    community_vocab = {comm_id: idx for idx, comm_id in enumerate(communities)}
    UNKNOWN_COMMUNITY_ID = len(communities)  # Index for unknown/padding
    vocab_size = len(communities) + 1
    
    # Precompute neighbor mappings
    print(f"    Computing neighbor mappings for {len(active_l9_hexes)} hexes...")
    l9_neighbor_map = {}
    
    for i, hex_id in enumerate(active_l9_hexes):
        if (i + 1) % 2000 == 0:
            print(f"      Progress: {i+1}/{len(active_l9_hexes)}", end='\r')
        
        # Get hex + 6 neighbors (grid_disk with k=1 returns center + ring)
        try:
            disk = list(h3.grid_disk(hex_id, 1))
        except:
            # Fallback for older h3 versions
            disk = list(h3.k_ring(hex_id, 1))
        
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
        community_ids = (community_ids + [UNKNOWN_COMMUNITY_ID] * 7)[:7]
        
        l9_neighbor_map[hex_id] = community_ids
    
    print(f"\n    ✓ Computed {len(l9_neighbor_map)} neighbor mappings")
    
    # Save mappings
    os.makedirs(output_dir, exist_ok=True)
    
    with open(output_l9_community, 'w') as f:
        json.dump(l9_to_community, f)
    
    with open(output_l9_neighbors, 'w') as f:
        json.dump(l9_neighbor_map, f)
    
    vocab_data = {
        'community_to_idx': community_vocab,
        'idx_to_community': {idx: comm_id for comm_id, idx in community_vocab.items()},
        'vocab_size': vocab_size,
        'unknown_idx': UNKNOWN_COMMUNITY_ID,
        'num_communities': len(communities)
    }
    
    with open(output_vocab, 'w') as f:
        json.dump(vocab_data, f, indent=2)
    
    print(f"    ✓ Saved mappings to {output_dir}/")
    
    return l9_to_community, l9_neighbor_map, vocab_data


def ensure_h3_l9_mappings(dataframe, community_map_path='data/community_map.json',
                          output_dir='data'):
    """
    Ensure H3 L9 neighbor mappings exist, computing them if necessary
    
    Args:
        dataframe: DataFrame with h3_09 or lat/lng columns
        community_map_path: Path to community map JSON
        output_dir: Directory for output files
    
    Returns:
        tuple: (l9_to_community, l9_neighbor_map, vocab_data)
    """
    return compute_h3_l9_neighbor_mappings(
        dataframe, 
        community_map_path=community_map_path,
        output_dir=output_dir,
        force_recompute=False
    )
