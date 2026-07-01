"""
H3 L9 Neighbor Mapper
Computes and caches h3_l9_neighbor_communities.json:
  { h3_09_hex: [community_idx_center, n1, n2, n3, n4, n5, n6] }

Community indices come from community_map.json (h3_09 → community ID).
Rows with no community mapping get the unknown index (= max community ID + 1).
No community vocab file is saved — community_map.json is the source of truth.
"""
import json
import os
import h3


def compute_neighbor_communities(
    dataframe,
    community_map_path: str = 'data/community_map.json',
    output_dir: str = 'data',
    force_recompute: bool = False,
):
    """
    Build (or load from cache) the h3_l8_neighbor_communities.json mapping.

    For every unique h3_09 hex in *dataframe*, computes grid_disk(hex, 1) —
    the center cell plus its 6 H3 neighbours — then maps each to its community
    index using community_map.json.  Unknown hexes get the unknown index.

    Parameters
    ----------
    dataframe         : DataFrame that must have an 'h3_09' column.
                        If it also has a 'community' column that column is used
                        directly; otherwise community_map.json is loaded and
                        applied first.
    community_map_path: Path to community_map.json  (h3_09 hex → community int).
    output_dir        : Directory to read/write the cached neighbor file.
    force_recompute   : Ignore the cache and recompute from scratch.

    Returns
    -------
    h3_neighbor_map : dict  { h3_08_hex: [int × 7] }
    n_communities   : int   number of real communities (unknown = n_communities)
    """
    output_path = os.path.join(output_dir, 'h3_l8_neighbor_communities.json')

    # --- Load cache if available ---
    if not force_recompute and os.path.exists(output_path):
        print(f"  Loading cached neighbor map from {output_path}...")
        with open(output_path, 'r') as f:
            h3_neighbor_map = json.load(f)
        # Derive n_communities from the community_map
        n_communities = _load_n_communities(community_map_path)
        print(f"  ✓ Loaded {len(h3_neighbor_map)} hex mappings  "
              f"(n_communities={n_communities}, unknown={n_communities})")
        return h3_neighbor_map, n_communities

    # --- Load community_map ---
    if not os.path.exists(community_map_path):
        raise FileNotFoundError(
            f"community_map.json not found at '{community_map_path}'. "
            "This file must exist before neighbor communities can be computed."
        )
    print(f"  Loading community map from {community_map_path}...")
    with open(community_map_path, 'r') as f:
        community_map = json.load(f)   # { h3_09_hex: community_id (int) }

    n_communities = max(community_map.values()) + 1
    unknown_idx   = n_communities                 # one past the last real community
    print(f"  Community map: {len(community_map)} hexes, "
          f"{n_communities} communities, unknown index = {unknown_idx}")

    # --- Ensure 'community' column exists on the dataframe ---
    df = dataframe.copy()
    if 'community' not in df.columns:
        print("  'community' column not found — deriving from community_map.json...")
        df['community'] = df['h3_08'].map(community_map)
        mapped = df['community'].notna().sum()
        print(f"  ✓ Mapped {mapped}/{len(df)} rows  ({len(df)-mapped} unmapped → unknown)")

    # --- Compute neighbor communities for every unique h3_09 ---
    active_hexes = df['h3_08'].dropna().unique()
    print(f"  Computing neighbor communities for {len(active_hexes)} unique H3 L8 hexes...")

    h3_neighbor_map = {}
    for i, hex_id in enumerate(active_hexes):
        if (i + 1) % 5000 == 0:
            print(f"    {i+1}/{len(active_hexes)}", end='\r')

        # grid_disk(k=1) returns center + up to 6 neighbours (always 7 for interior cells)
        try:
            disk = list(h3.grid_disk(hex_id, 1))
        except AttributeError:
            disk = list(h3.k_ring(hex_id, 1))   # older h3 API

        # Map each disk hex to its community index; pad/truncate to exactly 7
        community_ids = [
            community_map.get(n_hex, unknown_idx)
            for n_hex in disk
        ]
        community_ids = (community_ids + [unknown_idx] * 7)[:7]

        h3_neighbor_map[hex_id] = community_ids

    print(f"\n  ✓ Computed {len(h3_neighbor_map)} neighbor mappings")

    # --- Save cache ---
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(h3_neighbor_map, f)
    print(f"  ✓ Saved to {output_path}")

    return h3_neighbor_map, n_communities


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_n_communities(community_map_path: str) -> int:
    """Return the number of real communities from community_map.json."""
    with open(community_map_path, 'r') as f:
        community_map = json.load(f)
    return max(community_map.values()) + 1


def ensure_neighbor_communities(
    dataframe,
    community_map_path: str = 'data/community_map.json',
    output_dir: str = 'data',
):
    """
    Convenience wrapper: load from cache or compute, never force-recompute.

    Returns
    -------
    h3_neighbor_map : dict  { h3_08 hex: [int × 7] }
    n_communities   : int
    """
    return compute_neighbor_communities(
        dataframe,
        community_map_path=community_map_path,
        output_dir=output_dir,
        force_recompute=False,
    )
