import h3
import networkx as nx
import pandas as pd
import numpy as np


def create_location_network(df, location_var, max_k=5, min_neighbors=2):
    """
    Creates a spatial network using Dynamic Ring Expansion.
    - Dense areas: Connects at k=1 and stops.
    - Sparse areas: Expands up to max_k to find 'min_neighbors', avoiding islands.
    """
    active_hexes = set(df[location_var].unique())
    print(f'Active H3 Locations: {len(active_hexes)}')

    G = nx.Graph()
    G.add_nodes_from(active_hexes)

    for hex_id in active_hexes:
        neighbors_found = 0
        prev_disk = set([hex_id])

        for k in range(1, max_k + 1):
            current_disk = set(h3.grid_disk(hex_id, k))
            ring = current_disk - prev_disk
            prev_disk = current_disk

            for neighbor in ring:
                if neighbor in active_hexes:
                    weight = 1.0 / k
                    G.add_edge(hex_id, neighbor, weight=weight)
                    neighbors_found += 1

            if neighbors_found >= min_neighbors:
                break

    print(f'Created {G.number_of_edges()} dynamic spatial edges.')

    df_temp = df.copy()
    df_temp['sale_year'] = pd.to_datetime(df_temp['sale_date']).dt.year
    yearly_metrics = df_temp.groupby([location_var, 'sale_year']).agg({
        'price_per_sqft': ['median', 'std'],
        'sqft': ['median', 'std'],
        'sqft_lot': ['median', 'std'],
        'sale_nbr': 'median'
    }).fillna(0)
    yearly_metrics.columns = ['price_per_sqft', 'price_sqft_std', 'sqft', 'sqft_std', 'lot', 'lot_std', 'beds']
    features_df = yearly_metrics.groupby(level=location_var).mean()

    return G, features_df.to_dict('index'), features_df, features_df.values


def detect_communities(G, base_res=1.0, max_comm_size=50, seed=42):
    """
    Uses Recursive Louvain with escalating resolution to ensure no community
    exceeds max_comm_size.  If Louvain gets stuck and cannot split a subgraph
    further, a spatial hard-split forcibly partitions it into equal-sized chunks.

    Parameters
    ----------
    G            : networkx Graph with 'weight' edges
    base_res     : starting Louvain resolution (higher = more/smaller communities)
    max_comm_size: hard upper bound on community size in hexes
    seed         : random seed for Louvain reproducibility
    """
    print(f'Running Recursive Louvain (max_comm_size={max_comm_size}, base_res={base_res})...')

    def _louvain_split(sub_G, res):
        """Run Louvain on sub_G and return list of community node-sets."""
        return nx.community.louvain_communities(
            sub_G, weight='weight', resolution=res, seed=seed
        )

    def _hard_split(nodes, size):
        """
        Forcibly partition a set of nodes into chunks of at most `size`.
        Used when Louvain cannot split a subgraph any further.
        """
        node_list = list(nodes)
        return [
            set(node_list[i:i + size])
            for i in range(0, len(node_list), size)
        ]

    def _recursive_louvain(sub_G, res, depth=0):
        comms = _louvain_split(sub_G, res)

        final_comms = []
        for comm in comms:
            if len(comm) <= max_comm_size:
                final_comms.append(comm)
                continue

            # Community is too large — escalate resolution and recurse
            # Use an exponential step so resolution grows fast enough
            # to actually fracture very dense subgraphs
            new_res = res * 1.5 + 0.5

            fracture_G = sub_G.subgraph(comm).copy()
            fractured = _recursive_louvain(fracture_G, new_res, depth + 1)

            # Check if Louvain actually managed to split it
            if len(fractured) == 1 and len(fractured[0]) > max_comm_size:
                # Louvain is stuck — fall back to hard spatial split
                print(f"  [depth={depth}] Louvain stuck on {len(comm)} nodes at res={new_res:.2f}"
                      f" — applying hard split into chunks of {max_comm_size}")
                final_comms.extend(_hard_split(comm, max_comm_size))
            else:
                final_comms.extend(fractured)

        return final_comms

    best_communities = _recursive_louvain(G, base_res)

    # Paranoia pass: hard-split anything still over the limit
    checked = []
    for comm in best_communities:
        if len(comm) > max_comm_size:
            checked.extend(_hard_split(comm, max_comm_size))
        else:
            checked.append(comm)
    best_communities = checked

    print(f"Final number of communities: {len(best_communities)}")

    # Build mapping dict
    community_dict = {}
    for i, community in enumerate(best_communities):
        for node in community:
            community_dict[node] = i

    # Summary dataframe
    community_summary = pd.DataFrame({
        'community': list(community_dict.values()),
        'location':  list(community_dict.keys())
    }).groupby('community').agg(
        locations=('location', list),
        size=('location', 'size')
    ).sort_values('size', ascending=False)

    # ── Community size statistics ─────────────────────────────────────────────
    sizes = community_summary['size']
    print(f"\nCommunity size summary ({len(sizes)} communities):")
    print(f"  Total locations assigned : {sizes.sum()}")
    print(f"  Mean size                : {sizes.mean():.1f}")
    print(f"  Median size              : {sizes.median():.1f}")
    print(f"  Std dev                  : {sizes.std():.1f}")
    print(f"  Min size                 : {sizes.min()}")
    print(f"  Max size                 : {sizes.max()}")
    print(f"  10th percentile          : {sizes.quantile(0.10):.1f}")
    print(f"  25th percentile          : {sizes.quantile(0.25):.1f}")
    print(f"  75th percentile          : {sizes.quantile(0.75):.1f}")
    print(f"  90th percentile          : {sizes.quantile(0.90):.1f}")
    cap = max_comm_size
    bins   = [1, 5, 10, 25, 50, cap + 1, cap * 3 + 1]
    labels = ['1–4', '5–9', '10–24', '25–49', f'50–{cap}', f'>{cap}']
    # Walk the bin edges and stop at the first one that exceeds max_val,
    # so the bin that contains max_val is always included.
    max_val = int(sizes.max())
    used_bins = []
    for b in bins:
        used_bins.append(b)
        if b > max_val:
            break
    used_labels = labels[:len(used_bins) - 1]
    if len(used_bins) >= 2:
        counts = pd.cut(sizes, bins=used_bins, labels=used_labels, right=False).value_counts().sort_index()
        print(f"  Size distribution:")
        for label, count in counts.items():
            marker = ' ⚠' if str(label).startswith('>') else ''
            print(f"    {str(label):>12} hexes: {count} communities{marker}")

    return community_dict, community_summary


def run_community_analysis(df, location_var, max_k=5, max_comm_size=50,
                           min_neighbors=2, base_res=1.0, seed=42):
    print("1. Creating dynamic spatial network...")
    G, location_features, features_df, array = create_location_network(
        df, location_var, max_k=max_k, min_neighbors=min_neighbors
    )

    print("2. Detecting balanced communities...")
    community_dict, summary = detect_communities(
        G, base_res=base_res, max_comm_size=max_comm_size, seed=seed
    )

    # ── Allocation coverage ───────────────────────────────────────────────────
    total_rows    = len(df)
    assigned_mask = df[location_var].isin(community_dict)
    assigned_rows = assigned_mask.sum()
    unassigned_rows = total_rows - assigned_rows

    unique_locs     = df[location_var].nunique()
    assigned_locs   = len(community_dict)
    unassigned_locs = unique_locs - assigned_locs

    print(f"\nAllocation coverage:")
    print(f"  Rows  — assigned     : {assigned_rows:>7,}  ({100*assigned_rows/total_rows:.1f}%)")
    print(f"  Rows  — no community : {unassigned_rows:>7,}  ({100*unassigned_rows/total_rows:.1f}%)")
    print(f"  Locations — assigned     : {assigned_locs:>6,}  of {unique_locs:,} unique")
    print(f"  Locations — no community : {unassigned_locs:>6,}")

    return location_features, G, features_df, array, community_dict, summary