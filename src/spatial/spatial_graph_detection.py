import h3
import networkx as nx
import pandas as pd

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
    
    edges_added = 0
    
    for hex_id in active_hexes:
        neighbors_found = 0
        prev_disk = set([hex_id])
        
        # Expand outward ring by ring
        for k in range(1, max_k + 1):
            # Safely get hexes at exactly distance k
            current_disk = set(h3.grid_disk(hex_id, k))
            ring = current_disk - prev_disk
            prev_disk = current_disk
            
            for neighbor in ring:
                if neighbor in active_hexes:
                    # Weight decays with distance: 1.0 for k=1, 0.5 for k=2, 0.33 for k=3...
                    # This tells Louvain to prefer closer nodes if available
                    weight = 1.0 / k
                    G.add_edge(hex_id, neighbor, weight=weight)
                    neighbors_found += 1
            
            # If this hex has found enough spatial neighbors, stop expanding!
            # This keeps dense areas strictly local, while sparse areas reach out.
            if neighbors_found >= min_neighbors:
                break
                
    print(f'Created {G.number_of_edges()} dynamic spatial edges.')

    # Vectorized feature calculation (unchanged, still super fast)
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


def detect_communities(G, base_res=1.0, max_comm_size=150):
    """
    Uses Recursive Louvain to ensure no community exceeds max_comm_size.
    """
    print(f'Running Recursive Louvain (Target Max Size: {max_comm_size})...')
    
    def _recursive_louvain(sub_G, current_res):
        # Run Louvain on the current graph/subgraph
        # We explicitly use the 'weight' we calculated during dynamic expansion
        comms = nx.community.louvain_communities(sub_G, weight='weight', resolution=current_res)
        
        final_comms = []
        for comm in comms:
            # If the community is too big, isolate it and fracture it!
            if len(comm) > max_comm_size and len(comm) > 1:
                # Extract just this mega-community
                fracture_G = sub_G.subgraph(comm)
                
                # Recurse with slightly higher resolution to force a split
                fractured_comms = _recursive_louvain(fracture_G, current_res + 0.5)
                final_comms.extend(fractured_comms)
            else:
                final_comms.append(comm)
                
        return final_comms

    # Trigger the recursive loop
    best_communities = _recursive_louvain(G, base_res)
    
    print(f"Final number of balanced communities: {len(best_communities)}")

    # Create mapping dictionary
    community_dict = {}
    for i, community in enumerate(best_communities):
        for node in community:
            community_dict[node] = i
            
    # Create Summary
    community_summary = pd.DataFrame({
        'community': list(community_dict.values()),
        'location': list(community_dict.keys())
    }).groupby('community').agg(
        locations=('location', list),
        size=('location', 'size')
    ).sort_values('size', ascending=False)
    
    return community_dict, community_summary

def run_community_analysis(df, location_var, max_k=5, max_comm_size=150, min_neighbors=2):
    print("1. Creating dynamic spatial network...")
    # Will reach out up to k=5 to find at least 2 neighbors
    G, location_features, features_df, array = create_location_network(
        df, location_var, max_k=max_k, min_neighbors=min_neighbors
    )
    
    print("2. Detecting balanced communities...")
    # Will aggressively fracture any community larger than max_comm_size
    community_dict, summary = detect_communities(
        G, base_res=1.0, max_comm_size=max_comm_size
    )
    
    print("3. Analyzing community performance metrics...")
    community_stats = analyze_communities(df, community_dict, location_var)
    
    return location_features, G, features_df, array, community_dict, summary, community_stats