#%%
import pandas as pd
import networkx as nx
from community import best_partition # python-louvain
import numpy as np
from scipy.stats import zscore

def create_location_network(df, location_var=None):
    """
    Creates network of locations based on similarity of sale_price patterns and characteristics
    
    Parameters:
    df: DataFrame with columns: 'location', 'sale_price', 'sale_date', and property characteristics
    """
    # Calculate location-level features
    location_features = {}
    print(f'Locations: {df[location_var].unique().shape[0]}')

    for location in df[location_var].unique():
        loc_data = df[df[location_var] == location]
        metrics = {
            'price_per_sqft': loc_data.groupby(pd.Grouper(key='sale_date', freq='YE'))['price_per_sqft'].median(),
            'price_sqft_std': loc_data.groupby(pd.Grouper(key='sale_date', freq='YE'))['price_per_sqft'].std(),
            'sqft': loc_data.groupby(pd.Grouper(key='sale_date', freq='YE'))['sqft'].median(),
            'lot': loc_data.groupby(pd.Grouper(key='sale_date', freq='YE'))['sqft_lot'].median(),
            'sqft_std': loc_data.groupby(pd.Grouper(key='sale_date', freq='YE'))['sqft'].std(ddof=0),
            'lot_std': loc_data.groupby(pd.Grouper(key='sale_date', freq='YE'))['sqft_lot'].std(ddof=0),
            'beds': loc_data.groupby(pd.Grouper(key='sale_date', freq='YE'))['sale_nbr'].median(),
            'lng': loc_data['lng'].mean(),
            'lat': loc_data['lat'].mean()
        }
        location_features[location] = metrics

    # Create graph
    G = nx.Graph()
    # Add nodes
    for loc in location_features.keys():
        G.add_node(loc)

    # Create array with location codes and features
    features_array = []
    locations = list(location_features.keys())

    for loc in locations:
        # Extract values from the nested dictionary/series
        loc_features = [
            location_features[loc]['price_per_sqft'].mean(),
            location_features[loc]['sqft'].mean(),
            location_features[loc]['sqft_std'].mean(),
            location_features[loc]['lot'].mean(),
            location_features[loc]['lot_std'].mean(), 
            location_features[loc]['beds'].mean(),
            location_features[loc]['lat'],
            location_features[loc]['lng']
        ]
        features_array.append(loc_features)
    
    # Convert to numpy array and standardize
    features_array = np.array(features_array)
    features_standardized = zscore(features_array, axis=0, nan_policy='omit')
    
    # Create DataFrame with location codes and standardized features
    features_df = pd.DataFrame(
        features_standardized, 
        index=locations,  # This keeps location codes as index
        columns=['price_per_sqft' ,'sqft'
                 ,'sqft_std','lot','lot_std', 'beds', 'lat', 'lng'
        ]
    )
    # Create edges using standardized features
    for loc1 in features_df.index:
        for loc2 in features_df.index[features_df.index > loc1]:  # More efficient way to avoid duplicates
            similarity = 1 / (1 + np.linalg.norm(features_df.loc[loc1] - features_df.loc[loc2]))
            if similarity >= 0.5:
                G.add_edge(loc1, loc2, weight=similarity)
    
    return G, location_features, features_df, features_array


def detect_communities(G, method = 'gn', res=1):
    """
    Detects communities in the location network
    """
    def most_central_edge(G):
        centrality = nx.edge_betweenness_centrality(G, weight="weight")
        return max(centrality, key=centrality.get)
    # Use Louvain method for community detection
    print(f'Resolution: {res}')
    if method == 'gn':

        communities_generator = nx.community.girvan_newman(G, most_valuable_edge=most_central_edge)
        # Convert generator to list of tuples for multiple uses
        # Each tuple contains sets of nodes representing communities
        community_list = []
        modularity_scores = []
        
        # Evaluate each division
        for communities in communities_generator:
            # Convert communities to list of sets for modularity calculation
            communities = tuple(sorted(c) for c in communities)
            community_list.append(communities)
            
            # Calculate modularity score
            mod_score = nx.community.modularity(G, communities)
            modularity_scores.append(mod_score)
            
            # Optional: stop if we have too many communities
            if len(communities) > len(G.nodes) / 5:  # or some other threshold
                break
        
        # Find best division
        best_idx = np.argmax(modularity_scores)
        best_communities = community_list[best_idx]
        
        print(f"Best modularity score: {modularity_scores[best_idx]}")
        print(f"Number of communities: {len(best_communities)}")

    elif method == 'l':
        best_communities = nx.community.louvain_communities(G, resolution=res)
        print(f"Number of communities: {len(best_communities)}")
    # Create a dictionary mapping nodes to their community
    community_dict = {}
    for i, community in enumerate(best_communities):
        for node in community:
            community_dict[node] = i
    
    # Create summary of communities
    community_summary = pd.DataFrame({
        'community': community_dict.values(),
        'location': community_dict.keys()
    }).groupby('community').agg(
        locations=('location', list),
        size=('location', 'size')
    ).sort_values('size', ascending=False)
    
    return community_dict, community_summary


def communities_df(input):
    '''Create datafame from dict'''
    df = pd.DataFrame([(k, i) for k, v in input.items() for i in v], 
                      columns=['community', 'h3_index'])
    return df
    

def analyze_communities(df, communities, location_var):
    """
    Creates summary statistics for each community
    """
    # Initialize community statistics
    community_stats = pd.DataFrame()
    
    for i, locations in zip(communities.index, communities['locations']):
        # Get data for all locations in community
        community_data = df[df[location_var].isin(locations)]
        # Calculate community statistics
        stats = {
            'community_id': i,
            'num_locations': len(locations),
            'num_transactions': len(community_data),
            'avg_sale_price': community_data['sale_price'].mean(),
            'sale_price_std': community_data['sale_price'].std(ddof=0),
            'price_per_sqft': community_data['price_per_sqft'].mean(),
            'locations': ', '.join(locations)
        }
        
        # Add time-based metrics if available
        if 'sale_date' in df.columns:
            monthly_sale_prices = community_data.groupby(
                pd.to_datetime(community_data['sale_date']).dt.to_period('M')
            )['sale_price'].mean()
            
            stats.update({
                'sale_price_trend': monthly_sale_prices.pct_change().mean(),
                'sale_price_volatility': monthly_sale_prices.pct_change().std(ddof=0)
            })
        
        # Add to community statistics
        community_stats = pd.concat([
            community_stats,
            pd.DataFrame([stats])
        ])
    
    return community_stats.set_index('community_id',drop=False)

# Example execution:
def run_community_analysis(df, location_var, method, resolution=1):
    """
    Complete execution example
    """
    print("Creating location network...")
    G, location_features, features_df, array = create_location_network(df, location_var)
    print(f"Network created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    print("\nDetecting communities...")
    communities, summary = detect_communities(G, method, resolution)
    print(f"Found {len(communities)} communities")
    
    print("\nAnalyzing communities...")
    community_stats = analyze_communities(df, communities, location_var)
    
    return location_features, G, features_df, array, communities, summary, community_stats

# # Example usage with sample data:
# sample_data = pd.DataFrame({
#     'location': ['A', 'A', 'B', 'B', 'C', 'C'],
#     'sale_price': [100000, 120000, 200000, 220000, 150000, 160000],
#     'sqft': [100, 120, 150, 160, 130, 140],
#     'sale_date': pd.to_datetime(['2023-01-01', '2023-02-01', '2023-01-15', '2023-02-15','2023-01-01', '2023-02-01']),
#     'lat':[47.5,47.6,47.55,47.55,47.6,47.53],
#     'lng':[-122.23,-122.24,-122.24,-122.43,-122.42,-122.32]
# })
# sample_data['price_per_sqft']=sample_data['sale_price']/sample_data['sqft']

# G, communities, stats = run_community_analysis(sample_data, location_var = 'location')
# print("\nCommunity Statistics:")
# print(stats)

# %%
