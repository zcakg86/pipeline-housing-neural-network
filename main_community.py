
#%%

sys.path.insert(0,os.getcwd()+'/src/pricemodel')
sys.path.insert(0,os.getcwd()+'/src/spatial')
sys.path.insert(0,os.getcwd()+'/src/panel')
pd.set_option('mode.chained_assignment', None)
import json
import pandas as pd
import numpy as np
import h3
#%%
reload(spatial_graph_detection)
from spatial_graph_detection import *

#%%
df = pd.read_csv('data/sales_2020_25.csv')
# drop column 'Unnamed: 0'
df = df.drop('Unnamed: 0', axis=1)
df['price_per_sqft']=df['sale_price']/df['sqft']
df['sale_date']=pd.to_datetime(df['sale_date'])
# remove null and zero values
#%%
df = df.dropna(subset=['sale_price', 'lat', 'lng', 'sqft', 'sale_nbr', 'sale_date','sqft_lot'])
df = df[df['sale_price'] > 0]
df = df[df['sqft'] > 0]
df = df[df['sale_nbr'] > 0]
df = df.h3.geo_to_h3(resolution = 9, lat_col = 'lat', lng_col = 'lng', 
                             set_index = False)
#%% 
# Name of your location column
location_col = 'h3_09'

print("Starting Community Detection Pipeline...\n")

# Run the main analysis pipeline
(location_features, 
    G, 
    features_df, 
    features_array, 
    community_dict, 
    summary, 
    community_stats) = run_community_analysis(df=df,
                            location_var=location_col, 
                            # try to get 2 neighbours minimum
                            min_neighbors = 2,
                            # look 6 neighbours away for the neighbours
                            max_k = 6,
                            # don't communities to get bigger than this
                            max_comm_size = 150)

# Display the summary of the detected communities
print("\nCommunity Summary:")
print(summary[['size']].head())

# ==========================================
# 3. EXPORT TO JSON
# ==========================================
output_filename = "h3_09_community_mappings.json"

# Save the dictionary mapping { 'h3_09_string': community_id } to JSON
with open(output_filename, 'w') as f:
    json.dump(community_dict, f, indent=4)
    
print(f"\n✅ Successfully saved community mappings to '{output_filename}'")

# Optional: If you also want to save the aggregated community stats for your Transformer
stats_filename = "community_aggregated_stats.csv"
community_stats.to_csv(stats_filename, index=False)
print(f"✅ Successfully saved community statistical features to '{stats_filename}'")
# %%
