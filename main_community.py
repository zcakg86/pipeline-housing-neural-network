#%%
import sys
sys.path.append('src')

import json
import pandas as pd
import numpy as np
import h3
pd.set_option('mode.chained_assignment', None)
from spatial.spatial_graph_detection import *

#%%
df = pd.read_csv('data/sales_2020_25.csv')
# drop column 'Unnamed: 0'
df = df.drop('Unnamed: 0', axis=1)
df['price_per_sqft']=df['sale_price']/df['sqft']
df['sale_date']=pd.to_datetime(df['sale_date'])
# remove null and zero values
df = df.dropna(subset=['sale_price', 'lat', 'lng', 'sqft', 'sale_nbr', 'sale_date','sqft_lot'])
df = df[df['sale_price'] > 0]
df = df[df['sqft'] > 0]
df = df[df['sale_nbr'] > 0]
df['h3_08'] = df.apply(
    lambda row: h3.latlng_to_cell(row['lat'], row['lng'], 8) 
    if pd.notna(row['lat']) and pd.notna(row['lng']) 
    else None,
    axis=1
)
#%% 
# Name of your location column
location_col = 'h3_08'

print("Starting Community Detection Pipeline...\n")

# Run the main analysis pipeline
(location_features, 
    G, 
    features_df, 
    features_array, 
    community_dict, 
    summary) = run_community_analysis(df=df,
                            location_var=location_col, 
                            min_neighbors=2,
                            max_k=6,
                            max_comm_size=50,   # hard cap per community
                            base_res=1,       # start higher to favour smaller communities
                            seed=42)


# ==========================================
# 3. EXPORT TO JSON
# ==========================================
output_filename = "data/community_map.json"

# Save the dictionary mapping { 'h3_08_string': community_id } to JSON
with open(output_filename, 'w') as f:
    json.dump(community_dict, f, indent=4)
    
print(f"\n✅ Successfully saved community mappings to '{output_filename}'")

# %%
