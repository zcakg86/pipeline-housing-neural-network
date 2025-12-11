#%%
# Using seaborn color palettes
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h3
import h3pandas
#import contextily as ctx

def vectorized_get_parent_h3(series, target_res):
    return series.map(lambda x: h3.cell_to_parent(x, target_res))

def h3_map(df, color, group_by=None):
    "Draw chloropleth based on 'color'"
    if color is None:
        color = 'h3_parent'
    print(df[color].dtype)
    if group_by is not None:
        # aggregate color value by group, and replace original value
        agg_dict = df.groupby(by=group_by)[color].median().to_dict()
        df[color] = df[group_by].map(agg_dict)

    if df[color].dtype == 'object': # for categorical/discrete variables
        unique_val = df[color].unique()
        colors = sns.color_palette("Paired",n_colors=len(unique_val))
        color_map = dict(zip(unique_val, colors))
    else: 
        color_map = sns.color_palette("YlOrBr", as_cmap=True)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect(1 / np.cos(np.radians(df['lat'].mean())))
    sns.scatterplot(data=df, x='lng', y='lat', hue=color, s=5, palette=color_map,
                     legend = True)
        
    # Convert lat/lon to Web Mercator projection
    ax.set_xlim(df['lng'].min(), df['lng'].max())
    ax.set_ylim(df['lat'].min(), df['lat'].max())

    # Add OSM background
    ctx.add_basemap(
        ax,
        crs='EPSG:4326',  # lat/lon projection
        source=ctx.providers.OpenStreetMap.Mapnik
    )
    
    plt.show()
    plt.savefig('map_test.png')
    return ax
# %%
