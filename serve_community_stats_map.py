"""
Interactive Panel map showing community aggregated statistics
"""
import panel as pn
import holoviews as hv
import geoviews as gv
import pandas as pd
import numpy as np
import h3
from bokeh.models import NumeralTickFormatter, HoverTool
import cartopy.crs as ccrs

# Enable Panel extension
pn.extension()
hv.extension('bokeh')

print("Loading community aggregated stats...")

# Load the data
df = pd.read_csv('community_aggregated_stats.csv')
print(f"Loaded {len(df)} communities")

# Parse the locations column (comma-separated H3 indices)
df['locations_list'] = df['locations'].str.split(', ')
df['num_h3_cells'] = df['locations_list'].apply(len)

print(f"Total H3 cells across all communities: {df['num_h3_cells'].sum()}")

# Configuration for different variables
VARIABLE_CONFIGS = {
    'avg_sale_price': {
        'cmap': 'viridis',
        'format': '$0,0',
        'label': 'Avg Sale Price',
        'center_zero': False
    },
    'price_per_sqft': {
        'cmap': 'plasma',
        'format': '$0,0',
        'label': 'Price per Sqft',
        'center_zero': False
    },
    'num_transactions': {
        'cmap': 'YlOrRd',
        'format': '0,0',
        'label': 'Number of Transactions',
        'center_zero': False
    },
    'sale_price_std': {
        'cmap': 'coolwarm',
        'format': '$0,0',
        'label': 'Price Std Dev',
        'center_zero': False
    },
    'sale_price_trend': {
        'cmap': 'RdYlGn',
        'format': '0.000',
        'label': 'Price Trend',
        'center_zero': True
    },
    'sale_price_volatility': {
        'cmap': 'Reds',
        'format': '0.000',
        'label': 'Price Volatility',
        'center_zero': False
    },
    'num_locations': {
        'cmap': 'Blues',
        'format': '0,0',
        'label': 'Number of H3 Locations',
        'center_zero': False
    }
}


def create_hex_polygons_for_community(community_row, variable):
    """Convert all H3 indices in a community to polygon data"""
    
    polygons_data = []
    
    for h3_idx in community_row['locations_list']:
        h3_idx = h3_idx.strip()
        
        try:
            # Get boundary - h3 v4 API
            try:
                boundary = h3.cell_to_boundary(h3_idx)
                # New API returns (lat, lng) tuples
                lats = [coord[0] for coord in boundary]
                lons = [coord[1] for coord in boundary]
            except:
                # Try old API
                boundary = h3.h3_to_geo_boundary(h3_idx, geo_json=True)
                # Old API returns [lng, lat] lists
                lons = [coord[0] for coord in boundary]
                lats = [coord[1] for coord in boundary]
            
            # Create polygon data
            poly_data = {
                'Longitude': lons,
                'Latitude': lats,
                'value': community_row[variable],
                'community_id': community_row['community_id'],
                'num_transactions': community_row['num_transactions'],
                'avg_sale_price': community_row['avg_sale_price'],
                'price_per_sqft': community_row['price_per_sqft'],
                'sale_price_std': community_row['sale_price_std'],
                'sale_price_trend': community_row['sale_price_trend'],
                'sale_price_volatility': community_row['sale_price_volatility'],
                'num_locations': community_row['num_locations']
            }
            
            polygons_data.append(poly_data)
            
        except Exception as e:
            print(f"  Warning: Could not process H3 index {h3_idx}: {e}")
            continue
    
    return polygons_data


def create_community_map(variable, min_transactions=1, x_range=None, y_range=None):
    """Create hexagon map for selected variable"""
    
    config = VARIABLE_CONFIGS.get(variable, VARIABLE_CONFIGS['avg_sale_price'])
    
    # Filter by minimum transactions
    df_filtered = df[df['num_transactions'] >= min_transactions].copy()
    
    if len(df_filtered) == 0:
        return gv.Polygons([], crs=ccrs.PlateCarree()).opts(
            title="No Data in Range",
            width=900,
            height=700,
            xaxis=None,
            yaxis=None
        )
    
    print(f"Creating map for {variable} with {len(df_filtered)} communities...")
    
    # Calculate color limits
    vals = df_filtered[variable].dropna()
    if len(vals) > 0:
        vmin, vmax = np.percentile(vals, 5), np.percentile(vals, 95)
        if config.get('center_zero', False):
            limit = max(abs(vmin), abs(vmax))
            vmin, vmax = -limit, limit
    else:
        vmin, vmax = None, None
    
    # Create polygon data for all communities
    all_polygons = []
    for idx, row in df_filtered.iterrows():
        polygons = create_hex_polygons_for_community(row, variable)
        all_polygons.extend(polygons)
    
    print(f"  Created {len(all_polygons)} hexagons")
    
    if len(all_polygons) == 0:
        return gv.Polygons([], crs=ccrs.PlateCarree()).opts(
            title="No Data",
            width=900,
            height=700,
            xaxis=None,
            yaxis=None
        )
    
    # Tooltips
    hex_tooltips = [
        ('Community ID', '@community_id'),
        (config['label'], '@value{' + config['format'] + '}'),
        ('Transactions', '@num_transactions{0,0}'),
        ('Avg Price', '$@avg_sale_price{0,0}'),
        ('Price/Sqft', '$@price_per_sqft{0,0}'),
        ('Std Dev', '$@sale_price_std{0,0}'),
        ('Trend', '@sale_price_trend{0.000}'),
        ('Volatility', '@sale_price_volatility{0.000}'),
        ('H3 Locations', '@num_locations')
    ]
    
    # Create dimension with label
    value_dim = hv.Dimension('value', label=config['label'])
    
    # Create polygons using geoviews
    polygons = gv.Polygons(
        all_polygons,
        vdims=[value_dim, 'community_id', 'num_transactions', 'avg_sale_price',
               'price_per_sqft', 'sale_price_std', 'sale_price_trend',
               'sale_price_volatility', 'num_locations'],
        crs=ccrs.PlateCarree()
    ).opts(
        color='value',
        cmap=config['cmap'],
        clim=(vmin, vmax),
        colorbar=True,
        colorbar_opts={
            'formatter': NumeralTickFormatter(format=config['format']),
            'title': config['label']
        },
        line_color='white',
        line_width=0.5,
        alpha=0.7,
        width=900,
        height=700,
        xaxis=None,
        yaxis=None,
        tools=[HoverTool(tooltips=hex_tooltips)],
        title=f"{config['label']} by Community (min {min_transactions} transactions)"
    )
    
    return polygons


print("Creating widgets...")

# Variable selector
variable_selector = pn.widgets.Select(
    name='Variable to Display',
    options=list(VARIABLE_CONFIGS.keys()),
    value='avg_sale_price',
    width=300
)

# Minimum transactions filter
min_transactions_slider = pn.widgets.IntSlider(
    name='Min Transactions',
    start=1,
    end=100,
    value=10,
    step=5,
    width=300
)

# Statistics panel
def get_stats(variable, min_transactions):
    """Generate statistics for selected variable"""
    
    config = VARIABLE_CONFIGS.get(variable, VARIABLE_CONFIGS['avg_sale_price'])
    df_filtered = df[df['num_transactions'] >= min_transactions]
    
    if len(df_filtered) == 0:
        return pn.pane.Markdown("No communities match the filter criteria.")
    
    # Format values based on whether it's currency or not
    if '$' in config['format']:
        stats = f"""
    ### {config['label']}
    
    **Statistics:**
    - Min: ${df_filtered[variable].min():,.0f}
    - Max: ${df_filtered[variable].max():,.0f}
    - Mean: ${df_filtered[variable].mean():,.0f}
    - Median: ${df_filtered[variable].median():,.0f}
    - Std Dev: ${df_filtered[variable].std():,.0f}
    
    **Communities:**
    - Total: {len(df_filtered)}
    - Total Transactions: {df_filtered['num_transactions'].sum():,}
    - Total H3 Locations: {df_filtered['num_locations'].sum():,}
    
    **Top 5 Communities:**
    """
        # Add top 5
        top_5 = df_filtered.nlargest(5, variable)
        for idx, row in top_5.iterrows():
            stats += f"\n    - Community {row['community_id']}: ${row[variable]:,.0f} ({row['num_transactions']:,} txns)"
    else:
        stats = f"""
    ### {config['label']}
    
    **Statistics:**
    - Min: {df_filtered[variable].min():.3f}
    - Max: {df_filtered[variable].max():.3f}
    - Mean: {df_filtered[variable].mean():.3f}
    - Median: {df_filtered[variable].median():.3f}
    - Std Dev: {df_filtered[variable].std():.3f}
    
    **Communities:**
    - Total: {len(df_filtered)}
    - Total Transactions: {df_filtered['num_transactions'].sum():,}
    - Total H3 Locations: {df_filtered['num_locations'].sum():,}
    
    **Top 5 Communities:**
    """
        # Add top 5
        top_5 = df_filtered.nlargest(5, variable)
        for idx, row in top_5.iterrows():
            stats += f"\n    - Community {row['community_id']}: {row[variable]:.3f} ({row['num_transactions']:,} txns)"
    
    return pn.pane.Markdown(stats)


print("Creating dynamic map...")

# Create dynamic map
dmap = gv.DynamicMap(
    pn.bind(
        create_community_map,
        variable=variable_selector,
        min_transactions=min_transactions_slider
    )
)

# Add base map
map_with_base = gv.tile_sources.CartoLight() * dmap

# Create layout
print("Creating layout...")

controls = pn.Column(
    "## Controls",
    variable_selector,
    min_transactions_slider,
    "---",
    pn.bind(get_stats, variable=variable_selector, min_transactions=min_transactions_slider),
    width=350
)

map_panel = pn.Column(
    "## Community Statistics Map",
    pn.pane.Markdown("""
    This map shows aggregated statistics for each community across Seattle.
    
    Each hexagon represents an H3 location within a community, colored by the selected variable.
    Communities are groups of H3 cells with similar characteristics.
    
    Hover over hexagons to see detailed community information.
    """),
    map_with_base,
    width=950
)

# Create dashboard
dashboard = pn.template.FastListTemplate(
    title="Community Aggregated Statistics - Seattle Area",
    sidebar=[controls],
    main=[map_panel],
    accent_base_color="#2F4F4F",
    header_background="#2F4F4F"
)

print("\n" + "="*80)
print("Starting Panel server...")
print("="*80)
print("\nMap Features:")
print("  - H3 Hexagon heatmap showing community statistics")
print("  - Interactive variable selection")
print("  - Filter by minimum transactions")
print("  - Hover tooltips with detailed information")
print("\nVariables Available:")
for var, config in VARIABLE_CONFIGS.items():
    print(f"  - {config['label']}")
print("\n" + "="*80)

# Serve the dashboard
dashboard.show(port=5008, threaded=False)
