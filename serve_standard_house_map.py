"""
Interactive Panel map showing standard house values across H3 hexagons
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

print("Loading standard house predictions...")

# Load the predictions
df = pd.read_csv('outputs/standard_house_predictions_by_h3.csv')
print(f"Loaded {len(df)} H3 hexagons with predictions")

# Calculate price per sqft
df['price_per_sqft'] = df['predicted_price'] / df['sqft']
df['ci_width'] = df['price_upper_95'] - df['price_lower_95']

# Configuration for different variables
VARIABLE_CONFIGS = {
    'predicted_price': {
        'cmap': 'viridis',
        'format': '$0,0',
        'label': 'Predicted Price',
        'center_zero': False
    },
    'price_per_sqft': {
        'cmap': 'plasma',
        'format': '$0,0',
        'label': 'Price per Sqft',
        'center_zero': False
    },
    'ci_width': {
        'cmap': 'coolwarm',
        'format': '$0,0',
        'label': '95% CI Width',
        'center_zero': False
    },
    'cls_attn_community': {
        'cmap': 'RdYlGn',
        'format': '0.000',
        'label': 'Community Attention',
        'center_zero': False
    },
    'cls_attn_property': {
        'cmap': 'RdYlGn',
        'format': '0.000',
        'label': 'Property Attention',
        'center_zero': False
    },
    'cls_attn_time': {
        'cmap': 'RdYlGn',
        'format': '0.000',
        'label': 'Time Attention',
        'center_zero': False
    },
    'cls_attn_market': {
        'cmap': 'RdYlGn',
        'format': '0.000',
        'label': 'Market Attention',
        'center_zero': False
    }
}


def create_hex_polygons(df_with_h3):
    """Convert H3 indices to polygon data for geoviews"""
    
    polygons_data = []
    
    for idx, row in df_with_h3.iterrows():
        h3_idx = row['h3_07']
        
        # Get boundary - h3 v4 API
        try:
            boundary = h3.cell_to_boundary(h3_idx)
        except:
            # Try old API
            boundary = h3.h3_to_geo_boundary(h3_idx, geo_json=True)
        
        # Extract lons and lats
        if isinstance(boundary[0], tuple):
            # New API returns (lat, lng) tuples
            lats = [coord[0] for coord in boundary]
            lons = [coord[1] for coord in boundary]
        else:
            # Old API returns [lng, lat] lists
            lons = [coord[0] for coord in boundary]
            lats = [coord[1] for coord in boundary]
        
        # Create polygon data
        poly_data = {
            'Longitude': lons,
            'Latitude': lats,
            'value': row['value'],
            'h3_07': h3_idx,
            'community': row['community'],
            'predicted_price': row['predicted_price'],
            'price_per_sqft': row['price_per_sqft'],
            'price_lower_95': row['price_lower_95'],
            'price_upper_95': row['price_upper_95']
        }
        
        polygons_data.append(poly_data)
    
    return polygons_data


def create_hex_map(variable, x_range=None, y_range=None):
    """Create hexagon map for selected variable"""
    
    config = VARIABLE_CONFIGS.get(variable, VARIABLE_CONFIGS['predicted_price'])
    
    # Handle initial load
    if x_range is None or y_range is None:
        x_range = (df['lng'].min(), df['lng'].max())
        y_range = (df['lat'].min(), df['lat'].max())
    
    # Filter by viewport
    mask = ((df['lng'] >= x_range[0]) & (df['lng'] <= x_range[1]) &
            (df['lat'] >= y_range[0]) & (df['lat'] <= y_range[1]))
    
    df_filtered = df.loc[mask].copy()
    
    if df_filtered.empty:
        return gv.Polygons([], crs=ccrs.PlateCarree()).opts(
            title="No Data in Range",
            width=800,
            height=600,
            xaxis=None,
            yaxis=None
        )
    
    # Calculate color limits
    vals = df_filtered[variable].dropna()
    if len(vals) > 0:
        vmin, vmax = np.percentile(vals, 5), np.percentile(vals, 95)
        if config.get('center_zero', False):
            limit = max(abs(vmin), abs(vmax))
            vmin, vmax = -limit, limit
    else:
        vmin, vmax = None, None
    
    # Prepare data for plotting
    df_filtered['value'] = df_filtered[variable]
    
    # Create polygon data
    polygons_data = create_hex_polygons(df_filtered)
    
    # Tooltips
    hex_tooltips = [
        ('H3 Index', '@h3_07'),
        ('Community', '@community'),
        (config['label'], '@value{' + config['format'] + '}'),
        ('Predicted Price', '$@predicted_price{0,0}'),
        ('Price/Sqft', '$@price_per_sqft{0,0}'),
        ('95% CI', '[$@price_lower_95{0,0}, $@price_upper_95{0,0}]')
    ]
    
    # Create dimension with label
    value_dim = hv.Dimension('value', label=config['label'])
    
    # Create polygons using geoviews
    polygons = gv.Polygons(
        polygons_data,
        vdims=[value_dim, 'h3_07', 'community', 'predicted_price', 
               'price_per_sqft', 'price_lower_95', 'price_upper_95'],
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
        width=800,
        height=600,
        xaxis=None,
        yaxis=None,
        tools=[HoverTool(tooltips=hex_tooltips)],
        title=f"{config['label']} - Standard House (3 bed, 1500 sqft, 1.5 bath)"
    )
    
    return polygons


print("Creating widgets...")

# Variable selector
variable_selector = pn.widgets.Select(
    name='Variable to Display',
    options=list(VARIABLE_CONFIGS.keys()),
    value='predicted_price',
    width=300
)

# Statistics panel
def get_stats(variable):
    """Generate statistics for selected variable"""
    
    config = VARIABLE_CONFIGS.get(variable, VARIABLE_CONFIGS['predicted_price'])
    
    # Format values based on whether it's currency or not
    if '$' in config['format']:
        stats = f"""
    ### {config['label']}
    
    **Statistics:**
    - Min: ${df[variable].min():,.0f}
    - Max: ${df[variable].max():,.0f}
    - Mean: ${df[variable].mean():,.0f}
    - Median: ${df[variable].median():,.0f}
    - Std Dev: ${df[variable].std():,.0f}
    
    **Standard House:**
    - Bedrooms: 3
    - Sqft: 1,500
    - Bathrooms: 1.5
    - Lot: 5,000 sqft
    - Year Built: 1980
    
    **Coverage:**
    - H3 Hexagons: {len(df)}
    - Communities: {df['community'].nunique()}
    """
    else:
        stats = f"""
    ### {config['label']}
    
    **Statistics:**
    - Min: {df[variable].min():.3f}
    - Max: {df[variable].max():.3f}
    - Mean: {df[variable].mean():.3f}
    - Median: {df[variable].median():.3f}
    - Std Dev: {df[variable].std():.3f}
    
    **Standard House:**
    - Bedrooms: 3
    - Sqft: 1,500
    - Bathrooms: 1.5
    - Lot: 5,000 sqft
    - Year Built: 1980
    
    **Coverage:**
    - H3 Hexagons: {len(df)}
    - Communities: {df['community'].nunique()}
    """
    
    return pn.pane.Markdown(stats)


# Initial bounds
initial_x = (df['lng'].min(), df['lng'].max())
initial_y = (df['lat'].min(), df['lat'].max())

# Shared range stream for synchronized zooming
from holoviews.streams import RangeXY
common_range_stream = RangeXY(x_range=initial_x, y_range=initial_y)

print("Creating dynamic map...")

# Create dynamic map
dmap = gv.DynamicMap(
    pn.bind(create_hex_map, variable=variable_selector),
    streams=[common_range_stream]
)

# Add base map
map_with_base = gv.tile_sources.CartoLight() * dmap

# Create layout
print("Creating layout...")

controls = pn.Column(
    "## Controls",
    variable_selector,
    "---",
    pn.bind(get_stats, variable=variable_selector),
    width=350
)

map_panel = pn.Column(
    "## Standard House Value Map",
    pn.pane.Markdown("""
    This map shows predicted values for a **standard 3-bedroom, 1500 sqft house** 
    across all H3 hexagons in the Seattle area.
    
    Each hexagon represents a geographic area, colored by the selected variable.
    Hover over hexagons to see detailed information.
    """),
    map_with_base,
    width=850
)

# Create dashboard
dashboard = pn.template.FastListTemplate(
    title="Standard House Value Heatmap - Seattle Area",
    sidebar=[controls],
    main=[map_panel],
    accent_base_color="#2F4F4F",
    header_background="#2F4F4F"
)

print("\n" + "="*80)
print("Starting Panel server...")
print("="*80)
print("\nMap Features:")
print("  - H3 Hexagon heatmap showing standard house values")
print("  - Interactive variable selection")
print("  - Hover tooltips with detailed information")
print("  - Zoom and pan to explore different areas")
print("\nVariables Available:")
for var, config in VARIABLE_CONFIGS.items():
    print(f"  - {config['label']}")
print("\n" + "="*80)

# Serve the dashboard
dashboard.show(port=5007, threaded=False)
