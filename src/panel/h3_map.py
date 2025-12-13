import panel as pn
import holoviews as hv
import geoviews as gv
import pandas as pd
import numpy as np
import h3
import geopandas as gpd
from shapely.geometry import Polygon
from holoviews.streams import RangeXY
from bokeh.models import NumeralTickFormatter, HoverTool, LinearColorMapper, ColorBar
from bokeh.transform import factor_cmap
import cartopy.crs as ccrs # Required for the CRS fix

pn.extension()
hv.extension('bokeh')

# =========================================================
# 1. CONFIGURATION
# =========================================================
ZOOM_LEVELS = {
    'Low (7)': {'alpha': 0.6,  'col': 'h3_07'}, 
    'Medium (8)': {'alpha': 0.8, 'col': 'h3_08'}, 
    'Fine (10)':   {'alpha': 0.9, 'col': 'h3_10'}
}

COLUMN_CONFIGS = {
    'pct_error': {'cmap': 'Viridis', 'format': '0.0%', 'label': 'Error', 'center_zero': True},
    'predicted_price': {'cmap': 'Viridis', 'format': '$0a', 'label': 'Pred. Price', 'center_zero': False},
    'sale_price': {'cmap': 'Viridis', 'format': '$0a', 'label': 'Sale Price', 'center_zero': False},
    'sqft': {'cmap': 'Viridis', 'format': '0,0', 'label': 'Size (SqFt)', 'center_zero': False},
    'community': {'cmap': 'glasbey', 'format': '0', 'continuous':False,'label': 'Comm ID', 'center_zero': False},
    'cls_property': {'cmap': 'Turbo', 'format': '0.00', 'label': 'CLS Property', 'center_zero': False},
    'cls_community': {'cmap': 'Turbo', 'format': '0.00', 'label': 'CLS Community', 'center_zero': False},
    'cls_week': {'cmap': 'Turbo', 'format': '0.00', 'label': 'CLS Week', 'center_zero': False},
    'cls_year': {'cmap': 'Turbo', 'format': '0.00', 'label': 'CLS Year', 'center_zero': False}


}

# =========================================================
# 2. HELPER FUNCTIONS
# =========================================================
def add_hex_geometry(aggregated_df, h3_col):
    """Converts H3 index strings into Shapely Polygons."""
    geoms = [Polygon(h3.h3_to_geo_boundary(idx, geo_json=True)) for idx in aggregated_df.index]
    return gpd.GeoDataFrame(aggregated_df, geometry=geoms)

def get_most_frequent(x):
    """Helper to get mode of categorical data safely."""
    try:
        mode = x.mode().iloc[0]
        print(f'Mode = {mode}')
        return mode
    except:
        return np.nan
def get_dynamic_map(x_range, y_range, date_range, variable, zoom_level, data):
    """
    Main Map Logic: Filters, Aggregates, and Styles based on config.
    """
    # --- A. Setup & Config ---
    config = COLUMN_CONFIGS.get(variable, {'cmap': 'Viridis', 'format': '0,0', 'label': variable})

    # Handle initial load
    if x_range is None or y_range is None:
        x_range = (data['lng'].min(), data['lng'].max())
        y_range = (data['lat'].min(), data['lat'].max())
    
    # --- B. Filtering ---
    start_ts = pd.Timestamp(date_range[0])
    end_ts = pd.Timestamp(date_range[1])

    mask = ((data['lng'] >= x_range[0]) & (data['lng'] <= x_range[1])
            & (data['lat'] >= y_range[0]) & (data['lat'] <= y_range[1])
            & (data['sale_date'] >= start_ts) & (data['sale_date'] <= end_ts))
    df_filtered = data.loc[mask].copy()

    if df_filtered.empty:
        return gv.Polygons([], crs=ccrs.PlateCarree()).opts(
            title="No Data in Range", width=1000, height=500, xaxis=None, yaxis=None
        )

    # --- C. Determine Granularity ---
    zoom = ZOOM_LEVELS.get(zoom_level, {'alpha': 0.5,  'col': 'h3_07'})
    target_col = zoom['col']

    # --- D. Color Limits & Aggregation ---
    vmin, vmax = None, None
    is_continuous = config.get('continuous', True)

    # 1. Calculate Limits (Only for continuous)
    if is_continuous:
        # Force numeric to ensure formatting works
        df_filtered[variable] = pd.to_numeric(df_filtered[variable], errors='coerce')
        vals = df_filtered[variable].dropna()
        if len(vals) > 0:
            vmin, vmax = np.percentile(vals, 10), np.percentile(vals, 90)
            if config.get('center_zero', False):
                limit = max(abs(vmin), abs(vmax))
                vmin, vmax = -limit, limit

    # 2. Aggregation Logic (Moved OUTSIDE the vmin check so it runs for categorical too)
    if not is_continuous: # Categorical (e.g., Community)
        agg_df = df_filtered.groupby(target_col).agg(
            val=(variable, get_most_frequent),
            count=('community', 'count')
        )
    else: # Continuous
        agg_df = df_filtered.groupby(target_col).agg(
            val=(variable, 'mean'),
            count=('community', 'count')
        )

    # --- E. THE HOOK (Fixes the ColorBar) ---
    def colorbar_hook(plot, element):
        """
        Forces the Bokeh ColorBar to update its title and formatter.
        """
        # Access the underlying Bokeh plot
        fig = plot.state
        # Check if there are side panels (where the ColorBar lives)
        if fig.right:
            for item in fig.right:
                if type(item).__name__ == 'ColorBar':
                    from bokeh.models import NumeralTickFormatter
                    
                    # Force Title Update
                    item.title = config['label']
                    
                    # Force Formatter Update
                    # Instead of creating a new object, we update the existing one 
                    # if possible, or create new if needed.
                    if hasattr(item.formatter, 'format'):
                        item.formatter.format = config['format']
                    else:
                        item.formatter = NumeralTickFormatter(format=config['format'])

    # --- F. Styling ---
    style_opts = dict(
        cmap = config['cmap'],
        colorbar = is_continuous,
        show_legend = not is_continuous,
        # We still pass these, but the Hook ensures they actually apply
        colorbar_opts={'formatter': NumeralTickFormatter(format=config.get('format','0,0')),
                       'title': config['label']},
        width=700,
        alpha = zoom['alpha'],
        height=500,
        xaxis=None,
        yaxis=None,
        # Add the hook here
        hooks=[colorbar_hook], 
        title=f"({len(df_filtered)} points) x_range: {x_range} {config['label']}"
    )

    # Only apply color limits if we calculated them
    if vmin is not None:
        style_opts['clim'] = (vmin, vmax)

    # --- G. Construction ---
    gdf = add_hex_geometry(agg_df, target_col)

    # Tooltips
    hex_tooltips = [
        (config['label'], '@val{' + config['format'] + '}'),
        ('Count', '@count'),
    ]

    # Create Dimension with Label
    value_dim = hv.Dimension('val', label=config['label'])

    return gv.Polygons(
        gdf, 
        vdims = [value_dim, 'count'],
        crs = ccrs.PlateCarree() 
    ).opts(
        line_color='white', 
        line_width=0.5,
        **style_opts
    ).opts(tools=[HoverTool(tooltips=hex_tooltips)])