import panel as pn
import holoviews as hv
import geoviews as gv
import pandas as pd
import numpy as np
import h3
import geopandas as gpd
from shapely.geometry import Polygon
from holoviews.streams import RangeXY
from bokeh.models import NumeralTickFormatter, HoverTool
import cartopy.crs as ccrs # Required for the CRS fix

pn.extension()
hv.extension('bokeh')

# =========================================================
# 1. CONFIGURATION
# =========================================================
ZOOM_LEVELS = {
    'coarse': {'width': 2.0,  'col': 'h3_07'}, 
    'medium': {'width': 0.5, 'col': 'h3_08'}, 
    'fine':   {'width': 0.2, 'col': 'h3_10'}                 
}

COLUMN_CONFIGS = {
    'pct_error': {'cmap': 'Viridis', 'format': '0.0%', 'label': 'Error', 'center_zero': True},
    'predicted_price': {'cmap': 'Viridis', 'format': '$0.0a', 'label': 'Pred. Price', 'center_zero': False},
    'sale_price': {'cmap': 'Inferno', 'format': '$0.0a', 'label': 'Sale Price', 'center_zero': False},
    'sqft': {'cmap': 'Viridis', 'format': '0,0', 'label': 'Size (SqFt)', 'center_zero': False},
    'community': {'cmap': 'Turbo', 'format': '0', 'label': 'Comm ID', 'center_zero': False}
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
        return x.mode().iloc[0]
    except:
        return np.nan

def get_dynamic_map(x_range, y_range, date_range, variable, data):
    """
    Main Map Logic: Filters, Aggregates, and Styles based on config.
    """

    # --- A. Setup & Config ---
    config = COLUMN_CONFIGS.get(variable, {'cmap': 'Viridis', 'format': '0,0', 'label': variable})

    # Handle initial load
    if x_range is None or y_range is None:
        x_range = (data['lng'].min(), data['lng'].max())
        y_range = (data['lat'].min(), data['lat'].max())
    view_width = x_range[1] - x_range[0]
    
    # --- B. Filtering (FIXED DATE LOGIC) ---
    # Fix: Convert slider 'date' objects to Pandas 'Timestamp' objects
    start_ts = pd.Timestamp(date_range[0])
    end_ts = pd.Timestamp(date_range[1])

    # mask = (
    #     (data['lng'] >= x_range[0]) & (data['lng'] <= x_range[1]) &
    #     (data['lat'] >= y_range[0]) & (data['lat'] <= y_range[1]) 
    #     & (data['sale_date'] >= start_ts) & (data['sale_date'] <= end_ts)
    # )
    # df_filtered = data.loc[mask].copy()
    df_filtered = data

    # if df_filtered.empty:
    #     # Fix: Must provide a valid CRS when returning empty polygons
    #     # We use PlateCarree (Lat/Lon)
    #     return gv.Polygons([], crs=ccrs.PlateCarree()).opts(
    #         title="No Data in Range", width=1000, height=500, xaxis=None, yaxis=None
    #     )

    # --- C. Determine Granularity ---
    # view_width.. 
    # if 1.36>2...
    if view_width > ZOOM_LEVELS['coarse']['width']:
        target_col = ZOOM_LEVELS['coarse']['col']
        mode = "Coarse Hex"
    #if > 0.05
    elif view_width > ZOOM_LEVELS['medium']['width']:
        target_col = ZOOM_LEVELS['medium']['col']
        mode = "Medium Hex"
    elif view_width > ZOOM_LEVELS['fine']['width']:
        target_col = ZOOM_LEVELS['fine']['col']
        mode = "Fine Hex"
    else:
        mode = "Points"

    # --- D. Color Limits (10th - 90th Percentile) ---
    vmin, vmax = None, None
    
    # Only calculate percentiles for non-categorical data
    if variable != 'community':
        vals = df_filtered[variable].dropna()
        if len(vals) > 0:
            vmin, vmax = np.percentile(vals, 10), np.percentile(vals, 90)
            
            # Logic for symmetric error scales
            if config.get('center_zero', False):
                limit = max(abs(vmin), abs(vmax))
                vmin, vmax = -limit, limit
    
    # --- E. Rendering ---
    
    # Common Style Options
    formatter = NumeralTickFormatter(format=config['format'])
    style_opts = dict(
        cmap=config['cmap'],
        colorbar=True,
        colorbar_opts={'formatter': formatter, 'title': config['label']},
        width=1000,
        height=500,
        xaxis=None, # Remove Axis
        yaxis=None, # Remove Axis
        tools=['hover'],
        title=f"({len(df_filtered)} points) x_range: {x_range} {mode}: {config['label']}, View width: {view_width:.2f}°"
    )
    
    if vmin is not None:
        style_opts['clim'] = (vmin, vmax)

    # --- MODE 1: POINTS ---
    if mode == "Points":
        return gv.Points(
            df_filtered, 
            kdims=['lng', 'lat'], 
            vdims=[variable, 'sale_date', 'community']
        ).opts(
            color=variable,
            size=6,
            **style_opts
        )

    # --- MODE 2: HEXAGONS ---
    else:
        # Aggregation Strategy
        if variable == 'community':
            # For categorical, we find the "Mode"
            agg_df = df_filtered.groupby(target_col).agg(
                val=(variable, get_most_frequent),
                count=('community', 'count')
            )
        else:
            # For continuous, we calculate Mean
            agg_df = df_filtered.groupby(target_col).agg(
                val=(variable, 'mean'),
                count=('community', 'count')
            )
        
        gdf = add_hex_geometry(agg_df, target_col)
        print(f'Length of geodataframe, {len(gdf)}')
        # Override tooltips for Hexagons
        hex_tooltips = [
            (config['label'], '@val{' + config['format'] + '}'),
            ('Count', '@count'),
        ]
        return gv.Polygons(
            gdf, 
            vdims=['val', 'count'],
            crs = ccrs.PlateCarree()  # Specify the source CRS,
        ).opts(
            color='val',
            line_color='white', 
            line_width=0.5,
            alpha=0.7,
            **style_opts
        ).opts(tools=[HoverTool(tooltips=hex_tooltips)])

# =========================================================
# 3. INITIALIZATION
# =========================================================
