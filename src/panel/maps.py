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
import cartopy.crs as ccrs

pn.extension()
hv.extension('bokeh')

# =========================================================
# 1. SHARED CONFIGURATION
# =========================================================

ZOOM_LEVELS = {
    'Low (7)': {'alpha': 0.7,  'col': 'h3_07'}, 
    'Medium (8)': {'alpha': 0.7, 'col': 'h3_08'}, 
    'Fine (10)':   {'alpha': 0.7, 'col': 'h3_10'}
}

COLUMN_CONFIGS = {
    'pct_error': {'cmap': 'Viridis', 'format': '0.0', 'label': 'Error', 'center_zero': True},
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
# 2. SHARED HELPER FUNCTIONS
# =========================================================


def prepare_data_and_config(data, variable, x_range, y_range, date_range):
    """
    Handles the shared logic for:
    1. Looking up Config
    2. Filtering Data (Ranges & Date)
    3. Calculating Color Limits (vmin/vmax)
    4. Creating the Hook
    """
    # 1. Config
    config = COLUMN_CONFIGS.get(variable, {'cmap': 'Viridis', 'format': '0,0', 'label': variable})
    is_continuous = config.get('continuous', True)

    # 2. Filtering
    if x_range is None or y_range is None:
        x_range = (data['lng'].min(), data['lng'].max())
        y_range = (data['lat'].min(), data['lat'].max())
        
    start_ts = pd.Timestamp(date_range[0])
    end_ts = pd.Timestamp(date_range[1])
    
    mask = ((data['lng'] >= x_range[0]) & (data['lng'] <= x_range[1])
            & (data['lat'] >= y_range[0]) & (data['lat'] <= y_range[1])
            & (data['sale_date'] >= start_ts) & (data['sale_date'] <= end_ts))
    
    df_filtered = data.loc[mask].copy()

    # 3. Limits
    vmin, vmax = None, None
    if is_continuous and not df_filtered.empty:
        # Force numeric for calculation
        temp_series = pd.to_numeric(df_filtered[variable], errors='coerce')
        vals = temp_series.dropna()
        if len(vals) > 0:
            vmin, vmax = np.percentile(vals, 10), np.percentile(vals, 90)
            if config.get('center_zero', False):
                limit = max(abs(vmin), abs(vmax))
                vmin, vmax = -limit, limit

    return df_filtered, config, vmin, vmax, is_continuous


def get_most_frequent(x):
    """Helper for H3 aggregation."""
    try:
        mode = x.mode().iloc[0]
        return mode
    except:
        return np.nan

def add_hex_geometry(aggregated_df, h3_col):
    """Helper for H3 geometry."""
    geoms = [Polygon(h3.h3_to_geo_boundary(idx, geo_json=True)) for idx in aggregated_df.index]
    return gpd.GeoDataFrame(aggregated_df, geometry=geoms)

# =========================================================
# 3. MAP FUNCTIONS
# =========================================================

def point_map(x_range, y_range, date_range, variable, data):
    # --- Shared Prep ---
    df_filtered, config, vmin, vmax, is_continuous = prepare_data_and_config(
        data, variable, x_range, y_range, date_range
    )
    
    # Force Numeric conversion for plotting if continuous
    if is_continuous:
        df_filtered[variable] = pd.to_numeric(df_filtered[variable], errors='coerce')

    # --- Tooltips ---
    value_dim = hv.Dimension(variable, label=config['label'])
    
    # Dynamically build vdims from config keys to ensure tooltips work
    all_vdims = [value_dim, 'sale_date', 'community']
    for col_name in COLUMN_CONFIGS.keys():
        if col_name not in [variable, 'community']:
            all_vdims.append(col_name)

    point_tooltips = [('Date', '@sale_date{%F}')]
    for col_name, col_cfg in COLUMN_CONFIGS.items():
        ref = variable if col_name == variable else col_name
        point_tooltips.append((col_cfg['label'], f"@{ref}{{{col_cfg['format']}}}"))
        
    hover = HoverTool(tooltips=point_tooltips, formatters={'@sale_date': 'datetime'}, mode='mouse')
    

    def hook(plot, element):
        fig = plot.state
        if fig.right:
            for item in fig.right:
                if type(item).__name__ == 'ColorBar':                
                    item.title = config['label']
                    from bokeh.models import NumeralTickFormatter
                    if hasattr(item.formatter, 'format'):
                        item.formatter.format = format=config['format']
                    else:
                        item.formatter = NumeralTickFormatter(format=config['format'])
        return hook

    # --- Plotting ---
    return gv.Points(
        df_filtered, 
        kdims=['lng', 'lat'], 
        vdims=all_vdims, 
        crs=ccrs.PlateCarree()
    ).opts(
        color=value_dim, 
        cmap=config['cmap'],
        clim=(vmin, vmax),
        colorbar=is_continuous,
        show_legend=is_continuous,
        colorbar_opts={'formatter': NumeralTickFormatter(format=config.get('format','0,0')),
                       'title': config['label']},
        hooks=[hook],
        size=6,
        alpha = 0.7,
        tools=[hover],
        width=700,
        height=500,
        xaxis=None,
        yaxis=None,
        title=f"({len(df_filtered)} points) {config['label']}"
    )

def hex_map(x_range, y_range, date_range, variable, zoom_level, data):
    # --- Shared Prep ---
    df_filtered, config, vmin, vmax, is_continuous = prepare_data_and_config(
        data, variable, x_range, y_range, date_range
    )

    if df_filtered.empty:
        return gv.Polygons([], crs=ccrs.PlateCarree()).opts(
            title="No Data in Range", width=1000, height=500, xaxis=None, yaxis=None
        )

    # --- Aggregation ---
    zoom = ZOOM_LEVELS.get(zoom_level, {'alpha': 0.5,  'col': 'h3_07'})
    target_col = zoom['col']

    if not is_continuous:
        agg_df = df_filtered.groupby(target_col).agg(
            val=(variable, get_most_frequent),
            count=('community', 'count')
        )
    else:
        # Ensure numeric for aggregation
        df_filtered[variable] = pd.to_numeric(df_filtered[variable], errors='coerce')
        agg_df = df_filtered.groupby(target_col).agg(
            val=(variable, 'mean'),
            count=('community', 'count')
        )

    gdf = add_hex_geometry(agg_df, target_col)

    # --- Tooltips ---
    hex_tooltips = [
        (config['label'], '@val{' + config['format'] + '}'),
        ('Count', '@count'),
    ]
    value_dim = hv.Dimension('val', label=config['label'])

    def hook(plot, element):
        fig = plot.state
        if fig.right:
            for item in fig.right:
                if type(item).__name__ == 'ColorBar':                
                    item.title = config['label']
                    from bokeh.models import NumeralTickFormatter
                    if hasattr(item.formatter, 'format'):
                        item.formatter.format = format=config['format']
                    else:
                        item.formatter = NumeralTickFormatter(format=config['format'])
        return hook


    # --- Plotting ---
    # Construct opts dictionary to handle conditional clim if needed, 
    # matching your original logic structure
    opts = dict(
        line_color='white', 
        line_width=0.5,
        cmap=config['cmap'],
        colorbar=is_continuous,
        show_legend=is_continuous,
        colorbar_opts={'formatter': NumeralTickFormatter(format=config.get('format','0,0')),
                       'title': config['label']},
        width=700,
        alpha=zoom['alpha'],
        height=500,
        xaxis=None,
        yaxis=None,
        hooks=[hook],
        tools=[HoverTool(tooltips=hex_tooltips)],
        title=f"({len(df_filtered)} points) x_range: {x_range} {config['label']}"
    )

    if vmin is not None:
         opts['clim'] = (vmin, vmax)

    return gv.Polygons(
        gdf, 
        vdims=[value_dim, 'count'],
        crs=ccrs.PlateCarree() 
    ).opts(**opts)