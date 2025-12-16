
import panel as pn
import holoviews as hv
import geoviews as gv
import pandas as pd
import numpy as np
from bokeh.models import NumeralTickFormatter, HoverTool
import cartopy.crs as ccrs # Required for the CRS fix

pn.extension()
hv.extension('bokeh')

# =========================================================
# 1. CONFIGURATION
# =========================================================


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
# 2. HELPER FUNCTIONS
# =========================================================
    

def point_map(x_range, y_range, date_range, variable, communities, data):
    # --- A. Setup & Config ---
    config = COLUMN_CONFIGS.get(variable, {'cmap': 'Viridis', 'format': '0,0', 'label': variable})

    def colorbar_hook(plot, element):
        # Access the underlying Bokeh Figure
        fig = plot.state
        if fig.right:
            for item in fig.right:
                if type(item).__name__ == 'ColorBar':                
                    # 1. Update Title
                    item.title = config['label']
                    
                    # 2. Update Formatter
                    # We modify the property of the EXISTING formatter.
                    # This is much more reliable than replacing the object.
                    from bokeh.models import NumeralTickFormatter
                    if hasattr(item.formatter, 'format'):
                        item.formatter.format = config['format']
                    else:
                        item.formatter = NumeralTickFormatter(format=config['format'])

    is_continuous = config.get('continuous', True)
    # Force Numeric conversion
    if is_continuous:
        data[variable] = pd.to_numeric(data[variable], errors='coerce')

    if x_range is None or y_range is None:
        x_range = (data['lng'].min(), data['lng'].max())
        y_range = (data['lat'].min(), data['lat'].max())
    
    # --- B. Filtering ---
    start_ts = pd.Timestamp(date_range[0])
    end_ts = pd.Timestamp(date_range[1])
    mask = ((data['lng'] >= x_range[0]) & (data['lng'] <= x_range[1])
            & (data['lat'] >= y_range[0]) & (data['lat'] <= y_range[1])
            & (data['sale_date'] >= start_ts) & (data['sale_date'] <= end_ts))
    
    if communities:
        mask &= (data['community'].astype(str).isin(communities))
    df_filtered = data.loc[mask].copy()

    # --- D. Limits ---
    vmin, vmax = None, None
    if is_continuous and not df_filtered.empty:
        vals = df_filtered[variable].dropna()
        if len(vals) > 0:
            vmin, vmax = np.percentile(vals, 10), np.percentile(vals, 90)
            if config.get('center_zero', False):
                limit = max(abs(vmin), abs(vmax))
                vmin, vmax = -limit, limit

    # We use a simple Dimension with the label for the ColorBar mapping
    value_dim = hv.Dimension(variable, label=config['label'])
    # Build vdims: Ensure we strictly pass string names for extra columns
    # to avoid any dimension name confusion in the tooltip
    all_vdims = [value_dim, 'sale_date', 'community']
    for col_name in COLUMN_CONFIGS.keys():
        if col_name not in [variable, 'community']:
            all_vdims.append(col_name)

    # Build Tooltips
    point_tooltips = [('Date', '@sale_date{%F}')]
    for col_name, col_cfg in COLUMN_CONFIGS.items():
        # If it's the active variable, we refer to it by its variable name
        ref = variable if col_name == variable else col_name
        point_tooltips.append((col_cfg['label'], f"@{ref}{{{col_cfg['format']}}}"))
        
    hover = HoverTool(tooltips=point_tooltips, formatters={'@sale_date': 'datetime'}, mode='mouse')

    active_layer = gv.Points(
        df_filtered, 
        kdims=['lng', 'lat'], 
        vdims=all_vdims, 
        crs=ccrs.PlateCarree()
    ).opts(
        color=value_dim, 
        cmap=config['cmap'],
        clim=(vmin, vmax),
        colorbar = is_continuous,
        show_legend = not is_continuous,
        colorbar_opts={'formatter': NumeralTickFormatter(format=config.get('format','0,0')),
                       'title': config['label']},
        hooks=[colorbar_hook], # Pass Hook
        size=6,
        tools=[hover],
        width=700,
        height=500,
        xaxis=None,
        yaxis=None,
        title=f"({len(df_filtered)} points) {config['label']}"
    )
    
    return active_layer