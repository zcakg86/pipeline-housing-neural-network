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
import colorcet as cc
from itertools import cycle
import traceback

pn.extension()
hv.extension('bokeh')

# =========================================================
# 1. CONFIGURATION & UTILS
# =========================================================

class MapConfig:
    """Centralized configuration for map visualization."""
    
    ZOOM_LEVELS = {
        'Low (7)': {'alpha': 0.6,  'col': 'h3_07'}, 
        'Medium (8)': {'alpha': 0.8, 'col': 'h3_08'}, 
        'Fine (10)':   {'alpha': 0.9, 'col': 'h3_10'}
    }

    COLUMNS = {
        'pct_error': {'cmap': 'Viridis', 'format': '0.0%', 'label': 'Percentage Error', 'center_zero': True},
        'predicted_price': {'cmap': 'Viridis', 'format': '$0a', 'label': 'Pred. Price', 'center_zero': False},
        'sale_price': {'cmap': 'Viridis', 'format': '$0a', 'label': 'Sale Price', 'center_zero': False},
        'sqft': {'cmap': 'Viridis', 'format': '0,0', 'label': 'Size (SqFt)', 'center_zero': False},
        'community': {'cmap': 'glasbey', 'format': '0', 'continuous': False, 'label': 'Comm ID', 'center_zero': False},
        'cls_property': {'cmap': 'Turbo', 'format': '0.00', 'label': 'CLS Property', 'center_zero': False},
        'cls_community': {'cmap': 'Turbo', 'format': '0.00', 'label': 'CLS Community', 'center_zero': False},
        'cls_week': {'cmap': 'Turbo', 'format': '0.00', 'label': 'CLS Week', 'center_zero': False},
        'cls_year': {'cmap': 'Turbo', 'format': '0.00', 'label': 'CLS Year', 'center_zero': False}
    }

def get_most_frequent(x):
    """Helper to get mode of categorical data safely."""
    try:
        return x.mode().iloc[0]
    except:
        return np.nan

def add_hex_geometry(aggregated_df, h3_col):
    """Converts H3 index strings into Shapely Polygons."""
    geoms = [Polygon(h3.h3_to_geo_boundary(idx, geo_json=True)) for idx in aggregated_df.index]
    return gpd.GeoDataFrame(aggregated_df, geometry=geoms)


# =========================================================
# 2. VISUALIZER CLASS
# =========================================================

class RealEstateVisualizer:
    """
    Manages the creation of Point and Hex maps with shared state and filtering logic.
    """
    def __init__(self, dataframe):
        self.df = dataframe.copy()
        
        # Ensure date format
        if not pd.api.types.is_datetime64_any_dtype(self.df['sale_date']):
            self.df['sale_date'] = pd.to_datetime(self.df['sale_date'])

        # --- Generate Fixed Color Map for Communities ---
        # 1. Get all unique communities as strings
        all_communities = sorted(dataframe['community'].astype(str).unique())
        
        # 2. Get the Glasbey palette
        palette = cc.glasbey 
        
        # 3. Create a dictionary: {'101': '#hexcolor', '102': '#hexcolor'}
        self.community_color_map = {
            comm: color for comm, color in zip(all_communities, cycle(palette))
        }

        # --- Initialize Widgets ---
        self._init_widgets()
        
        # --- Shared Stream ---
        self.initial_x = (self.df['lng'].min(), self.df['lng'].max())
        self.initial_y = (self.df['lat'].min(), self.df['lat'].max())
        self.range_stream = RangeXY(x_range=self.initial_x, y_range=self.initial_y)

    def _init_widgets(self):
        """Create and configure all dashboard widgets."""
        min_date = self.df['sale_date'].min().date()
        max_date = self.df['sale_date'].max().date()

        self.w_date = pn.widgets.DateRangeSlider(
            name='Date Range', start=min_date, end=max_date, value=(min_date, max_date)
        )
        self.w_var = pn.widgets.Select(
            name='Variable', options=list(MapConfig.COLUMNS.keys()), value='pct_error'
        )
        self.w_zoom = pn.widgets.Select(
            name='H3 Index', options=list(MapConfig.ZOOM_LEVELS.keys()), value='Low (7)'
        )
        unique_communities = sorted(list(self.df['community'].astype(str).unique()))
        self.w_comm = pn.widgets.MultiChoice(
            name='Filter Communities (Leave empty for All)',
            options=unique_communities,
            value=[], 
            solid=False
        )

    def _create_colorbar_hook(self, label, fmt):
        """
        Factory function that returns a Bokeh hook to fix ColorBar formatting.
        """
        def hook(plot, element):
            fig = plot.state
            if fig.right:
                for item in fig.right:
                    if type(item).__name__ == 'ColorBar':
                        item.title = label
                        # Simply replace the formatter (Avoids hasattr read-only error)
                        from bokeh.models import NumeralTickFormatter
                        item.formatter = NumeralTickFormatter(format=fmt)
        return hook
    
    def _filter_data(self, x_range, y_range, date_range, communities):
        """Shared filtering logic."""
        if x_range is None or y_range is None:
            x_range, y_range = self.initial_x, self.initial_y

        start_ts, end_ts = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        mask = (self.df['sale_date'] >= start_ts) & (self.df['sale_date'] <= end_ts)
        mask &= (self.df['lng'] >= x_range[0]) & (self.df['lng'] <= x_range[1])
        mask &= (self.df['lat'] >= y_range[0]) & (self.df['lat'] <= y_range[1])

        if communities:
            mask &= (self.df['community'].astype(str).isin(communities))

        return self.df.loc[mask].copy()

    def _get_vlims(self, df, variable, config):
        """Calculates color limits based on percentiles."""
        vmin, vmax = None, None
        is_continuous = config.get('continuous', True) 
        
        if is_continuous and not df.empty:
            vals = pd.to_numeric(df[variable], errors='coerce').dropna()
            if len(vals) > 0:
                vmin, vmax = np.percentile(vals, 10), np.percentile(vals, 90)
                if config.get('center_zero', False):
                    limit = max(abs(vmin), abs(vmax))
                    vmin, vmax = -limit, limit
        
        return vmin, vmax, is_continuous

    # =========================================================
    # RENDERERS
    # =========================================================
    def render_points(self, x_range, y_range, date_range, variable, communities):
        """Callback for Point Map."""
        try:
            df_filtered = self._filter_data(x_range, y_range, date_range, communities)
            config = MapConfig.COLUMNS.get(variable)
            
            # 1. Calc Limits & Check Continuity
            vmin, vmax, is_continuous = self._get_vlims(df_filtered, variable, config)
            
            # 2. Handle Categorical Data Types
            if not is_continuous:
                df_filtered[variable] = df_filtered[variable].astype(str)
                if variable == 'community':
                    target_cmap = self.community_color_map
                else:
                    target_cmap = config['cmap']
                
                # FIX TOOLTIP: No formatting for strings (prevents ???)
                tooltip_format_str = f"@{variable}" 
            else:
                target_cmap = config['cmap']
                tooltip_format_str = f"@{variable}{{{config['format']}}}"

            # 3. Build Tooltips
            point_tooltips = [('Date', '@sale_date{%F}')]
            for col, cfg in MapConfig.COLUMNS.items():
                if col == variable:
                    # Use our logic determined above
                     point_tooltips.append((cfg['label'], tooltip_format_str))
                else:
                    # Standard behavior for others
                    point_tooltips.append((cfg['label'], f"@{col}{{{cfg['format']}}}"))
            
            # 4. Construct Options Dynamically
            hook = self._create_colorbar_hook(config['label'], config['format'])

            opts_dict = dict(
                color=variable, 
                cmap=target_cmap,
                colorbar=is_continuous,
                show_legend=not is_continuous,
                hooks=[hook],
                size=6,
                tools=[HoverTool(tooltips=point_tooltips, formatters={'@sale_date': 'datetime'})],
                width=600, height=500, xaxis=None, yaxis=None,
                title=f"Points: {len(df_filtered)}"
            )

            # 5. ONLY add clim if we are continuous
            if is_continuous and vmin is not None and vmax is not None:
                opts_dict['clim'] = (vmin, vmax)

            # 6. Create Plot
            # FIX CRASH: Add label=variable to force HoloViews to rebuild plot when variable changes
            points = gv.Points(
                df_filtered, 
                kdims=['lng', 'lat'], 
                vdims=[variable, 'sale_date', 'community'] + [c for c in MapConfig.COLUMNS if c not in [variable, 'community']],
                crs=ccrs.PlateCarree(),
                label=variable # <--- CRITICAL FIX
            ).opts(**opts_dict)
            
            return points

        except Exception as e:
            traceback.print_exc()
            return gv.Points([], crs=ccrs.PlateCarree()).opts(title=f"Error: {str(e)}")
        
    def render_hex(self, x_range, y_range, date_range, variable, zoom_level, communities):
        """Callback for Hex Map."""
        try:
            df_filtered = self._filter_data(x_range, y_range, date_range, communities)
            config = MapConfig.COLUMNS.get(variable)
            
            if df_filtered.empty:
                return gv.Polygons([], crs=ccrs.PlateCarree()).opts(title="No Data", width=600, height=500)

            zoom = MapConfig.ZOOM_LEVELS[zoom_level]
            target_col = zoom['col']
            
            is_continuous = config.get('continuous', True)
            
            # --- 1. Handle Categorical Data Types ---
            if not is_continuous:
                df_filtered[variable] = df_filtered[variable].astype(str)
            
            # --- 2. Aggregation ---
            if is_continuous:
                df_filtered[variable] = pd.to_numeric(df_filtered[variable], errors='coerce')
                agg_df = df_filtered.groupby(target_col).agg(
                    val=(variable, lambda x: x.abs().mean()), 
                    count=('community', 'count')
                )
            else:
                agg_df = df_filtered.groupby(target_col).agg(
                    val=(variable, get_most_frequent), 
                    count=('community', 'count')
                )

            # --- 3. Rename 'val' to actual variable name ---
            agg_df = agg_df.rename(columns={'val': variable})

            # Geometry
            gdf = add_hex_geometry(agg_df, target_col)
            
            # --- 4. Limits & Color Map ---
            vmin, vmax = None, None
            target_cmap = config['cmap']

            if is_continuous:
                vals = gdf[variable].dropna()
                if len(vals) > 0:
                    vmin, vmax = np.percentile(vals, 10), np.percentile(vals, 90)
                    if config.get('center_zero', False):
                        limit = max(abs(vmin), abs(vmax))
                        vmin, vmax = -limit, limit
                
                # Use format string for continuous
                tooltip_val = f"@{variable}{{{config['format']}}}"
            else:
                if variable == 'community':
                    target_cmap = self.community_color_map
                
                # FIX TOOLTIP: No format string for categorical
                tooltip_val = f"@{variable}"

            # --- 5. Tooltips & Options ---
            hook = self._create_colorbar_hook(config['label'], config['format'])
            
            hex_tooltips = [
                (config['label'], tooltip_val), 
                ('Count', '@count')
            ]

            opts_dict = dict(
                line_color='white', line_width=0.5,
                cmap=target_cmap,
                alpha=zoom['alpha'],
                colorbar=is_continuous,
                show_legend=not is_continuous,
                hooks=[hook],
                tools=[HoverTool(tooltips=hex_tooltips)],
                width=600, height=500, xaxis=None, yaxis=None,
                title=f"H3 Aggregation ({len(agg_df)} hexes)"
            )

            if vmin is not None and vmax is not None:
                opts_dict['clim'] = (vmin, vmax)

            # --- 6. Create Polygons ---
            # FIX CRASH: Add label=variable
            polys = gv.Polygons(
                gdf, 
                vdims=[variable, 'count'], 
                crs=ccrs.PlateCarree(),
                label=variable # <--- CRITICAL FIX
            ).opts(**opts_dict)
            
            return polys

        except Exception as e:
            traceback.print_exc()
            return gv.Polygons([], crs=ccrs.PlateCarree()).opts(title=f"Error: {str(e)}")
        
    def view(self):
        """Constructs the Panel Layout."""
        dmap_points = gv.DynamicMap(
            pn.bind(self.render_points, 
                    date_range=self.w_date, 
                    variable=self.w_var, 
                    communities=self.w_comm),
            streams=[self.range_stream]
        )

        dmap_hex = gv.DynamicMap(
            pn.bind(self.render_hex, 
                    date_range=self.w_date, 
                    variable=self.w_var, 
                    zoom_level=self.w_zoom,
                    communities=self.w_comm),
            streams=[self.range_stream]
        )

        tiles = gv.tile_sources.CartoLight()
        
        map_row = pn.Row(
            pn.Column("### Aggregated Map", tiles * dmap_hex),
            pn.Column("### Individual Sales", tiles * dmap_points)
        )
        
        controls = pn.Column(
            "## Controls",
            self.w_date,
            self.w_var,
            self.w_zoom,
            self.w_comm
        )
        
        return pn.template.FastListTemplate(
            title="Real Estate Analytics Dashboard",
            sidebar=[controls],
            main=[map_row],
            accent_base_color="#2F4F4F",
            header_background="#2F4F4F"
        )