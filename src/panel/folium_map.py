import traceback # Import this to see full error logs
import panel as pn
import holoviews as hv
import geoviews as gv
import datashader as ds
from holoviews.operation.datashader import rasterize, dynspread
from bokeh.models import NumeralTickFormatter, HoverTool, CustomJSHover
import pandas as pd
import numpy as np
from geoviews.operation import project
from bokeh.io import curdoc

def create_map(selected_column, date_range, column_configs, global_points,
               plot_options, xlim_merc, ylim_merc):
    
    lat_custom = CustomJSHover(code="""
        var meters = special_vars.y;
        var lat = (2 * Math.atan(Math.exp(meters / 6378137)) - Math.PI / 2) * 180 / Math.PI;
        return lat.toFixed(5);
    """)

    lon_custom = CustomJSHover(code="""
        var meters = special_vars.x;
        var lon = (meters / 6378137) * 180 / Math.PI;
        return lon.toFixed(5);
    """)

    date_custom = CustomJSHover(code="""
        var ts = value;
        if (isNaN(ts) || ts === 0) { return "N/A"; }
        var date = new Date(ts);
        return date.toISOString().split('T')[0];
    """)

    # --- 1. SETUP ---
    base_map = gv.tile_sources.CartoLight()

    if selected_column is None:
        selected_column = plot_options[0]

    config = column_configs.get(selected_column)

    # --- 2. FILTERING ---
    start_date, end_date = date_range
    filtered_points = global_points.select(sale_date=(start_date, end_date))
    
    if len(filtered_points) == 0:
        return base_map.opts(title="No data in selected range")

    # --- 3. LIMITS ---
    try:
        data_array = filtered_points.dimension_values(selected_column)
        vmin, vmax = np.nanpercentile(data_array, 5), np.nanpercentile(data_array, 95)
        if config.get('center_zero', False):
            limit = max(abs(vmin), abs(vmax))
            vmin, vmax = -limit, limit
        if vmin == vmax: vmin, vmax = 0, 1
    except:
        vmin, vmax = 0, 1

    # --- 4. PROJECT & RASTERIZE ---
    projected_points = project(filtered_points)

    agg_dict = {col: ds.mean(col) for col in plot_options}
    agg_dict['sale_date_ts'] = ds.max('sale_date_ts')
    
    raster = rasterize(projected_points, aggregator=ds.summary(**agg_dict))

    # --- 5. REORDER & SPREAD ---
    def reorder_and_spread_layer(element):
        try:
            ds_data = element.data
            
            # A. Rename Coords (lng/lat -> x/y) if needed
            coords = list(ds_data.coords.keys())
            rename_map = {}
            if 'lng' in coords: rename_map['lng'] = 'x'
            if 'lat' in coords: rename_map['lat'] = 'y'
            if rename_map:
                ds_data = ds_data.rename(rename_map)

            # B. Determine Variables
            available_vars = list(ds_data.data_vars.keys())
            target_col = selected_column
            if target_col is None or target_col not in available_vars:
                target_col = available_vars[0]

            other_cols = [c for c in plot_options if c in available_vars and c != target_col]
            extras = []
            if 'sale_date_ts' in available_vars:
                extras.append('sale_date_ts')
            
            full_order = [target_col] + other_cols + extras
            full_order = [c for c in full_order if c is not None]

            # C. SPREAD LOOP
            import xarray as xr
            spread_arrays = {}
            
            for col in full_order:
                # 1. Isolate the variable
                tmp_ds = ds_data[col]
                
                # 2. Create Temp Image
                # We force kdims=['x', 'y'] to ensure the spreader knows it's a grid
                tmp_img = hv.Image(tmp_ds, kdims=['x', 'y'], vdims=[col])
                
                # 3. Spread (max_px=3 = Bigger Points)
                spread_result = dynspread(tmp_img, max_px=3, threshold=0)
                
                # 4. Extract Data and Transpose
                # HoloViews Image expects (y, x). XArray sometimes gives (x, y).
                # We enforce the transpose to ensure alignment.
                res_data = spread_result.data
                
                # Get the variable (Datashader might retain the name or not)
                # But since we isolated it, it's the only data_var
                var_key = list(res_data.data_vars.keys())[0]
                arr = res_data[var_key]
                
                # Force Transpose: Ensure dims are ('y', 'x')
                if arr.dims == ('x', 'y'):
                    arr = arr.transpose('y', 'x')
                
                # Rename back to the original column name so tooltips find it
                arr.name = col
                spread_arrays[col] = arr

            # D. RECOMBINE
            # Use the coordinates from the main variable
            final_coords = spread_arrays[target_col].coords
            
            # Create final Dataset
            final_ds = xr.Dataset(spread_arrays, coords=final_coords)
            
            # Create final Image
            # kdims must match the dataset coordinates
            return hv.Image(final_ds, kdims=['x', 'y'], vdims=full_order)

        except Exception as e:
            print(f"Spread Error: {e}")
            print(traceback.format_exc())
            return element

    raster_ordered = raster.apply(reorder_and_spread_layer)

    # --- 6. TOOLTIPS & STYLING ---
    formatter = NumeralTickFormatter(format=config['format'])
    
    tooltips = [
        ('Latitude', '$y{custom}'), 
        ('Longitude', '$x{custom}'),
        ('Recent Sale', '@sale_date_ts{custom}')
    ]
    for col, cfg in column_configs.items():
        tooltips.append((cfg['label'], f"@{col}{{{cfg['format']}}}"))

    hover = HoverTool(
        tooltips=tooltips,
        formatters={
            '$y': lat_custom,
            '$x': lon_custom,
            '@sale_date_ts': date_custom
        }
    )

    styled_raster = raster_ordered.opts(
        cmap=config['cmap'],
        clim=(vmin, vmax),
        colorbar=True,
        colorbar_opts={'formatter': formatter, 'title': config['label']},
        clabel=config['label'],
        alpha=0.8,
        tools=[hover]
    )

    return (base_map * styled_raster).opts(
        width=1000, 
        height=1000,
        title=f"Analysis: {config['label']} ({len(filtered_points)} points)",
        xaxis=None,      
        yaxis=None,
        xlim=xlim_merc,
        ylim=ylim_merc
    )


import xarray as xr
from holoviews.operation.datashader import spread # Use spread, not dynspread

def create_map2(selected_column, date_range, column_configs, global_points,
               plot_options, xlim_merc, ylim_merc):
    
    lat_custom = CustomJSHover(code="""
        var meters = special_vars.y;
        var lat = (2 * Math.atan(Math.exp(meters / 6378137)) - Math.PI / 2) * 180 / Math.PI;
        return lat.toFixed(5);
    """)

    lon_custom = CustomJSHover(code="""
        var meters = special_vars.x;
        var lon = (meters / 6378137) * 180 / Math.PI;
        return lon.toFixed(5);
    """)

    date_custom = CustomJSHover(code="""
        var ts = value;
        if (isNaN(ts) || ts === 0) { return "N/A"; }
        var date = new Date(ts);
        return date.toISOString().split('T')[0];
    """)


    # --- 1. SETUP ---
    base_map = gv.tile_sources.CartoLight()
    if selected_column is None: selected_column = plot_options[0]
    config = column_configs.get(selected_column)

    # --- 2. FILTERING ---
    start_date, end_date = date_range
    filtered_points = global_points.select(sale_date=(start_date, end_date))
    if len(filtered_points) == 0: return base_map.opts(title="No data")

    # --- 3. LIMITS ---
    try:
        data_array = filtered_points.dimension_values(selected_column)
        vmin, vmax = np.nanpercentile(data_array, 5), np.nanpercentile(data_array, 95)
        if config.get('center_zero', False):
            limit = max(abs(vmin), abs(vmax))
            vmin, vmax = -limit, limit
        if vmin == vmax: vmin, vmax = 0, 1
    except: vmin, vmax = 0, 1

    # --- 4. PROJECT & RASTERIZE ---
    projected_points = project(filtered_points)

    agg_dict = {col: ds.mean(col) for col in plot_options}
    agg_dict['sale_date_ts'] = ds.max('sale_date_ts')
    
    # FIX A: HARDCODE RESOLUTION
    # width/height=600 forces the pixels to be larger blocks.
    # This solves "points too small" naturally.
    raster = rasterize(projected_points, aggregator=ds.summary(**agg_dict), 
                       width=600, height=600)

    # --- 5. REORDER & SPREAD ---
    def process_layer(element):
        try:
            ds_data = element.data
            
            # A. Rename Coords
            if 'lng' in ds_data.coords: ds_data = ds_data.rename({'lng': 'x'})
            if 'lat' in ds_data.coords: ds_data = ds_data.rename({'lat': 'y'})
            
            # B. Identify Columns
            available = list(ds_data.data_vars.keys())
            target = selected_column if selected_column in available else available[0]
            
            # C. SPREAD LOOP (Using xr.merge)
            # We spread each variable individually to ensure the 'Halo' has data
            spread_parts = []
            
            # List of all cols we need
            needed_cols = [target] + [c for c in plot_options if c in available and c != target]
            if 'sale_date_ts' in available: needed_cols.append('sale_date_ts')

            for col in needed_cols:
                # 1. Isolate
                layer = ds_data[col]
                # 2. Image Wrapper
                img = hv.Image(layer, kdims=['x', 'y'], vdims=[col])
                # 3. Spread (px=1 adds 1 pixel border around every point)
                # This makes every point a 3x3 grid
                spread_img = spread(img, px=1)
                # 4. Collect the resulting DataArray
                spread_parts.append(spread_img.data[col])

            # D. MERGE
            # XArray merge handles alignment automatically
            merged_ds = xr.merge(spread_parts)
            
            # E. RETURN IMAGE
            # We construct the final image from the merged, spread dataset
            return hv.Image(merged_ds, kdims=['x', 'y'], vdims=needed_cols)

        except Exception as e:
            print(f"Layer Error: {e}")
            return element

    raster_ordered = raster.apply(process_layer)

    # --- 6. TOOLTIPS & STYLING ---
    formatter = NumeralTickFormatter(format=config['format'])
    
    tooltips = [
        ('Latitude', '$y{custom}'), 
        ('Longitude', '$x{custom}'),
        ('Recent Sale', '@sale_date_ts{custom}')
    ]
    for col, cfg in column_configs.items():
        tooltips.append((cfg['label'], f"@{col}{{{cfg['format']}}}"))

    hover = HoverTool(
        tooltips=tooltips,
        formatters={'$y': lat_custom, '$x': lon_custom, '@sale_date_ts': date_custom}
    )

    styled_raster = raster_ordered.opts(
        cmap=config['cmap'],
        clim=(vmin, vmax),
        colorbar=True,
        colorbar_opts={'formatter': formatter, 'title': config['label']},
        clabel=config['label'],
        alpha=0.8,
        tools=[hover]
    )

    return (base_map * styled_raster).opts(
        width=1000, height=1000,
        title=f"Analysis: {config['label']} ({len(filtered_points)} points)",
        xaxis=None, yaxis=None,
        xlim=xlim_merc, ylim=ylim_merc
    )