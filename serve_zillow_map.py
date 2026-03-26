"""
Panel web map showing Zillow listings with predictions and H3 L9 aggregated historical sales
"""
import panel as pn
import geoviews as gv
import holoviews as hv
import pandas as pd
import numpy as np
import h3
import geopandas as gpd
from shapely.geometry import Polygon
import cartopy.crs as ccrs
from holoviews import opts
from holoviews.streams import RangeXY

# Enable Panel extension
pn.extension()
hv.extension('bokeh')

# Load data
print("Loading data...")

# Zillow listings with predictions
zillow_df = pd.read_csv('data/zillow_with_predictions.csv')
print(f"Loaded {len(zillow_df)} Zillow listings")

# Historical sales data (V4 model dataset with combined data)
sales_df = pd.read_csv('data/sales_2020_25_with_predictions_v4.csv')
sales_df['sale_date'] = pd.to_datetime(sales_df['sale_date'])
print(f"Loaded {len(sales_df)} historical sales")

# Ensure h3_09 column exists
if 'h3_09' not in sales_df.columns:
    print("Generating H3 L9 indices for sales data...")
    sales_df['h3_09'] = sales_df.apply(
        lambda row: h3.latlng_to_cell(row['lat'], row['lng'], 9) 
        if pd.notna(row['lat']) and pd.notna(row['lng']) 
        else None,
        axis=1
    )

# Filter sales to recent years for performance
sales_df_recent = sales_df[sales_df['sale_date'] >= '2020-01-01'].copy()
print(f"Filtered to {len(sales_df_recent)} sales (2020+)")

# Prepare Zillow data
zillow_df['display_text'] = zillow_df.apply(lambda row: 
    f"{row['address']}<br>"
    f"Listed: ${row['sale_price']:,.0f}<br>"
    f"Predicted: ${row['predicted_price']:,.0f}<br>"
    f"Error: {row['pct_error']:.1f}%<br>"
    f"Beds: {row['beds']:.0f} | Baths: {row['baths']:.1f}<br>"
    f"Sqft: {row['sqft']:.0f}<br>"
    f"Type: {row['home_type']}<br>"
    f"Days on Zillow: {row['days_on_zillow']:.0f}",
    axis=1
)


def aggregate_sales_by_h3(df, variable='pct_error'):
    """Aggregate sales data by H3 L9 hexagons"""
    
    # Group by H3 and calculate aggregates
    agg_dict = {
        'sale_price': 'mean',
        'predicted_price': 'mean',
        'pct_error': 'mean',
        'sqft': 'mean',
        'beds': 'mean',
        'baths': 'mean',
        'lat': 'first',
        'lng': 'first',
        'sale_date': ['min', 'max', 'count']
    }
    
    h3_agg = df.groupby('h3_09').agg(agg_dict).reset_index()
    
    # Flatten column names
    h3_agg.columns = ['h3_09', 'avg_sale_price', 'avg_predicted_price', 'avg_pct_error',
                      'avg_sqft', 'avg_beds', 'avg_baths', 'lat', 'lng',
                      'min_date', 'max_date', 'num_sales']
    
    # Create display text
    h3_agg['display_text'] = h3_agg.apply(lambda row:
        f"H3 Hex: {row['h3_09']}<br>"
        f"Sales: {row['num_sales']:.0f}<br>"
        f"Avg Price: ${row['avg_sale_price']:,.0f}<br>"
        f"Avg Predicted: ${row['avg_predicted_price']:,.0f}<br>"
        f"Avg Error: {row['avg_pct_error']:.1f}%<br>"
        f"Avg Sqft: {row['avg_sqft']:.0f}<br>"
        f"Avg Beds: {row['avg_beds']:.1f}<br>"
        f"Date Range: {row['min_date'].strftime('%Y-%m-%d')} to {row['max_date'].strftime('%Y-%m-%d')}",
        axis=1
    )
    
    # Get the variable to display
    if variable == 'pct_error':
        h3_agg['display_value'] = h3_agg['avg_pct_error']
        h3_agg['value_label'] = 'Avg Error (%)'
    elif variable == 'sale_price':
        h3_agg['display_value'] = h3_agg['avg_sale_price']
        h3_agg['value_label'] = 'Avg Sale Price ($)'
    elif variable == 'sqft':
        h3_agg['display_value'] = h3_agg['avg_sqft']
        h3_agg['value_label'] = 'Avg Sqft'
    elif variable == 'num_sales':
        h3_agg['display_value'] = h3_agg['num_sales']
        h3_agg['value_label'] = 'Number of Sales'
    
    return h3_agg


def create_zillow_map(error_threshold_zillow=50, sqft_min=0, sqft_max=10000):
    """Create map of Zillow listings colored by prediction error"""
    
    # Filter by error and sqft
    df = zillow_df[
        (zillow_df['pct_error'].abs() <= error_threshold_zillow) &
        (zillow_df['sqft'] >= sqft_min) &
        (zillow_df['sqft'] <= sqft_max)
    ].copy()
    
    # Add clickable link
    df['link'] = df['url'].apply(lambda x: f'<a href="{x}" target="_blank">View Listing</a>')
    
    # Create points
    points = gv.Points(
        df,
        kdims=['lng', 'lat'],
        vdims=['pct_error', 'display_text', 'link']
    )
    
    # Style the points - blue border for Zillow listings
    points = points.opts(
        opts.Points(
            color='pct_error',
            cmap='RdYlGn_r',
            clim=(-error_threshold_zillow, error_threshold_zillow),
            size=10,
            alpha=0.8,
            line_color='blue',
            line_width=1.5,
            tools=['hover', 'tap'],
            hover_tooltips='<div style="width: 250px;">@display_text<br>@link</div>',
            colorbar=True,
            colorbar_opts={'title': 'Prediction Error (%)'},
            xaxis=None,
            yaxis=None,
            width=900,
            height=700,
            title=f"Zillow Listings ({len(df)} properties)"
        )
    )
    
    return points


def create_sales_map(date_range, error_threshold_sales=50, sqft_min=0, sqft_max=10000, variable='pct_error'):
    """Create map of H3 aggregated sales colored by selected variable"""
    
    start_date, end_date = date_range
    
    # Filter by date range and sqft
    df = sales_df_recent[
        (sales_df_recent['sale_date'] >= pd.Timestamp(start_date)) &
        (sales_df_recent['sale_date'] <= pd.Timestamp(end_date)) &
        (sales_df_recent['sqft'] >= sqft_min) &
        (sales_df_recent['sqft'] <= sqft_max)
    ].copy()
    
    # Filter extreme errors
    df = df[df['pct_error'].abs() <= error_threshold_sales]
    
    if len(df) == 0:
        return gv.Polygons([], crs=ccrs.PlateCarree()).opts(
            opts.Polygons(
                width=900,
                height=700,
                xaxis=None,
                yaxis=None,
                title="Historical Sales (No data in range)"
            )
        )
    
    # Aggregate by H3
    h3_agg = aggregate_sales_by_h3(df, variable=variable)
    
    # Create Shapely polygons from H3 indices
    # cell_to_boundary returns (lat, lng) tuples, need to swap to (lng, lat) for Shapely
    geometries = [
        Polygon([(lng, lat) for lat, lng in h3.cell_to_boundary(h3_idx)]) 
        for h3_idx in h3_agg['h3_09']
    ]
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(
        h3_agg,
        geometry=geometries,
        crs='EPSG:4326'
    )
    
    # Determine color limits based on variable
    if variable == 'pct_error':
        clim = (-error_threshold_sales, error_threshold_sales)
        cmap = 'RdYlGn_r'
        colorbar_title = 'Avg Error (%)'
    elif variable == 'sale_price':
        clim = (gdf['display_value'].quantile(0.05), gdf['display_value'].quantile(0.95))
        cmap = 'viridis'
        colorbar_title = 'Avg Sale Price ($)'
    elif variable == 'sqft':
        clim = (gdf['display_value'].quantile(0.05), gdf['display_value'].quantile(0.95))
        cmap = 'plasma'
        colorbar_title = 'Avg Sqft'
    elif variable == 'num_sales':
        clim = (1, gdf['display_value'].quantile(0.95))
        cmap = 'YlOrRd'
        colorbar_title = 'Number of Sales'
    
    # Create polygons using GeoDataFrame
    polygons = gv.Polygons(
        gdf,
        vdims=['display_value', 'display_text'],
        crs=ccrs.PlateCarree()
    ).opts(
        opts.Polygons(
            color='display_value',
            cmap=cmap,
            clim=clim,
            alpha=0.5,
            line_color='darkgray',
            line_width=0.5,
            tools=['hover'],
            hover_tooltips='<div style="width: 300px;">@display_text</div>',
            colorbar=True,
            colorbar_opts={'title': colorbar_title},
            xaxis=None,
            yaxis=None,
            width=900,
            height=700,
            title=f"H3 Aggregated Sales ({len(h3_agg)} hexes, {int(h3_agg['num_sales'].sum())} sales)"
        )
    )
    
    return polygons


# Create widgets
print("Creating widgets...")

# Date range slider for historical sales
min_date = sales_df_recent['sale_date'].min().date()
max_date = sales_df_recent['sale_date'].max().date()

date_slider = pn.widgets.DateRangeSlider(
    name='Historical Sales Date Range',
    start=min_date,
    end=max_date,
    value=(min_date, max_date),
    width=400
)

# Error threshold sliders - separate for each dataset
error_slider_zillow = pn.widgets.IntSlider(
    name='Zillow Max Error (%)',
    start=10,
    end=100,
    value=50,
    step=5,
    width=300
)

error_slider_sales = pn.widgets.IntSlider(
    name='Sales Max Error (%)',
    start=10,
    end=100,
    value=50,
    step=5,
    width=300
)

# Sqft filter
sqft_range = pn.widgets.RangeSlider(
    name='Square Feet Range',
    start=0,
    end=10000,
    value=(0, 10000),
    step=100,
    width=400
)

# Variable selector for sales layer
variable_selector = pn.widgets.Select(
    name='Sales Layer Variable',
    options=['pct_error', 'sale_price', 'sqft', 'num_sales'],
    value='pct_error',
    width=300
)

# Layer toggles
show_zillow = pn.widgets.Checkbox(name='Show Zillow Listings', value=True)
show_sales = pn.widgets.Checkbox(name='Show Historical Sales (H3 Aggregated)', value=True)

# Statistics
def get_stats():
    """Generate statistics panel"""
    
    zillow_stats = f"""
    ### Zillow Listings
    - Total: {len(zillow_df)}
    - Avg List Price: ${zillow_df['sale_price'].mean():,.0f}
    - Avg Predicted: ${zillow_df['predicted_price'].mean():,.0f}
    - Avg Error: {zillow_df['pct_error'].abs().mean():.1f}%
    """
    
    sales_stats = f"""
    ### Historical Sales (2020+)
    - Total: {len(sales_df_recent)}
    - Unique H3 Hexes: {sales_df_recent['h3_09'].nunique()}
    - Avg Sale Price: ${sales_df_recent['sale_price'].mean():,.0f}
    - Avg Predicted: ${sales_df_recent['predicted_price'].mean():,.0f}
    - Avg Error: {sales_df_recent['pct_error'].abs().mean():.1f}%
    """
    
    return pn.pane.Markdown(zillow_stats + sales_stats)


# Create dynamic map
print("Creating map...")

# Initial bounds
all_lngs = pd.concat([zillow_df['lng'], sales_df_recent['lng']])
all_lats = pd.concat([zillow_df['lat'], sales_df_recent['lat']])
initial_x = (all_lngs.min(), all_lngs.max())
initial_y = (all_lats.min(), all_lats.max())

# Shared range stream for synchronized zooming
common_range_stream = RangeXY(x_range=initial_x, y_range=initial_y)


def create_combined_map(date_range, error_threshold_zillow, error_threshold_sales, 
                        sqft_range_val, variable, show_z, show_s, x_range=None, y_range=None):
    """Create combined map with both layers"""
    
    sqft_min, sqft_max = sqft_range_val
    
    layers = []
    
    # Base map
    base = gv.tile_sources.CartoLight()
    layers.append(base)
    
    # Sales layer (behind, half opacity)
    if show_s:
        sales_layer = create_sales_map(date_range, error_threshold_sales, sqft_min, sqft_max, variable)
        layers.append(sales_layer)
    
    # Zillow layer (on top)
    if show_z:
        zillow_layer = create_zillow_map(error_threshold_zillow, sqft_min, sqft_max)
        layers.append(zillow_layer)
    
    # Combine layers
    if len(layers) == 1:
        return layers[0]
    else:
        combined = layers[0]
        for layer in layers[1:]:
            combined = combined * layer
        return combined


# Create dynamic map
dmap = gv.DynamicMap(
    pn.bind(
        create_combined_map,
        date_range=date_slider,
        error_threshold_zillow=error_slider_zillow,
        error_threshold_sales=error_slider_sales,
        sqft_range_val=sqft_range,
        variable=variable_selector,
        show_z=show_zillow,
        show_s=show_sales
    ),
    streams=[common_range_stream]
)

# Create layout
print("Creating layout...")

controls = pn.Column(
    "## Controls",
    "### Layer Visibility",
    show_zillow,
    show_sales,
    "### Filters",
    sqft_range,
    "### Error Filters",
    error_slider_zillow,
    error_slider_sales,
    "### Sales Layer",
    variable_selector,
    date_slider,
    width=400
)

info_panel = pn.Column(
    "## Legend",
    pn.pane.Markdown("""
    **Zillow Listings**: Blue border, points, on top  
    **Historical Sales**: H3 L9 polygons, half opacity, behind  
    **Color**: Varies by selected variable
    """),
    "---",
    get_stats(),
    width=350
)

map_panel = pn.Column(
    "## Seattle Real Estate: Listings & H3 Aggregated Sales",
    dmap,
    sizing_mode='stretch_both'
)

# Create dashboard with left and right sidebars
dashboard = pn.template.FastListTemplate(
    title="Seattle Real Estate Map",
    sidebar=[controls],
    main=[map_panel, info_panel],
    accent_base_color="#2F4F4F",
    header_background="#2F4F4F"
)

print("\n" + "="*80)
print("Starting Panel server...")
print("="*80)
print("\nMap Features:")
print("  - Zillow Listings: Current listings with predictions (on top)")
print("  - Historical Sales: H3 L9 aggregated averages (behind, 50% opacity)")
print("  - Sqft Filter: Filter both layers by square footage")
print("  - Variable Selector: Choose what to display for sales layer")
print("\nControls:")
print("  - Toggle layers on/off")
print("  - Filter by date range and sqft")
print("  - Adjust error threshold")
print("  - Select sales layer variable")
print("\n" + "="*80)

# Serve the dashboard
dashboard.show(port=5007, threaded=False)
