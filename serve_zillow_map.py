"""
Panel web map showing Zillow listings with predictions and historical sales data
"""
import panel as pn
import geoviews as gv
import holoviews as hv
import pandas as pd
import numpy as np
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

# Historical sales data (V3 model dataset)
sales_df = pd.read_csv('data/sales_2020_25_with_predictions_v4.csv')
sales_df['sale_date'] = pd.to_datetime(sales_df['sale_date'])
print(f"Loaded {len(sales_df)} historical sales")

# RentCast recent sales with predictions
rentcast_df = pd.read_csv('data/rentcast_with_predictions_v3_finetuned.csv')
rentcast_df['sale_date'] = pd.to_datetime(rentcast_df['sale_date'])
print(f"Loaded {len(rentcast_df)} RentCast recent sales")

# Harmonize column names
if 'bath_full' in sales_df.columns and 'baths' not in sales_df.columns:
    sales_df['baths'] = sales_df['bath_full']
if 'baths' in rentcast_df.columns and 'bath_full' not in rentcast_df.columns:
    rentcast_df['bath_full'] = rentcast_df['baths']

# Combine sales data
sales_df = pd.concat([sales_df, rentcast_df], ignore_index=True)
print(f"Combined total: {len(sales_df)} sales records")

# Prepare Zillow data
zillow_df['display_text'] = zillow_df.apply(lambda row: 
    f"{row['address']}\n"
    f"Listed: ${row['sale_price']:,.0f}\n"
    f"Predicted: ${row['predicted_price']:,.0f}\n"
    f"Error: {row['pct_error']:.1f}%\n"
    f"Beds: {row['beds']:.0f} | Baths: {row['baths']:.1f}\n"
    f"Sqft: {row['sqft']:.0f}\n"
    f"Type: {row['home_type']}\n"
    f"Days on Zillow: {row['days_on_zillow']:.0f}",
    axis=1
)

# Prepare sales data (use baths column which now exists in both)
sales_df['display_text'] = sales_df.apply(lambda row:
    f"Sale Date: {row['sale_date'].strftime('%Y-%m-%d')}\n"
    f"Sale Price: ${row['sale_price']:,.0f}\n"
    f"Predicted: ${row['predicted_price']:,.0f}\n"
    f"Error: {row['pct_error']:.1f}%\n"
    f"Beds: {row['beds']:.0f} | Baths: {row['baths']:.1f}\n"
    f"Sqft: {row['sqft']:.0f}\n"
    f"Community: {row['community']}",
    axis=1
)

# Filter sales to recent years for performance
sales_df_recent = sales_df[sales_df['sale_date'] >= '2020-01-01'].copy()
print(f"Filtered to {len(sales_df_recent)} sales (2020+)")

# Add source label for visualization
sales_df_recent['source'] = 'Historical'
sales_df_recent.loc[sales_df_recent['sale_date'] >= '2025-09-01', 'source'] = 'RentCast'

print(f"  Historical: {(sales_df_recent['source'] == 'Historical').sum()}")
print(f"  RentCast: {(sales_df_recent['source'] == 'RentCast').sum()}")


def create_zillow_map(error_threshold_zillow=50):
    """Create map of Zillow listings colored by prediction error"""
    
    # Filter extreme errors for better visualization
    df = zillow_df[zillow_df['pct_error'].abs() <= error_threshold_zillow].copy()
    
    # Create points
    points = gv.Points(
        df,
        kdims=['lng', 'lat'],
        vdims=['pct_error', 'display_text', 'sale_price', 'sqft', 'address', 'url']
    )
    
    # Style the points - blue border for Zillow listings
    points = points.opts(
        opts.Points(
            color='pct_error',
            cmap='RdYlGn_r',  # Red for overpriced, green for underpriced
            clim=(-error_threshold_zillow, error_threshold_zillow),
            size=10,
            alpha=0.8,
            line_color='blue',
            line_width=1.5,
            tools=['hover', 'tap'],
            hover_tooltips=[
                ('Address', '@address'),
                ('Listed', '$@sale_price{0,0}'),
                ('Info', '@display_text')
            ],
            colorbar=True,
            colorbar_opts={'title': 'Prediction Error (%)'},
            width=600,
            height=500,
            title="Zillow Listings (Blue Border)"
        )
    )
    
    return points


def create_sales_map(date_range, error_threshold_sales=50):
    """Create map of historical sales colored by prediction error"""
    
    start_date, end_date = date_range
    
    # Filter by date range
    df = sales_df_recent[
        (sales_df_recent['sale_date'] >= pd.Timestamp(start_date)) &
        (sales_df_recent['sale_date'] <= pd.Timestamp(end_date))
    ].copy()
    
    # Filter extreme errors
    df = df[df['pct_error'].abs() <= error_threshold_sales]
    
    if len(df) == 0:
        # Return empty plot
        return gv.Points([]).opts(
            opts.Points(
                width=600,
                height=500,
                title="Historical Sales (No data in range)"
            )
        )
    
    # Create points
    points = gv.Points(
        df,
        kdims=['lng', 'lat'],
        vdims=['pct_error', 'display_text', 'sale_price', 'sqft', 'sale_date']
    )
    
    # Style the points - red border for historical sales
    points = points.opts(
        opts.Points(
            color='pct_error',
            cmap='RdYlGn_r',
            clim=(-error_threshold_sales, error_threshold_sales),
            size=6,
            alpha=0.6,
            line_color='red',
            line_width=1,
            tools=['hover', 'tap'],
            hover_tooltips=[
                ('Sale Date', '@sale_date{%F}'),
                ('Sale Price', '$@sale_price{0,0}'),
                ('Info', '@display_text')
            ],
            colorbar=True,
            colorbar_opts={'title': 'Prediction Error (%)'},
            width=600,
            height=500,
            title=f"Historical Sales ({len(df)} properties, Red Border)"
        )
    )
    
    return points


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

# Layer toggles
show_zillow = pn.widgets.Checkbox(name='Show Zillow Listings', value=True)
show_sales = pn.widgets.Checkbox(name='Show Historical Sales', value=True)

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
    
    historical_count = (sales_df_recent['source'] == 'Historical').sum()
    rentcast_count = (sales_df_recent['source'] == 'RentCast').sum()
    
    sales_stats = f"""
    ### Historical Sales
    - Total (2020+): {len(sales_df_recent)}
    - Historical: {historical_count}
    - RentCast: {rentcast_count}
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


def create_combined_map(date_range, error_threshold_zillow, error_threshold_sales, show_z, show_s, x_range=None, y_range=None):
    """Create combined map with both layers"""
    
    layers = []
    
    # Base map
    base = gv.tile_sources.CartoLight()
    layers.append(base)
    
    # Zillow layer
    if show_z:
        zillow_layer = create_zillow_map(error_threshold_zillow)
        layers.append(zillow_layer)
    
    # Sales layer
    if show_s:
        sales_layer = create_sales_map(date_range, error_threshold_sales)
        layers.append(sales_layer)
    
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
    "### Error Filters",
    error_slider_zillow,
    error_slider_sales,
    "### Date Range",
    date_slider,
    "---",
    "## Legend",
    pn.pane.Markdown("""
    **Zillow Listings**: Blue border, larger points  
    **Historical Sales**: Red border, smaller points  
    **Color**: Red = overpriced, Green = underpriced
    """),
    "---",
    get_stats(),
    width=300
)

map_panel = pn.Column(
    "## Seattle Real Estate: Listings & Sales",
    dmap,
    width=650
)

# Create dashboard
dashboard = pn.template.FastListTemplate(
    title="Seattle Real Estate Map - Zillow Listings & Historical Sales",
    sidebar=[controls],
    main=[map_panel],
    accent_base_color="#2F4F4F",
    header_background="#2F4F4F"
)

print("\n" + "="*80)
print("Starting Panel server...")
print("="*80)
print("\nMap Features:")
print("  - Zillow Listings: Current listings with predictions")
print("  - Historical Sales: Past sales from model training data (2020+)")
print("  - RentCast Sales: Recent sales from RentCast API (2025-2026)")
print("  - Color: Red = overpriced, Green = underpriced")
print("  - Hover: View property details and listing URL")
print("\nControls:")
print("  - Toggle layers on/off")
print("  - Filter by date range")
print("  - Adjust error threshold")
print("\n" + "="*80)

# Serve the dashboard
dashboard.show(port=5006, threaded=False)
