## To do 12/8/2025
# check inclusion of bedrooms (done)
# present CLS figures and error with Panel
# calculate close to water features, close to major transit, highway, travel times to cbvd by both methods..  with h3 for each h3 index.
# add breakdown of CLS for property features.
# 12/9/2025
# Bring in King Co latest sales data.

#%%
import sys
import os
import holoviews as hv
import pandas as pd


sys.path.insert(0,os.getcwd()+'/src/pricemodel')
sys.path.insert(0,os.getcwd()+'/src/spatial')
sys.path.insert(0,os.getcwd()+'/src/panel')
#sys.path.insert(0,'/Users/marie/PycharmProjects/neural-networks-house-prices/src/pricemodel')
from importlib import reload
import embedding_model
reload(embedding_model)
from embedding_model import *
import panel as pn
import geoviews as gv
import folium_map
reload(folium_map)


#%%  Set up date
df = pd.read_csv('data/sales_202025.csv')
# process features
data = dataset()._prepare_data(df)
# keep copy of full dataframe
full_dataframe = data.dataframe

#%% Declare model manager object prepare tensors and test/trai split
model = modelmanager()
model.processor(data)
model.split_data()

#%% Train model on full dataset
model.train_model(embedding_dim=8, hidden_dim=8, property_dim=3, 
                  epochs = 20, batch = 256, learning_rate = 0.001)
model.add_predictions_to_data()


#%%


# plot_options are variables to display on map
plot_options = ['pct_error', 'predicted_price', 'sale_price', 'sqft', 'community']
# We include ALL columns we might ever need in vdims
all_vdims = plot_options + ['sale_date', 'sale_date_ts']
# --- CONFIGS ---
column_configs = {
    'pct_error': {'cmap': 'Viridis', 'format': '0.0%', 'label': 'Error', 'center_zero': True},
    'predicted_price': {'cmap': 'Viridis', 'format': '$0.0a', 'label': 'Pred. Price', 'center_zero': False},
    'sale_price': {'cmap': 'Inferno', 'format': '$0.0a', 'label': 'Sale Price', 'center_zero': False},
    'sqft': {'cmap': 'Viridis', 'format': '0,0', 'label': 'Size (SqFt)', 'center_zero': False},
    'community': {'cmap': 'Turbo', 'format': '0', 'label': 'Comm ID', 'center_zero': False}
}
color_selector = pn.widgets.Select(name='Variable to display',
                                   options=plot_options, value='pct_error')

model.dataframe['sale_date'] = pd.to_datetime(model.dataframe['sale_date'])
model.dataframe['sale_date_ts'] = model.dataframe['sale_date'].astype('int64') 
min_date = model.dataframe['sale_date'].min()
max_date = model.dataframe['sale_date'].max()

date_slider =  pn.widgets.DateRangeSlider(
    name='Filter Date Range',
    start=min_date, end=max_date,
    value=(min_date, max_date),
    step= 1
)
#%%
# region start mapping)


#%%
import folium_map
reload(folium_map)
from folium_map import create_map, get_mercator_bounds, create_map2

pn.extension()
hv.extension('bokeh')
global_points = gv.Points(model.dataframe, kdims=['lng', 'lat'], vdims=all_vdims)
xlim_merc, ylim_merc = get_mercator_bounds(global_points.data['lng'], global_points.data['lat'])
# This creates a "DynamicMap" that updates whenever color_selector changes
interactive_map = pn.bind(create_map2,
                          selected_column=color_selector,
                          date_range=date_slider,
                          column_configs=column_configs,
                          global_points=global_points,
                          plot_options=plot_options,
                          xlim_merc=xlim_merc,
                          ylim_merc=ylim_merc
                          )

# Layout
template = pn.template.FastListTemplate(
    title='Real Estate Model Playground',
    sidebar=[
        "## Filters",
        date_slider,
        "## Visualization",
        color_selector
    ],
    main=[pn.Row(pn.Card(interactive_map, title="Map", sizing_mode='stretch_both'))],
    accent_base_color="#1f77b4",
    header_background="#202427"
)
template.show()
# region use h3 map
#%%
import h3_map
reload(h3_map)
from h3_map import get_dynamic_map, ZOOM_LEVELS
from holoviews.streams import RangeXY
# =========================================================
# 3. INITIALIZATION
# =========================================================
COLUMN_CONFIGS = {
    'pct_error': {'cmap': 'Viridis', 'format': '0.0%', 'label': 'Error', 'center_zero': True},
    'predicted_price': {'cmap': 'Viridis', 'format': '$0.0a', 'label': 'Pred. Price', 'center_zero': False},
    'sale_price': {'cmap': 'Inferno', 'format': '$0.0a', 'label': 'Sale Price', 'center_zero': False},
    'sqft': {'cmap': 'Viridis', 'format': '0,0', 'label': 'Size (SqFt)', 'center_zero': False},
    'community': {'cmap': 'Turbo', 'format': '0', 'label': 'Comm ID', 'center_zero': False}
}
var_selector = pn.widgets.Select(
    name='Variable', 
    options=list(COLUMN_CONFIGS.keys()), 
    value='pct_error'
)

# Convert min/max to .date() for the slider widget, as it expects date objects
min_date = model.dataframe['sale_date'].min().date()
max_date = model.dataframe['sale_date'].max().date()

date_slider = pn.widgets.DateRangeSlider(
    name='Date Range',
    start=min_date,
    end=max_date,
    value=(min_date, max_date)
)

# Dummy Box
data_bounds = (
    model.dataframe['lng'].min(), model.dataframe['lng'].max(),
    model.dataframe['lat'].min(), model.dataframe['lat'].max()
)
dummy_box = gv.Polygons([]).opts(
    xlim=(data_bounds[0], data_bounds[1]), 
    ylim=(data_bounds[2], data_bounds[3]),
    alpha=0
) 

range_stream = RangeXY(source=dummy_box)

interactive_map = pn.bind(
    get_dynamic_map,
    x_range=range_stream.param.x_range,
    y_range=range_stream.param.y_range,
    date_range=date_slider, 
    variable=var_selector,
    data=model.dataframe
)

# Composite Map
#dmap_composite = gv.tile_sources.CartoLight() * dummy_box * hv.DynamicMap(interactive_map)

dmap_composite =  dummy_box * hv.DynamicMap(interactive_map)

layout = pn.template.FastListTemplate(
    title="H3 Multi-Scale Analysis",
    sidebar=[
        "## Controls",
        var_selector,
        date_slider,
        "**Zoom Levels:**",
        f"- Coarse: {ZOOM_LEVELS['coarse']['col']}",
        f"- Medium: {ZOOM_LEVELS['medium']['col']}",
        "- Fine: Points / Level 10"
    ],
    main=[pn.Row(dmap_composite)],
    accent_base_color="#2F4F4F",
    header_background="#2F4F4F"
)

layout.show()
# %%
