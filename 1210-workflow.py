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
import cartopy.crs as ccrs # Required for the CRS fix
from holoviews.streams import RangeXY


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
import h3_map
reload(h3_map)
from h3_map import *


#%%  Set up date
df = pd.read_csv('data/sales_202025.csv')
# process features
data = dataset()._prepare_data(df)
# keep copy of full dataframe
full_dataframe = data.dataframe

# Declare model manager object prepare tensors and test/trai split
model = modelmanager()
model.processor(data)
model.split_data()

# Train model on full dataset
model.train_model(embedding_dim=8, hidden_dim=8, property_dim=3, 
                  epochs = 20, batch = 256, learning_rate = 0.001)
model.add_predictions_to_data()

# region Start mapping
#%%
import h3_map
reload(h3_map)
from h3_map import *
import point_map
reload(point_map)
from point_map import *


source_crs = ccrs.PlateCarree()
display_crs = ccrs.Mercator()

min_date = model.dataframe['sale_date'].min().date()
max_date = model.dataframe['sale_date'].max().date()

date_slider = pn.widgets.DateRangeSlider(
    name='Date Range',
    start=min_date,
    end=max_date,
    value=(min_date, max_date)
)

var_selector = pn.widgets.Select(
    name='Variable', 
    options=list(h3_map.COLUMN_CONFIGS.keys()), 
    value='pct_error'
)
hex_size_selector = pn.widgets.Select(
    name='H3 Index', 
    options=list(h3_map.ZOOM_LEVELS.keys()), 
    value='Low (7)'
)
# Dummy Box

# Define the initial bounds (e.g., the whole US or your city)
# You can grab these from your dataframe
initial_x = (model.dataframe['lng'].min(), model.dataframe['lng'].max())
initial_y = (model.dataframe['lat'].min(), model.dataframe['lat'].max())

# Create the shared stream object
common_range_stream = RangeXY(x_range=initial_x, y_range=initial_y)


dmap = gv.DynamicMap(pn.bind(
    get_dynamic_map,
    date_range=date_slider, 
    variable=var_selector,
    zoom_level=hex_size_selector,
    data=model.dataframe)
    ,streams=[common_range_stream])
    
pmap = gv.DynamicMap(pn.bind(
    point_map,
    date_range=date_slider, 
    variable=var_selector,
    data=model.dataframe),
    streams=[common_range_stream])
#dmap = hv.DynamicMap(interactive_map, streams = [range_stream])

#range_stream.source = dmap
# Composite Map
#dmap_composite = gv.tile_sources.CartoLight() * dummy_box * hv.DynamicMap(interactive_map)
dmap_composite =  gv.tile_sources.CartoLight() * dmap
pmap_composite =  gv.tile_sources.CartoLight() * pmap
#%%
# Create a dashboard layout
map_layout = pn.Row(
    pn.Column(pn.Row(hex_size_selector, var_selector), dmap_composite),
    pn.Column(date_slider, pmap_composite))

layout = pn.template.FastListTemplate(
    title="H3 Multi-Scale Analysis",
    main=[map_layout],
    # sidebar=[
    #     "## Controls",
    #     hex_size_selector,
    #     var_selector,
    #     date_slider],
    # main=[pn.Row(dmap_composite),
    #       pn.Row(pmap_composite)],
    accent_base_color="#2F4F4F",
    header_background="#2F4F4F"
)

layout.show()
# %%
