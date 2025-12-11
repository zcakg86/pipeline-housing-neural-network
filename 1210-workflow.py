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

#%% Declare model manager object prepare tensors and test/trai split
model = modelmanager()
model.processor(data)
model.split_data()

#%% Train model on full dataset
model.train_model(embedding_dim=8, hidden_dim=8, property_dim=3, 
                  epochs = 20, batch = 256, learning_rate = 0.001)
model.add_predictions_to_data()

# region Start mapping

#%%
import h3_map
reload(h3_map)
from h3_map import *

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
# Dummy Box
data_bounds = (
    model.dataframe['lng'].min(), model.dataframe['lng'].max(),
    model.dataframe['lat'].min(), model.dataframe['lat'].max()
)

# Create stream WITHOUT a source first
# range_stream = RangeXY(x_range=(data_bounds[0],data_bounds[1]),
#                        y_range=(data_bounds[2],data_bounds[3]))
#%%
interactive_map = pn.bind(
    get_dynamic_map,
    x_range=range_stream.param.x_range,
    y_range=range_stream.param.y_range,
    date_range=date_slider, 
    variable=var_selector,
    data=model.dataframe)

dmap = hv.DynamicMap(interactive_map)
#dmap = hv.DynamicMap(interactive_map, streams = [range_stream])

#range_stream.source = dmap
# Composite Map
#dmap_composite = gv.tile_sources.CartoLight() * dummy_box * hv.DynamicMap(interactive_map)
dmap_composite =  gv.tile_sources.CartoLight() * dmap



layout = pn.template.FastListTemplate(
    title="H3 Multi-Scale Analysis",
    sidebar=[
        "## Controls",
        var_selector,
        date_slider],
    main=[pn.Row(dmap_composite)],
    accent_base_color="#2F4F4F",
    header_background="#2F4F4F"
)

layout.show()
# %%
