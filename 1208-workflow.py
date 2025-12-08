## To do 12/8/2025
# check inclusion of bedrooms (done)
# present CLS figures and error with Panel
# calculate close to water features, close to major transit, highway, travel times to cbvd by both methods..  with h3 for each h3 index.
# add breakdown of CLS for property features.

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
print(model.dataframe['pct_error'].mean())


#%%
hv.extension('bokeh')
# region start mapping
plot_options=['sale_price', 'predicted_price','sqft',
             'pct_error','community','cls_week', 'cls_year', 'cls_community',
             'cls_property']
color_selector = pn.widgets.Select(name='Color By', options=plot_options, value='pct_error')
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
# This creates a "DynamicMap" that updates whenever color_selector changes
interactive_map = pn.bind(folium_map.create_map, df = model.dataframe,
                          variables_show = plot_options, 
                          date_range = date_slider, 
                          selected_column=color_selector)
# Layout the dashboard
# Create a professional template
template = pn.template.FastListTemplate(
    title='Real Estate Model Playground',
    sidebar=[
        "## Filters",
        date_slider,
        "## Visualization",
        color_selector,
        "**Instructions:** Use the date slider to narrow down specific sales periods."
    ],
    main=[
        pn.Row(pn.Card(interactive_map, title="Geospatial Error Analysis", sizing_mode='stretch_both'))
    ],
    accent_base_color="#1f77b4",
    header_background="#202427"
)
template.servable()
template.show()

# %%
