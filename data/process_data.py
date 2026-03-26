#%% process data from Kaggle (just to reduce file size)
import pandas as pd

df = pd.read_csv('data/kingco_sales.csv')
df = df[['sale_date','sale_price','sale_nbr','latitude','longitude','sqft','sqft_lot',
         'year_built','year_reno','sqft_fbsmt','sqft_1','grade','fbsmt_grade','condition',
         'stories','beds','bath_full','bath_3qtr','bath_half','garb_sqft','gara_sqft']]
df['sale_date']=pd.to_datetime(df['sale_date'])
df = df[df['sale_date']>'2020-01-01']
df = df.rename(columns={'latitude': 'lat', 'longitude': 'lng'})
df = df.h3.geo_to_h3(resolution = 7, lat_col = 'lat', lng_col = 'lng', 
                             set_index = False)
df = df.h3.geo_to_h3(resolution = 10, lat_col = 'lat', lng_col = 'lng', 
                             set_index = False)
df = df.reset_index(drop=True)
df.to_csv('data/sales_2020_25.csv')

# %%
