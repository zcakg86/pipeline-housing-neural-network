#%%

#from src.pricemodel.embedding_model import *
# need to run
import sys
sys.path.insert(0,'/Users/marie/PycharmProjects/neural-networks-house-prices/src/pricemodel')
from importlib import reload
from embedding_model import *

import pandas as pd
df = pd.read_csv('data/sales_202025.csv')
df = df[df['lat'].between(47.55,47.65) & df['lng'].between(-122.35,-122.25)]
#df = df.sample(n=1000, random_state = 92)

data = dataset()
data._prepare_data(df)
data._get_community_features()
# Scale and Create tensors
data._processor(scale_mode = 'fit')

embedding_dim=8
hidden_dim=8
property_dim=2

#%%
model = modelmanager(data,embedding_dim, hidden_dim, property_dim)
model.split_data()
model.train_model(epochs = 10, batch = 256, learning_rate = 0.01)
model.add_predictions_to_data()
data.dataframe.groupby(['year'])['pct_error'].mean()
data.dataframe.groupby(['year'])['sale_price'].median()
#model.add_predictions_to_data()
model.save_model()

# checking dataframe index matches tensor
scaler_sqft = data.scalers['sqft']
scaler_sqft.inverse_transform(data.tensors.tensors[4][:3,0].reshape(1,-1))
data.dataframe['sqft'].iloc[:3]


scaler_price = data.scalers['log_price']

scaler_price.inverse_transform(data.tensors.tensors[5].detach().cpu().numpy().reshape(-1,1))

scaler_price.inverse_transform([4,3])