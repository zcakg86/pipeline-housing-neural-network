#%%

#from src.pricemodel.embedding_model import *
# need to run
import sys
sys.path.insert(0,'/Users/marie/PycharmProjects/neural-networks-house-prices/src/pricemodel')
from importlib import reload
from embedding_model import *
reload(embedding_model)
import pandas as pd
df = pd.read_csv('data/sales_202025.csv')
df = df[df['lat'].between(47.55,47.65) & df['lng'].between(-122.35,-122.25)]
df = df.sample(n=1000, random_state = 92)

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
model.train_model(epochs = 10, batch = 256)


#model.add_predictions_to_data()
#%%
data2 = data._processor(mode= 'test')
# %%
for i in data.tensors[:][0]:  
    print(data.tensors[i][0])

# %%
for i in model.dataset.tensors[:][0]:  
    print(model.dataset.tensors[i][0])
# %%
