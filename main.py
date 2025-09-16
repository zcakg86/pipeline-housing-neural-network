#%%
import sys
sys.path.insert(0,'src/pricemodel')
#sys.path.insert(0,'/Users/marie/PycharmProjects/neural-networks-house-prices/src/pricemodel')

from importlib import reload

#%%
import embedding_model
import embedding_new
import modelanalyzer
reload(embedding_model)
reload(embedding_new)
reload(modelanalyzer)
from embedding_model import *
from embedding_new import *
from modelanalyzer import *
#%%
import pandas as pd
df = pd.read_csv('data/sales_202025.csv')
df = df[df['lat'].between(47.55,47.65) & df['lng'].between(-122.35,-122.25)]
df = df.sample(n=1000)

data = dataset()
data._prepare_data(df)
data._get_community_features()
#%%

# Scale and Create tensors
data._processor(scale_mode = 'fit')
#%%
embedding_dim=8
hidden_dim=8
property_dim=2

#%%
#%%
model = modelmanager(data,embedding_dim, hidden_dim, property_dim)
model.split_data()

model.train_model(epochs = 100, batch = 128, learning_rate = 0.001, analyze_every=5)
#%%
model.save_model()

#%%
model.add_predictions_to_data()
#%%

model.results['feature_importance'][0]['property_features']
#%%
for name, param in (model.predictor.model.named_parameters()):
    if param.requires_grad:
        print(f"Layer: {name} | Mean: {param.data.mean()} | Std: {param.data.std()}")
for name, param in (model.predictor.model.named_parameters()):
    if param.requires_grad and param.grad is not None:
        print(f"Layer: {name} | Grad Mean: {param.grad.mean()} | Grad Std: {param.grad.std()}")
for name, param in (model.predictor.model.named_parameters()):
    print(name, param.requires_grad)

model.add_predictions_to_data()
data.dataframe.groupby(['year'])['pct_error'].mean()
data.dataframe.groupby(['year'])['sale_price'].median()
#model.add_predictions_to_data()
model.save_model()

analysis = ModelAnalyzer(model.model,device = model.device)
analysis.visualize_attention_analysis()
# checking dataframe index matches tensor
scaler_sqft = data.scalers['sqft']
scaler_sqft.inverse_transform(data.tensors.tensors[4][:3,0].reshape(1,-1))
data.dataframe['sqft'].iloc[:3]


scaler_price = data.scalers['log_price']

scaler_price.inverse_transform(data.tensors.tensors[5].detach().cpu().numpy().reshape(-1,1))

scaler_price.inverse_transform([4,3])
# %%
