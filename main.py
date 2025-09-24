#%%
import sys
import os
#%%
sys.path.insert(0,os.getcwd()+'/src/pricemodel')
sys.path.insert(0,os.getcwd()+'/src/spatial')
#sys.path.insert(0,'/Users/marie/PycharmProjects/neural-networks-house-prices/src/pricemodel')
from importlib import reload

#%%
import embedding_model
reload(embedding_model)
from embedding_model import *
#%%
import pandas as pd
df = pd.read_csv('data/sales_202025.csv')
# %% Model Parameters
embedding_dim=8
hidden_dim=8
property_dim=2
# %%


#%%
model = modelmanager()
model.processor(dataset()._prepare_data(df))
model.split_data()

model.train_model(embedding_dim=8, hidden_dim=8, property_dim=2, 
                  epochs = 2, batch = 256, learning_rate = 0.001)

#%%
model.add_predictions_to_data()
model.save_model()
#%%

model_load = modelmanager().load_model_and_artifacts('outputs/models/20250923_164311')
#%%
model_load.processor(dataset()._prepare_data(df), scale_mode = 'load')

#%%
model_load.add_predictions_to_data()
#%%
model_load.dataframe.columns
#%%
import matplotlib.pyplot as plt
train_loss = model.results['train_losses']
val_loss = model.results['val_losses']
x_values = list(range(len(train_loss)))
start_at = 0
plt.plot(x_values[start_at:], train_loss[start_at:], color='b', label='Train Loss')
plt.plot(x_values[start_at:], val_loss[start_at:], color='r', label='Val Loss')
plt.legend()
#%%
# model.save_model()
#
#%%
model.add_predictions_to_data()


#%%
plt.scatter(model.dataset.dataframe['target'],model.dataset.dataframe['predicted_price'])
#%%
plt.scatter(model.dataset.dataframe['sale_price'],model.dataset.dataframe['pct_error'])

# #%%
#
# model.results['feature_importance'][0]['property_features']
# #%%
# for name, param in (model.predictor.model.named_parameters()):
#     if param.requires_grad:
#         print(f"Layer: {name} | Mean: {param.data.mean()} | Std: {param.data.std()}")
# for name, param in (model.predictor.model.named_parameters()):
#     if param.requires_grad and param.grad is not None:
#         print(f"Layer: {name} | Grad Mean: {param.grad.mean()} | Grad Std: {param.grad.std()}")
# for name, param in (model.predictor.model.named_parameters()):
#     print(name, param.requires_grad)
#
# model.add_predictions_to_data()
# data.dataframe.groupby(['year'])['pct_error'].mean()
# data.dataframe.groupby(['year'])['sale_price'].median()
# #model.add_predictions_to_data()
# model.save_model()
#
# analysis = ModelAnalyzer(model.model,device = model.device)
# analysis.visualize_attention_analysis()
# # checking dataframe index matches tensor
# scaler_sqft = data.scalers['sqft']
# scaler_sqft.inverse_transform(data.tensors.tensors[4][:3,0].reshape(1,-1))
# data.dataframe['sqft'].iloc[:3]
#
#
# scaler_price = data.scalers['log_price']
#
# scaler_price.inverse_transform(data.tensors.tensors[5].detach().cpu().numpy().reshape(-1,1))
#
# scaler_price.inverse_transform([4,3])
# # %%

# %%

