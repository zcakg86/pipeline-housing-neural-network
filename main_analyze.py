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
from modelanalyzer import ModelAnalyzer
#%%
model = '20250805_160603'
model_p = f'outputs/models/{model}/model.pth'
device = torch.device('mps' if torch.mps.is_available()
                                   else 'cuda' if torch.cuda.is_available()
                                   else 'cpu')

#%%
model = modelmanager.load_saved_model()
# %%
