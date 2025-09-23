#%%
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import os
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import pickle
import json


# Function to prepare data
#region dataset class
class dataset:
    def __init__(self, df):
        self.length = None
        self.n_communities = None
        self.year_length = None
        self.week_length = None
        self.scalers = {}
        self.indices = []
        self.directory = 'outputs/models'
        self.timestamp = None
        self.community_df = pd.DataFrame()
        self.community_array = np.empty(0)
        self.community_feature_dim = None
        self.community_indices = torch.empty(0)
        self.year_indices = torch.empty(0)
        self.week_indices = torch.empty(0)
        self.property_features = torch.empty(0)
        self.target = torch.empty(0)
        self.dataframe = df

    def _prepare_data(self):
        # replace references to df with self.dataframe
        df = self.dataframe
        """Expected columns: ['sale_date', 'sale_price', 'lat', 'lng', 'sqft', 'sale_nbr','sqft_lot']"""
        # Convert date to datetime
        df['sale_date'] = pd.to_datetime(df['sale_date'])
        # Sort by date
        df = df.sort_values('sale_date')
        # Need to filter out non-null values
        df = df.dropna(subset=['sale_price', 'lat', 'lng', 'sqft', 'sale_nbr', 'sale_date','sqft_lot'])
        # And Zero values
        df = df[df['sale_price'] > 0]
        df = df[df['sqft'] > 0]
        df = df[df['sale_nbr'] > 0]
        df = df[df['sqft_lot'] > 0]
        # Create derived features
        df['price_per_sqft'] = df['sale_price'] / df['sqft']
        df['month'] = df['sale_date'].dt.month
        df['year'] = df['sale_date'].dt.isocalendar().year
        df['week'] = df['sale_date'].dt.isocalendar().week
        df['log_price']= np.log(df['sale_price'])

        self.week_vocab = {value: index for index, value in enumerate(range(1,54))}
        self.year_vocab = {value: index for index, value in enumerate(range(2020,2026))}

        self.community_vocab = create_vocab(df,'community')
        #community_ids = sorted(df['community'].unique()) # Sort for consistent order across runs
        #self.community_vocab = {community_id: index for index, community_id in enumerate(community_ids)}
        self.n_communities = len(self.community_vocab)
        # Convert community IDs to indices using the vocabulary
        df['community_index'] = df['community'].map(self.community_vocab) # New column with indices
        #df['year_index']
        #df['week_index']
        self.length = df.shape[0]
        self.year_length = len(np.unique(df['year']))
        self.week_length = len(np.unique(df['week']))

        self.dataframe = df.reset_index(drop=True)

        return self
    
    def _get_community_features(self):
        self.community_df = self.dataframe.groupby(['community_index', 'year']).agg({
            'sale_price': ['mean', 'median', 'std'],
            'sqft' : ['mean'],
            'beds' : 'median'
            })

        community_dict = self.community_df.to_dict('index')
        # Flatten the dictionary values
        for key, value in community_dict.items():
            community_dict[key] = [v for sublist in value.values() for v in (sublist if isinstance(sublist, list) else [sublist])]
        self.community_array = np.array([community_dict[(c, y)] for c, y in zip(self.dataframe['community_index'], self.dataframe['year'])])
        self.community_feature_dim = self.community_array.shape[1]

        return self

    def _processor(self, scale_mode = "fit"):
        """ Function transform and construct TensorDataset from dataframe features
            Parameters:
                scale_mode (str): If scalers need to be fit ("fit") on data or read from file for transformation only.
        """
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.directory = os.path.join(self.directory, self.timestamp)
        os.makedirs(self.directory,exist_ok = True)
        for feature in ['sqft','sqft_lot','log_price']: # List the features to scale
            if scale_mode == "fit":
                self.scalers[feature] = StandardScaler() # Create a new scaler for each feature
                self.dataframe[f"{feature}_scaled"] = self.scalers[feature].fit_transform(self.dataframe[[feature]]) # Fit and transform
                # Save the scaler
                print(f"{feature}_scaled")
                print(f"{feature}_scaled" in self.dataframe.columns)
                joblib.dump(self.scalers[feature], os.path.join(self.directory, f"{feature}_scaler.pkl"))
            # else: 
            #     # Load the pre-fitted scaler
                self.scalers[feature] = joblib.load(os.path.join(self.directory, f"{feature}_scaler.pkl"))
            #     self.dataframe[feature] = scaler.transform(self.dataframe[[feature]])
                self.dataframe[f"{feature}_scaled"] = self.scalers[feature].transform(self.dataframe[[feature]]) # Fit and transform

        # # Community df
        # if scale_mode == "fit":
        #     self.scalers['community'] = StandardScaler()
        #     self.community_array = self.scalers[feature].fit_transform(self.community_array)
        #     joblib.dump(self.scalers['community'], os.path.join(self.directory, "communities_scaler.pkl"))

        # else: 
        #     self.scalers['community'] = joblib.load(os.path.join(self.directory, "communities_scaler.pkl"))
        #     self.community_array = self.scalers.transform(self.community_array)

        # Create tensor with each observation being contiguous, and scale fields.
        self.tensors = TensorDataset(torch.tensor(self.dataframe['community_index'].values, dtype=torch.int),
                                     #torch.tensor(self.community_array,dtype = torch.float32),
                                     torch.tensor(self.dataframe['year'].values, dtype=torch.int),
                                     torch.tensor(self.dataframe['week'].values, dtype=torch.int),
                                     torch.tensor(self.dataframe[['sqft_scaled','sqft_lot_scaled']].values, dtype=torch.float32),
                                     torch.tensor(self.dataframe['log_price_scaled'].values, dtype=torch.float32))
        
        return self
#endregion
#region Model init
import torch
import torch.nn as nn
class EmbeddingModelEnhanced(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, property_dim,
                 community_embedding_length, 
                 year_length, week_length, monitor_attention=True):
        """Enhanced model with proper attention layer initialization and weight tracking"""
        super().__init__()

        # Whether to calculate and store attention
        self.monitor_attention = monitor_attention
        # Embedding Layers
        self.community_embedding = nn.Embedding(int(community_embedding_length), embedding_dim)
        self.year_embedding = nn.Embedding(int(year_length), embedding_dim)
        self.week_embedding = nn.Embedding(int(week_length), embedding_dim)
        # Feature Processing Layers
        self.property_feature_layer = nn.Linear(property_dim, embedding_dim)
        
        # Learnable CLS token (shape: (1, 1, E))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        # Initialize attention layer
        self.attention_layer = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=2,
            batch_first=True)

        # Hidden and Output Layers
        self.hidden_layer1 = nn.Linear(embedding_dim, hidden_dim)
        self.hidden_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)

        self.relu = nn.ReLU()

        # Storage for attention weights and intermediate outputs
        self.last_attention_weights = None
        self.intermediate_outputs = {}
        self.save_intermediates = False

#region Model forward
    def forward(self, community_indices, year, week, property_features):
        # Clear previous intermediate outputs
        if self.save_intermediates:
            self.intermediate_outputs = {}

        need_w = self.monitor_attention  # only compute when needed

        # Embeddings
        community_embeddings = self.community_embedding(community_indices)
        year_embeddings = self.year_embedding(year)
        week_embeddings = self.week_embedding(week)
                                  
        if self.save_intermediates:
            self.intermediate_outputs['community_embeddings'] = community_embeddings.detach()
            self.intermediate_outputs['year_embeddings'] = year_embeddings.detach()
            self.intermediate_outputs['week_embeddings'] = week_embeddings.detach()

        # Feature Processing
        #processed_community_features = self.relu(self.community_feature_layer(community_features))
        processed_property_features = self.relu(self.property_feature_layer(property_features))

        if self.save_intermediates:
        #    self.intermediate_outputs['processed_community_features'] = processed_community_features.detach()
            self.intermediate_outputs['processed_property_features'] = processed_property_features.detach()


        # Stack embedding and property layers
        tokens = torch.stack([community_embeddings, year_embeddings, week_embeddings, processed_property_features],
                             dim = 1)
        # Prepend CLS (B, 1, E) -> (B, 5, E)
        B = tokens.size(0)
        cls = self.cls_token.expand(B, -1, -1)  # expand along batch
        seq = torch.cat([cls, tokens], dim=1)   # [CLS, community, year, week, property]

        # Attention Layer with weight extraction
        attention_output, attention_weights = self.attention_layer(
            seq, seq, seq,
            need_weights=True, average_attn_weights=False
        )

        # Use only CLS output to make prediction
        cls_out = attention_output[:, 0, :]  # (B, E)

        # Store attention weights (optional)
        if need_w:
            # Full attention (including CLS)
            self.last_attention_weights = attention_weights.detach()
            # CLS row attending to all tokens (including itself)
            # attn_w: (B, H, T+1, T+1). We want row 0 (CLS as query).
            # Usually you’ll interpret the CLS attention to the non-CLS tokens:
            self.last_cls_attention = attention_weights[:, :, 0, 1:].detach()  # (B, H, 4)
        else:
            self.last_attention_weights = None
            self.last_cls_attention = None

        # Hidden Layers
        # MLP head
        h1 = self.relu(self.hidden_layer1(cls_out))
        h2 = self.relu(self.hidden_layer2(h1))
        output  = self.output_layer(h2)  # (B, 1)

        if self.save_intermediates:
            self.intermediate_outputs['tokens'] = tokens.detach()
            self.intermediate_outputs['seq_with_cls'] = seq.detach()
            self.intermediate_outputs['attn_out'] = attention_output.detach()
            if need_w:
                self.intermediate_outputs['attn_w'] = attention_weights.detach()
        
        return output
        
#region predictor
class price_predictor:
    def __init__(self, device, embedding_dim, hidden_dim, property_dim, community_embedding_length,
                 year_length, week_length, learning_rate):
        self.device = device
        self.model = EmbeddingModelEnhanced(embedding_dim, hidden_dim, property_dim,
                                    community_embedding_length, 
                                    year_length, week_length, monitor_attention=True).to(device)
        # Specify loss measure
        self.criterion = nn.MSELoss()
        # And Adam optimiser
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=learning_rate)
    def eval(self):
        self.model.eval()
#region training loop
    def train(self, train_loader, val_loader, epochs, analyze_every=10):
        train_losses = []
        val_losses = []
        attention_evolution = []
        feature_importance = []
        # analyzer = ModelAnalyzer(self.model, self.device)
        # analyzer.hook_attention_weights()

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            # with torch.autograd.detect_anomaly():
            for batch in train_loader:
                # Move each tensor in the batch to the device
                batch = tuple(t.to(self.device) for t in batch)
                # Unpack the batch
                community,  year, week, property, targets = batch
                self.optimizer.zero_grad()

                predictions = self.model(community, year,
                                            week, property)

                if torch.isnan(predictions).any():
                    print(f"{torch.isnan(predictions).sum().item()} NaN values detected in outputs out of {torch.numel(predictions)}. Skipping this iteration.")
                    continue

                loss = self.criterion(predictions.squeeze(), targets)
                # print(f'Train Community Indices Min: {community.min().item()} and ',
                #       f'Max: {community.max().item()}')
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    # Move each tensor in the batch to the device
                    batch = tuple(t.to(self.device) for t in batch)
                    # Unpack the batch
                    community, year, week, property, targets = batch
                    # print(f'Val Community Indices Min: {community.min().item()} and ',
                    #       f'Max: {community.max().item()}')
                    predictions = self.model(community, year, week, property)
                    loss = self.criterion(predictions.squeeze(), targets)  # Fixed: calculate loss here

                    val_loss += loss.item()

            train_losses.append(train_loss / len(train_loader))
            val_losses.append(val_loss / len(val_loader))

            # if (epoch+1)%analyze_every == 0:
            #     attention_stats = analyzer.analyze_attention_patterns(val_loader, num_batches=5)
            #     attention_evolution.append({
            #         'epoch': epoch+1}|attention_stats)
            #     print(f'Epoch [{epoch+1}/{epochs}], '
            #       f'Train Loss: {train_losses[-1]:.4f}, '
            #       f'Val Loss: {val_losses[-1]:.4f}, '
            #       f'Mean attention weight: {np.mean(attention_stats["mean_weights"]):.4f}')
            #     feature_importance.append(analyzer.compute_feature_importance_gradients(val_loader, num_batches=5))
            #     print(self.model.intermediate_outputs.get('attention_output'))

            # else:
            print(f'Epoch [{epoch+1}/{epochs}], '
                f'Train Loss: {train_losses[-1]:.4f}, '
                f'Val Loss: {val_losses[-1]:.4f}')
        return train_losses, val_losses #, feature_importance, attention_evolution
    
#region Manager
class modelmanager:
    def __init__(self, dataset, embedding_dim, hidden_dim, property_dim, model_name="property_model"):

        self.dataset = dataset
        self.device = torch.device('mps' if torch.mps.is_available()
                                   else 'cuda' if torch.cuda.is_available()
                                   else 'cpu')
        self.model_name = model_name
        self.results = {
            'train_losses': [],
            'val_losses': [],
            'metrics': {},
            'attention_evolution':{},
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.property_dim = property_dim
        self.n_communities = self.dataset.n_communities
        self.community_feature_dim = self.dataset.community_feature_dim
        self.week_length = self.dataset.week_length
        self.year_length = self.dataset.year_length
        self.train_year_length = 0
        self.train_week_length = 0
        self.week_vocab = None
        self.year_vocab = None

    def split_data_and_index(self):
        """Function splits dataset for training and validation, applies vocab with data present in the training dataset,
            to create index to be used in embedding"""
        # Split data, and create DataLoader for batches.
        # Sizes from model attributes.
        train_size = int(0.8 * self.dataset.length)
        val_size = self.dataset.length - train_size

        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            self.dataset.tensors, [train_size, val_size]
        )
        # remember tensor order :
        # community, year, week, property, targets
        self.community_embedding_length = self.dataset.tensors[:][0].unique().numel()

        self.year_vocab = create_tensor_vocab(self.train_dataset[:][1])
        self.week_vocab = create_tensor_vocab(self.train_dataset[:][2])

        self.train_year_length = len(self.year_vocab)
        self.train_week_length = len(self.week_vocab)

        year_train_tensor = vocab_replace_tensor(self.train_dataset.dataset.tensors[:][1], self.year_vocab)
        year_val_tensor = vocab_replace_tensor(self.val_dataset.dataset.tensors[:][1], self.year_vocab)

        week_train_tensor = vocab_replace_tensor(self.train_dataset.dataset.tensors[:][2], self.week_vocab)
        week_val_tensor = vocab_replace_tensor(self.val_dataset.dataset.tensors[:][2], self.week_vocab)

        # Create a new TensorDataset with the updated tensors
        new_train_dataset = list(self.train_dataset.dataset.tensors)  # Convert tuple to list
        new_train_dataset[1] = year_train_tensor  # Replace the old tensor with the updated one
        new_train_dataset[2] = week_train_tensor # Same for weeks
        new_train_dataset = TensorDataset(*new_train_dataset)  # Create new TensorDataset
        self.train_dataset = Subset(new_train_dataset, self.train_dataset.indices)  # Use the original indices from the random_split

        new_val_dataset = list(self.val_dataset.dataset.tensors)
        new_val_dataset[1] = year_val_tensor  # Replace the old tensor with the updated one
        new_val_dataset[2] = week_val_tensor
        new_val_dataset = TensorDataset(*new_val_dataset)  # Create new TensorDataset
        self.val_dataset = Subset(new_val_dataset, self.val_dataset.indices)  # Use the original indices

    def train_model(self, epochs=10, batch=128, learning_rate = 0.01, analyze_every=10):
        """Function to create final DataLoader and run model training"""
        train_loader = DataLoader(self.train_dataset, batch_size=batch, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=batch, drop_last = True)

        # Create and train model. price_predictor contains model spec.
        self.predictor = price_predictor(self.device, self.embedding_dim, self.hidden_dim, self.property_dim,
                                    self.community_embedding_length, 
                                    self.train_year_length,
                                    self.train_week_length,
                                    learning_rate)

        train_losses, val_losses = self.predictor.train(train_loader, val_loader, epochs = epochs,
                                                                            analyze_every=analyze_every)
        self.results['train_losses'] = train_losses
        self.results['val_losses'] = val_losses

    def add_predictions_to_data(self):
        """Predict with model and add to dataframe"""

        year_tensor = vocab_replace_tensor(self.dataset.tensors.tensors[1], self.year_vocab)
        week_tensor = vocab_replace_tensor(self.dataset.tensors.tensors[2], self.week_vocab)

        # Create a new TensorDataset with the updated tensors
        new_dataset = list(self.dataset.tensors.tensors)  # Convert tuple to list

        new_dataset[1] = year_tensor  # Replace the old tensor with the updated one
        new_dataset[2] = week_tensor # Same for weeks

        new_dataset = TensorDataset(*new_dataset)  # Create new TensorDataset

        # Create DataLoader for prediction
        loader = DataLoader(new_dataset, batch_size=256)

        self.predictor.eval()
        current_idx = 0

        predictions = []
        prediction_indices = []
        target = []
        target_indices = []
        cls_output = []
        with torch.no_grad():
            for batch in loader:
                # Move each tensor in the batch to the device
                # print(f'Batch size {batch[0].size().numel()}')
                batch = tuple(t.to(self.predictor.device) for t in batch)
                # Unpack the batch
                community, year, week, property, targets = batch

                pred = self.predictor.model(community, year, week, property)
                batch_predictions = pred.cpu().numpy() 
                batch_targets = targets.cpu().numpy()
                batch_cls = self.predictor.model.last_cls_attention.mean(dim=1).cpu().numpy()


                for i, p in enumerate(batch_predictions):
                        predictions.append(p)
                        prediction_indices.append(current_idx + i)

                for i, p in enumerate(batch_targets):
                    target.append(p)
                    target_indices.append(current_idx + i)

                for i, p in enumerate(batch_cls):
                    cls_output.append(p)
                # add len of current batch for next one
                current_idx += len(community)

        # Reshape predictions
        predictions = np.array(predictions).reshape(-1, 1)
        # and targets (for QA)
        target = np.array(target).reshape(-1, 1)

        #scaler_path = os.path.join(self.dataset.directory, "log_price_scaler.pkl")
        scaler = self.dataset.scalers['log_price']

        predicted_log_price = scaler.inverse_transform(predictions).ravel()
        target_log_price = scaler.inverse_transform(target).ravel()
        cls_labels = ["cls_community", "cls_year", "cls_week", "cls_property"]
        self.dataset.dataframe.loc[target_indices,cls_labels] = cls_output

        # initialise dataframe columns
        self.dataset.dataframe['predicted_value'] = pd.Series(dtype=float)
        self.dataset.dataframe['predicted_price'] = pd.Series(dtype=float)
        self.dataset.dataframe['pct_error'] = pd.Series(dtype=float)
        self.dataset.dataframe['target_log'] = pd.Series(dtype=float)
        self.dataset.dataframe['target'] = pd.Series(dtype=float)

        self.dataset.dataframe.iloc[target_indices, self.dataset.dataframe.columns.get_loc('predicted_value')] = predicted_log_price

        self.dataset.dataframe.iloc[target_indices, self.dataset.dataframe.columns.get_loc('target_log')] = target_log_price        
        self.dataset.dataframe.iloc[target_indices, self.dataset.dataframe.columns.get_loc('target')] = np.exp(target_log_price)

        self.dataset.dataframe.iloc[prediction_indices, self.dataset.dataframe.columns.get_loc('predicted_price')] = np.exp(predicted_log_price)

        self.dataset.dataframe['price_error']= self.dataset.dataframe['sale_price']-self.dataset.dataframe['predicted_price']
        self.dataset.dataframe['pct_error']=self.dataset.dataframe['price_error']/self.dataset.dataframe['sale_price']

    def save_model(self):
        """Save model, config, processor, and results"""
        import os
        path = self.dataset.directory
        os.makedirs(path, exist_ok=True)
        config = {}
        for name, module in self.predictor.model.named_modules():
            params = {}
            for param_name, param in module.named_parameters(recurse=False):
                params[param_name] = [param.shape]
            config[name] = params
        # Save model state
        torch.save({
            'model_state_dict': self.predictor.model.state_dict(),
            'optimizer_state_dict': self.predictor.optimizer.state_dict(),
            'model_config': config,
            'results': self.results
        }, f'{self.dataset.directory}/model.pth')

        with open(f'{self.dataset.directory}/community_vocab.json', 'w') as f:
            json.dump(self.dataset.community_vocab, f)

        # Save results separately as JSON
        with open(f'{self.dataset.directory}/results.json', 'w') as f:
            json.dump(self.results, f)

          # Save config as JSON
        with open(f'{self.dataset.directory}/config.json', 'w') as f:
            json.dump(config, f)

        # Save processor (scalers and parameters)
        with open(f'{self.dataset.directory}/week_vocab.json', 'w') as f:
            json.dump(self.week_vocab, f)
        # Save processor (scalers and parameters)
        with open(f'{self.dataset.directory}/year_vocab.json', 'w') as f:
            json.dump(self.year_vocab, f)


        print(f"Model and results saved in {path}")

def create_vocab(df, column, min=None, max=None):
    # if min && max:
    ids = sorted(df[column].unique())
    vocab = {int(id): index for index, id in enumerate(ids)}
    return vocab

def create_tensor_vocab(tensor):
    values = sorted(tensor.unique().tolist())
    vocab = {year: idx for idx, year in enumerate(values)}
    # add unknown when value for when not in training data
    vocab["unknown"] = len(vocab)  # Add "unknown" token
    return vocab

def vocab_replace_tensor(tensor, vocab):
    replaced = [vocab.get(value.item(), vocab['unknown']) for value in tensor]
    return torch.tensor(replaced, dtype=torch.int)
