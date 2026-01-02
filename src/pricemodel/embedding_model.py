import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import os
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import json

#region dataset class
class dataset:
    """
    Handles data ingestion, cleaning, feature engineering, and vocabulary creation
    for categorical variables.
    """
    def __init__(self):
        """Initializes the dataset object with empty placeholders for data and attributes."""
        self.length = None
        self.n_communities = None
        self.year_length = None
        self.week_length = None
        self.scalers = {}
        self.indices = []
        self.timestamp = None
        self.community_df = pd.DataFrame()
        self.community_array = np.empty(0)
        self.community_feature_dim = None
        self.community_indices = torch.empty(0)
        self.year_indices = torch.empty(0)
        self.week_indices = torch.empty(0)
        self.property_features = torch.empty(0)
        self.target = torch.empty(0)

    def _prepare_data(self, dataframe):
        """
        Cleans the raw dataframe, creates derived time/price features, and generates 
        vocabularies for categorical mapping (Community, Year, Week).

        Parameters:
            dataframe (pd.DataFrame): Raw input data containing sales info.

        Returns:
            self: Returns the instance with the processed dataframe (`self.dataframe`) 
                  and vocabulary attributes populated.
        """
        # replace references to df with self.dataframe
        df = dataframe
        """Expected columns: ['sale_date', 'sale_price', 'lat', 'lng', 'sqft', 'sale_nbr','sqft_lot']"""
        
        # --- Data Cleaning & Filtering ---
        # Convert date column to datetime objects
        df['sale_date'] = pd.to_datetime(df['sale_date'])
        # Sort chronologically
        df = df.sort_values('sale_date')
        # Remove rows with missing essential values
        df = df.dropna(subset=['sale_price', 'lat', 'lng', 'sqft', 'sale_nbr', 'sale_date','sqft_lot'])
        # Remove invalid zero-value entries
        df = df[df['sale_price'] > 0]
        df = df[df['sqft'] > 0]
        df = df[df['sale_nbr'] > 0]
        df = df[df['sqft_lot'] > 0]

        # --- Feature Engineering ---
        # Calculate price per square foot
        df['price_per_sqft'] = df['sale_price'] / df['sqft']
        # Extract temporal features from the date
        df['month'] = df['sale_date'].dt.month
        df['year'] = df['sale_date'].dt.isocalendar().year
        df['week'] = df['sale_date'].dt.isocalendar().week
        # Log-transform the target variable (price) to normalize distribution
        df['log_price']= np.log(df['sale_price'])

        # --- Vocabulary Creation ---
        # specific hardcoded ranges for weeks (1-53) and years (2020-2025)
        self.week_vocab = {value: index for index, value in enumerate(range(1,54))}
        self.year_vocab = {value: index for index, value in enumerate(range(2020,2026))}
        
        # Add 'unknown' token for handling out-of-distribution data later
        self.week_vocab["unknown"] = len(self.week_vocab)
        self.year_vocab["unknown"] = len(self.year_vocab)

        # Create vocabulary for communities dynamically from data
        self.community_vocab = create_vocab(df,'community')
        self.community_vocab["unknown"] = len(self.community_vocab )
        
        self.n_communities = len(self.community_vocab)
        
        # Map categorical IDs to integer indices
        df['community_index'] = df['community'].map(self.community_vocab) 

        # Store dataset dimensions
        self.length = df.shape[0]
        self.year_length = len(np.unique(df['year']))
        self.week_length = len(np.unique(df['week']))

        self.dataframe = df.reset_index(drop=True)

        return self
    
    def _get_community_features(self):
        """
        Aggregates statistics (mean, median, std) for sales within specific 
        communities and years to create dense feature vectors for communities.
        
        Returns:
            self: Updates self.community_array with the aggregated features.
        """
        self.community_df = self.dataframe.groupby(['community_index', 'year']).agg({
            'sale_price': ['mean', 'median', 'std'],
            'sqft' : ['mean'],
            'beds' : 'median'
            })

        community_dict = self.community_df.to_dict('index')
        # Flatten the dictionary values into a list for array conversion
        for key, value in community_dict.items():
            community_dict[key] = [v for sublist in value.values() for v in (sublist if isinstance(sublist, list) else [sublist])]
        
        # Map aggregated features back to the main dataframe order
        self.community_array = np.array([community_dict[(c, y)] for c, y in zip(self.dataframe['community_index'], self.dataframe['year'])])
        self.community_feature_dim = self.community_array.shape[1]

        return self

#endregion
#region Model init

class EmbeddingModel(nn.Module):
    """
    Transformer-based neural network for tabular data. 
    Uses embeddings for categorical features, projects numerical features,
    and applies Self-Attention via a CLS token to predict prices.
    """
    def __init__(self, embedding_dim, hidden_dim, property_dim,
                 community_embedding_length, 
                 year_length, week_length, monitor_attention=True):
        """
        Initialize the model layers.

        Parameters:
            embedding_dim (int): Size of the latent space for all tokens.
            hidden_dim (int): Size of the hidden layer in the final MLP.
            property_dim (int): Number of numerical property features (e.g., sqft).
            community_embedding_length (int): Size of community vocabulary.
            year_length (int): Size of year vocabulary.
            week_length (int): Size of week vocabulary.
            monitor_attention (bool): Whether to store attention weights for analysis.
        """
        super().__init__()

        # Whether to calculate and store attention
        self.monitor_attention = monitor_attention
        
        # --- Embedding Layers (Categorical) ---
        self.community_embedding = nn.Embedding(int(community_embedding_length), embedding_dim)
        self.year_embedding = nn.Embedding(int(year_length), embedding_dim)
        self.week_embedding = nn.Embedding(int(week_length), embedding_dim)
        
        # --- Feature Projection (Numerical) ---
        # Projects numerical features to match the embedding dimension
        self.property_feature_layer = nn.Linear(property_dim, embedding_dim)
        
        # --- Learnable CLS Token ---
        # A generic token prepended to the sequence to aggregate global context
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        
        # --- Attention Mechanism ---
        self.attention_layer = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=2,
            batch_first=True)

        # --- Regressor Head (MLP) ---
        self.hidden_layer1 = nn.Linear(embedding_dim, hidden_dim)
        self.hidden_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)

        self.relu = nn.ReLU()

        # Storage for attention weights and intermediate outputs (debugging/analysis)
        self.last_attention_weights = None
        self.intermediate_outputs = {}
        self.save_intermediates = False

#region Model forward
    def forward(self, community_indices, year, week, property_features):
        """
        Forward pass of the network.

        Steps:
        1. Embed categorical indices.
        2. Project numerical features.
        3. Stack all embeddings to form a sequence.
        4. Prepend CLS token.
        5. Apply Self-Attention.
        6. Extract CLS token output and pass through MLP for prediction.

        Parameters:
            community_indices (Tensor): Batch of community IDs.
            year (Tensor): Batch of year IDs.
            week (Tensor): Batch of week IDs.
            property_features (Tensor): Batch of numerical features.

        Returns:
            Tensor: Predicted values (batch_size, 1).
        """
        # Clear previous intermediate outputs
        if self.save_intermediates:
            self.intermediate_outputs = {}

        need_w = self.monitor_attention  # only compute when needed

        # --- 1. Embed Inputs ---
        community_embeddings = self.community_embedding(community_indices)
        year_embeddings = self.year_embedding(year)
        week_embeddings = self.week_embedding(week)
                                  
        if self.save_intermediates:
            self.intermediate_outputs['community_embeddings'] = community_embeddings.detach()
            self.intermediate_outputs['year_embeddings'] = year_embeddings.detach()
            self.intermediate_outputs['week_embeddings'] = week_embeddings.detach()

        # --- 2. Process Numerical Features ---
        #processed_community_features = self.relu(self.community_feature_layer(community_features))
        processed_property_features = self.relu(self.property_feature_layer(property_features))

        if self.save_intermediates:
        #    self.intermediate_outputs['processed_community_features'] = processed_community_features.detach()
            self.intermediate_outputs['processed_property_features'] = processed_property_features.detach()


        # --- 3. Stack Sequence ---
        tokens = torch.stack([community_embeddings, year_embeddings, week_embeddings, processed_property_features],
                             dim = 1)
        
        # --- 4. Prepend CLS Token ---
        # Prepend CLS (B, 1, E) -> Result (B, 5, E)
        B = tokens.size(0)
        cls = self.cls_token.expand(B, -1, -1)  # expand along batch
        seq = torch.cat([cls, tokens], dim=1)   # [CLS, community, year, week, property]


        # --- 5. Self-Attention ---
        attention_output, attention_weights = self.attention_layer(
            seq, seq, seq,
            need_weights=True, average_attn_weights=False
        )

        # Use only CLS output to make prediction.
        # We discard the feature tokens here; CLS has aggregated their info.
        cls_out = attention_output[:, 0, :]  # (B, E)

        # Store attention weights (optional for analysis)
        if need_w:
            # Full attention (including CLS)
            self.last_attention_weights = attention_weights.detach()
            # Extract specifically how CLS attended to other tokens
            # attn_w: (B, H, T+1, T+1). We want row 0 (CLS as query).
            self.last_cls_attention = attention_weights[:, :, 0, 1:].detach()  # (B, H, 4)
        else:
            self.last_attention_weights = None
            self.last_cls_attention = None

        # --- 6. MLP Head ---
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
        
#region Predictor class
class price_predictor:
    """
    Wrapper class that handles the model instantiation, optimizer setup, 
    and the training loop logic.
    """
    def __init__(self, device, embedding_dim, hidden_dim, property_dim, community_embedding_length,
                 year_length, week_length, learning_rate):
        """
        Initializes the model, loss function, and optimizer.

        Parameters:
            device (torch.device): CPU or GPU.
            embedding_dim, hidden_dim, property_dim, etc: Model architecture parameters.
            learning_rate (float): Step size for the optimizer.
        """
        self.device = device
        self.model = EmbeddingModel(embedding_dim, hidden_dim, property_dim,
                                    community_embedding_length, 
                                    year_length, week_length, monitor_attention=True).to(device)
        # Specify loss measure (Mean Squared Error for regression)
        self.criterion = nn.MSELoss()
        # And Adam optimiser
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=learning_rate)
    
    def eval(self):
        """Sets the model to evaluation mode (disables dropout/batchnorm updates)."""
        self.model.eval()

#region Training loop
    def train(self, train_loader, val_loader, epochs, analyze_every=10):
        """
        Executes the training loop over specified epochs.

        Parameters:
            train_loader (DataLoader): Batched training data.
            val_loader (DataLoader): Batched validation data.
            epochs (int): Number of passes through the dataset.
            analyze_every (int): Frequency of analysis (unused in current logic but reserved).

        Returns:
            tuple: (train_losses, val_losses) - Lists of average loss per epoch.
        """
        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            # --- Training Phase ---
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

            # --- Validation Phase ---
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

            print(f'Epoch [{epoch+1}/{epochs}], '
                f'Train Loss: {train_losses[-1]:.4f}, '
                f'Val Loss: {val_losses[-1]:.4f}')
        return train_losses, val_losses #, feature_importance, attention_evolution
    
#region Model Manager class
class modelmanager:
    """
    Orchestrator class for the entire pipeline.
    Handles data processing, splitting, training execution, results logging, 
    and artifact saving/loading.
    """
    def __init__(self, model_name="property_model"):
        """Initializes directories and sets up the computing device."""
        self.device = torch.device('mps' if torch.mps.is_available()
                                   else 'cuda' if torch.cuda.is_available()
                                   else 'cpu')
        self.model_name = model_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.directory_prefix = 'outputs/models/'
        self.directory = os.path.join(self.directory_prefix, self.timestamp)
        self.results = {
            'train_losses': [],
            'val_losses': [],
            'metrics': {},
            'attention_evolution':{},
            'timestamp': self.timestamp
        }
        self.train_year_length = 0
        self.train_week_length = 0
        self.week_vocab = None
        self.year_vocab = None
        self.community_vocab = None
        self.scalers = {}
        self.predictor = None
        os.makedirs(self.directory,exist_ok = True)

# region data processor
    def processor(self, data, scale_mode = "fit"):
        """ 
        Transforms dataframe features into PyTorch tensors and scales numerical data.
        Priority: Use self.scalers -> Load from Disk -> Fit New

        Parameters:
            data (dataset): The dataset object containing the raw dataframe.
            scale_mode (str): "fit" to calculate new scaling statistics, 
                              or "transform" to use loaded scalers.

        Returns:
            self: Updated with processed TensorDataset and scalers.
        """

        
        features_to_scale = ['sqft', 'sqft_lot', 'log_price','beds']

        for feature in features_to_scale:
            scaler_path = os.path.join(self.directory, f"{feature}_scaler.pkl")
            scaler = None

            # 1. Priority: Check if we already have this scaler in memory
            if feature in self.scalers:
                # print(f"Using in-memory scaler for {feature}")
                scaler = self.scalers[feature]

            # 2. Priority: Check if it exists on disk (unless we explicitly want to force a re-fit)
            elif os.path.exists(scaler_path) and scale_mode != "force_new":
                # print(f"Loading scaler for {feature} from disk")
                scaler = joblib.load(scaler_path)
                self.scalers[feature] = scaler # Save to memory for next time

            # Apply Logic
            if scaler is not None:
                # TRANSFORM ONLY (Use existing math)
                data.dataframe[f"{feature}_scaled"] = scaler.transform(data.dataframe[[feature]])
            else:
                # FIT & TRANSFORM (Learn math from this data)
                # print(f"Fitting new scaler for {feature}")
                scaler = StandardScaler()
                data.dataframe[f"{feature}_scaled"] = scaler.fit_transform(data.dataframe[[feature]])
                
                # Save for future use
                self.scalers[feature] = scaler
                joblib.dump(scaler, scaler_path)

        # Create tensor with each observation being contiguous, and scale fields.
        self.tensors = TensorDataset(torch.tensor(data.dataframe['community_index'].values, dtype=torch.int),
                                     #torch.tensor(self.community_array,dtype = torch.float32),
                                     torch.tensor(data.dataframe['year'].values, dtype=torch.int),
                                     torch.tensor(data.dataframe['week'].values, dtype=torch.int),
                                     torch.tensor(data.dataframe[['sqft_scaled','sqft_lot_scaled','beds_scaled']].values, dtype=torch.float32),
                                     torch.tensor(data.dataframe['log_price_scaled'].values, dtype=torch.float32))
        self.dataframe = data.dataframe
        self.year_length = data.year_length
        self.week_length = data.week_length
        self.n_communities = data.n_communities
        self.data_length = data.length
        self.community_vocab = data.community_vocab
        self.week_vocab = data.week_vocab
        self.year_vocab = data.year_vocab

        return self
    
# region data splitting
    def split_data(self):
        """
        Splits the TensorDataset into training (80%) and validation (20%) sets.
        Also updates the tensor data to ensure vocab consistency (handling unknowns).
        """
        # Split data, and create DataLoader for batches.
        # Sizes from model attributes.
        self.tensor_length = self.tensors.tensors[0].shape[0]

         # 80/20 split
        train_size = int(0.8 * self.tensor_length)
        val_size = self.tensor_length - train_size

        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            self.tensors, [train_size, val_size]
        )
        # keep tensor order:
        # community, year, week, property, targets
        
        # Ensure vocab consistency on the split tensors
        year_train_tensor = vocab_replace_tensor(self.train_dataset.dataset.tensors[:][1], self.year_vocab)
        year_val_tensor = vocab_replace_tensor(self.val_dataset.dataset.tensors[:][1], self.year_vocab)

        week_train_tensor = vocab_replace_tensor(self.train_dataset.dataset.tensors[:][2], self.week_vocab)
        week_val_tensor = vocab_replace_tensor(self.val_dataset.dataset.tensors[:][2], self.week_vocab)

        # Create a new TensorDataset with the updated tensors for Training
        new_train_dataset = list(self.train_dataset.dataset.tensors)  # Convert tuple to list
        new_train_dataset[1] = year_train_tensor  # Replace the old tensor with the updated one
        new_train_dataset[2] = week_train_tensor # Same for weeks
        new_train_dataset = TensorDataset(*new_train_dataset)  # Create new TensorDataset
        self.train_dataset = Subset(new_train_dataset, self.train_dataset.indices)  # Use the original indices from the random_split

        # Create a new TensorDataset with the updated tensors for Validation
        new_val_dataset = list(self.val_dataset.dataset.tensors)
        new_val_dataset[1] = year_val_tensor  # Replace the old tensor with the updated one
        new_val_dataset[2] = week_val_tensor
        new_val_dataset = TensorDataset(*new_val_dataset)  # Create new TensorDataset
        self.val_dataset = Subset(new_val_dataset, self.val_dataset.indices)  # Use the original indices
# region train method
    def train_model(self, embedding_dim, hidden_dim, property_dim,
                    epochs=10, batch=128, learning_rate = 0.01, analyze_every=10):
        """
        Configures DataLoaders and initiates the training process via the predictor.

        Parameters:
            embedding_dim (int): Latent dimension size.
            hidden_dim (int): Hidden layer size.
            property_dim (int): Number of numerical features.
            epochs (int): Number of training epochs.
            batch (int): Batch size.
            learning_rate (float): Optimizer learning rate.
            analyze_every (int): Metric logging frequency.
        """
        train_loader = DataLoader(self.train_dataset, batch_size=batch, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=batch, drop_last = False)

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.property_dim = property_dim

        self.learning_rate = learning_rate
        # Create and train model. price_predictor contains model spec.
            # 2. Check if the predictor already exists
        if self.predictor is None:
            # FIRST TIME TRAINING: Initialize the model and optimizer
            self.embedding_dim = embedding_dim
            self.hidden_dim = hidden_dim
            self.property_dim = property_dim
            self.learning_rate = learning_rate
            
            self.predictor = price_predictor(self.device, self.embedding_dim, self.hidden_dim, self.property_dim,
                                        self.n_communities, 
                                        self.year_length,
                                        self.week_length,
                                        self.learning_rate)
        else:
            # CONTINUED TRAINING: Update learning rate if it changed
            # (The model weights and optimizer state are preserved in self.predictor)
            for param_group in self.predictor.optimizer.param_groups:
                param_group['lr'] = learning_rate

        start_time = datetime.now()
        train_losses, val_losses = self.predictor.train(train_loader, val_loader, epochs = epochs,
                                                                            analyze_every=analyze_every)
        end_time = datetime.now()
        print(f'{(end_time-start_time).total_seconds():.2f} seconds to train')

        self.results['train_losses'] = train_losses
        self.results['val_losses'] = val_losses
# region get predictions
# Add model prediction tensors and add to dataframe
    def add_predictions_to_data(self):
        """
        Runs inference on the dataset, reverses transformations (log-price -> price),
        calculates errors, and appends predictions to the internal dataframe.
        """
        year_tensor = vocab_replace_tensor(self.tensors.tensors[1], self.year_vocab)
        week_tensor = vocab_replace_tensor(self.tensors.tensors[2], self.week_vocab)
        # Create a new TensorDataset with the updated tensors
        new_dataset = list(self.tensors.tensors)  # Convert tuple to list

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
        scaler = self.scalers['log_price']

        # Reverse scaling to get actual log values
        predicted_log_price = scaler.inverse_transform(predictions).ravel()
        target_log_price = scaler.inverse_transform(target).ravel()
        
        # Store CLS attention values
        cls_labels = ["cls_community", "cls_year", "cls_week", "cls_property"]
        self.dataframe.loc[target_indices,cls_labels] = cls_output

        # initialise dataframe columns
        self.dataframe['predicted_value'] = pd.Series(dtype=float)
        self.dataframe['predicted_price'] = pd.Series(dtype=float)
        self.dataframe['pct_error'] = pd.Series(dtype=float)
        self.dataframe['target_log'] = pd.Series(dtype=float)
        self.dataframe['target'] = pd.Series(dtype=float)


        self.dataframe.iloc[target_indices, self.dataframe.columns.get_loc('predicted_value')] = predicted_log_price

        self.dataframe.iloc[target_indices, self.dataframe.columns.get_loc('target_log')] = target_log_price        
        self.dataframe.iloc[target_indices, self.dataframe.columns.get_loc('target')] = np.exp(target_log_price)

        # Convert log price back to actual price
        self.dataframe.iloc[prediction_indices, self.dataframe.columns.get_loc('predicted_price')] = np.exp(predicted_log_price)

        # Calculate error metrics
        self.dataframe['price_error']= self.dataframe['predicted_price']-self.dataframe['sale_price']
        self.dataframe['pct_error']=100*(self.dataframe['price_error']/self.dataframe['sale_price'])
        self.dataframe['sale_price_per_sqft'] = self.dataframe['sale_price']/self.dataframe['sqft']
        self.dataframe['predicted_price_per_sqft'] = self.dataframe['predicted_price']/self.dataframe['sqft']
        
        print(f'Mean absolute percentage error: {self.dataframe["pct_error"].abs().mean():.2f}')

# region save model
    def save_model(self):
        """
        Saves the model state, optimizer state, configuration, scalers, 
        and vocabulary files to the output directory.
        """
        import os
        
        os.makedirs(self.directory, exist_ok=True)

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
            'results': self.results,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'property_dim': self.property_dim,
            'community_embedding_length': self.community_embedding_length,
            'year_length': self.train_year_length,
            'week_length': self.train_week_length,
            "learning_rate": self.learning_rate
        }, f'{self.directory}/model.pth')

        with open(f'{self.directory}/community_vocab.json', 'w') as f:
            json.dump(self.community_vocab, f)

        # Save results separately as JSON
        with open(f'{self.directory}/results.json', 'w') as f:
            json.dump(self.results, f)

          # Save config as JSON
        with open(f'{self.directory}/config.json', 'w') as f:
            json.dump(config, f)

        # Save processor (scalers and parameters)
        with open(f'{self.directory}/week_vocab.json', 'w') as f:
            json.dump(self.week_vocab, f)
        # Save processor (scalers and parameters)
        with open(f'{self.directory}/year_vocab.json', 'w') as f:
            json.dump(self.year_vocab, f)

        print(f"Model and results saved in {self.directory}")
# region load model
    def load_model_and_artifacts(self, directory):
        from pathlib import Path
        """
        Loads a saved model, optimizer, results, and vocabularies from disk.

        Parameters:
            directory (str or Path): Path to the saved model directory.
        
        Returns:
            self: The object populated with loaded state.
        """
        directory = Path(directory)

        #replace directory
        self.directory = directory
        ckpt = torch.load(directory / "model.pth", map_location=self.device)

        # --- Reconstruct model with saved args ---

        self.embedding_dim=ckpt['embedding_dim']
        self.hidden_dim=ckpt['hidden_dim']
        self.property_dim=ckpt['property_dim']
        self.community_embedding_length=ckpt['community_embedding_length']
        self.year_length=ckpt['year_length']
        self.week_length=ckpt['week_length']
        self.learning_rate=ckpt['learning_rate']

        self.predictor = price_predictor(self.device, self.embedding_dim, self.hidden_dim, self.property_dim,
                                         self.community_embedding_length, self.year_length, self.week_length, 
                                         self.learning_rate)

        # Load weights
        self.predictor.model.load_state_dict(ckpt['model_state_dict'])
        self.predictor.eval()
        # Load Optimizer
        self.predictor.optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        for feature in ['sqft','sqft_lot','log_price','beds']: # List the features to scale
            self.scalers[feature] = joblib.load(os.path.join(self.directory, f"{feature}_scaler.pkl"))

        # --- Load vocabs ---
        with open(directory / "community_vocab.json", "r") as f:
            self.community_vocab = json.load(f)
        with open(directory / "year_vocab.json", "r") as f:
            self.year_vocab = json.load(f)
        with open(directory / "week_vocab.json", "r") as f:
            self.week_vocab = json.load(f)

        # --- Load results (train/val loss curves etc) ---
        self.results = ckpt.get("results", {})

        return self
    
def create_vocab(df, column, min=None, max=None):
    """Creates a dictionary mapping unique values in a DF column to integers."""
    # if min && max:
    ids = sorted(df[column].unique())
    vocab = {int(id): index for index, id in enumerate(ids)}
    return vocab

def create_tensor_vocab(tensor):
    """Creates a vocab dictionary from a unique values in a tensor."""
    values = sorted(tensor.unique().tolist())
    vocab = {year: idx for idx, year in enumerate(values)}
    # add unknown when value for when not in training data
    vocab["unknown"] = len(vocab)  # Add "unknown" token
    return vocab

def vocab_replace_tensor(tensor, vocab):
    """
    Replaces values in a tensor with their indices from a vocabulary dictionary.
    Handles missing values by mapping them to the 'unknown' token.
    """
    replaced = [vocab.get(value.item(), vocab['unknown']) for value in tensor]
    return torch.tensor(replaced, dtype=torch.int)