"""
Embedding Model
Continuous time features, market indicator integration,
uncertainty estimation, H3 L9 neighborhood-aware community embeddings.
"""
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
from typing import Optional, Tuple, Dict

try:
    from fetch_market_indicators import fetch_fred_series, CACHE_FILE
except ImportError:
    from ..fetch_market_indicators import fetch_fred_series, CACHE_FILE

# Handle both relative and absolute imports
try:
    from .h3_community_embedding import H3CommunityEmbedding
    from .h3_neighbor_mapper import ensure_neighbor_communities
except ImportError:
    from h3_community_embedding import H3CommunityEmbedding
    from h3_neighbor_mapper import ensure_neighbor_communities


class dataset:
    """
    Enhanced dataset handler with market indicators and continuous time features
    """
    def __init__(self):
        self.length = None
        self.n_communities = None   # set by _map_communities(); unknown index = n_communities
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
        self.continuous_time_features = torch.empty(0)
        self.market_features = torch.empty(0)
        self.target = torch.empty(0)
        self.reference_date = None

    def _map_communities(self, dataframe,
                         community_map_path='community_map.json',
                         output_dir='data'):
        """
        Ensure H3  neighbor mappings exist and map each row to a 7-element
        community index vector (center hex + 6 neighbours).

        Must be called before _prepare_data so that 'community_neighbors' and
        'n_communities' are ready when the tensor is built.

        Indices in community_neighbors are already 0-based (0 … n_communities-1),
        with n_communities used as the unknown/padding index.

        Parameters:
            dataframe:            Raw input DataFrame (mutated in-place on the copy
                                  stored as self._raw_df for _prepare_data to pick up)
            community_map_path:   Path to H3 (L8) → community ID JSON
            output_dir:           Where cached mapping files live / are written
        """
        import h3 as _h3

        df = dataframe.copy()

        # --- Ensure h3_09 column ---
        if 'h3_08' not in df.columns:
            if 'lat' in df.columns and 'lng' in df.columns:
                print("  Generating H3 L8 indices from lat/lng...")
                df['h3_08'] = df.apply(
                    lambda row: _h3.latlng_to_cell(row['lat'], row['lng'], 8)
                    if pd.notna(row['lat']) and pd.notna(row['lng'])
                    else None,
                    axis=1
                )
                print(f"  ✓ Generated H3 L9 indices for {df['h3_08'].notna().sum()} properties")
            else:
                print("  Warning: No h3_08 column and no lat/lng columns — "
                      "community mapping unavailable")
                df['h3_08'] = None

        # --- Map h3_08 → community via community_map.json (H3 → community ID) ---
        community_map_path = os.path.join(output_dir, 'community_map.json')
        if 'h3_08' in df.columns and df['h3_08'].notna().any() and os.path.exists(community_map_path):
            print(f"  Loading community map from {community_map_path}...")
            with open(community_map_path, 'r') as f:
                community_map = json.load(f)

            df['community'] = df['h3_08'].map(community_map)
            mapped = df['community'].notna().sum()
            print(f"  ✓ Mapped {mapped}/{len(df)} rows to a community "
                  f"({len(df) - mapped} unmapped will receive unknown index)")
        else:
            if not os.path.exists(community_map_path):
                print(f"  Warning: {community_map_path} not found — 'community' column not set")
            df['community'] = None
            community_map = {}

        # --- Load / compute neighbor map (community_neighbors) ---
        h3_neighbor_map = None

        if 'h3_08' in df.columns and df['h3_08'].notna().any():
            try:
                h3_neighbor_map, n_communities = ensure_neighbor_communities(
                    df,
                    community_map_path=community_map_path,
                    output_dir=output_dir
                )
                self.n_communities = n_communities
                print(f"  ✓ Neighbor mapping ready: {self.n_communities} communities "
                      f"(unknown index = {self.n_communities})")
            except Exception as e:
                print(f"  Warning: Could not compute H3 L8 mappings: {e}")
                print("  Falling back — community_neighbors will be None")

        # --- Fill unmapped community values with the unknown index ---
        # n_communities is one past the last real index, matching the convention
        # used in h3_neighbor_mapper and the ONNX model embedding table.
        if self.n_communities is not None:
            unknown_idx = self.n_communities
            unmapped = df['community'].isna().sum()
            if unmapped:
                df['community'] = df['community'].fillna(unknown_idx).astype(int)
                print(f"  Assigned unknown index ({unknown_idx}) to {unmapped} unmapped rows")
            else:
                df['community'] = df['community'].astype(int)

        if h3_neighbor_map is not None:
            df['community_neighbors'] = df['h3_08'].map(h3_neighbor_map)
            missing = df['community_neighbors'].isna().sum()
            if missing:
                print(f"  Warning: {missing} rows have no H3 L8 neighbor mapping "
                      f"(will use unknown index {self.n_communities})")
            print(f"  ✓ Mapped {len(df) - missing}/{len(df)} rows to H3 L8 neighbourhoods")
        else:
            df['community_neighbors'] = None
            print("  ✗ No neighbourhood mapping available")

        # Stash the enriched dataframe so _prepare_data can use it directly
        self.dataframe = df
        return self
    
    def _add_market_indicators(self, df, market_indicator_cache_path=None):
        if market_indicator_cache_path is None:
            market_indicator_cache_path = CACHE_FILE

        indicator_frame = None

        try:
            print("  Attempting to fetch indicators from FRED...")
            indicator_frame = pd.DataFrame({
                'mortgage_rate': fetch_fred_series('MORTGAGE30US', start='2019-01-01'),
                'unemployment_rate': fetch_fred_series('UNRATE', start='2019-01-01'),
            })
            indicator_frame.index.name = 'date'
        except Exception as exc:
            if os.path.exists(market_indicator_cache_path):
                print(f"  Warning: refresh from FRED failed ({exc}); using cached data.")
                print(f"  Loading cached indicators from {market_indicator_cache_path}...")
                indicator_frame = pd.read_csv(market_indicator_cache_path, index_col='date', parse_dates=True)
            else:
                raise RuntimeError(f"Could not fetch indicators from FRED and no cache is available: {exc}") from exc

        if indicator_frame is not None:
            indicator_frame = indicator_frame.copy()
            indicator_frame.index = pd.to_datetime(indicator_frame.index)
            indicator_frame = indicator_frame.sort_index()
            indicator_frame = indicator_frame.ffill().bfill()

            sale_dates = pd.to_datetime(df['sale_date']).dt.normalize()
            for col in ['mortgage_rate', 'unemployment_rate']:
                if col in indicator_frame.columns:
                    df[col] = sale_dates.map(indicator_frame[col])
            df[['mortgage_rate', 'unemployment_rate']] = df[['mortgage_rate', 'unemployment_rate']].ffill().bfill()

    def _prepare_data(self, market_indicator_cache_path=None):
        """
        Data preparation: add market indicators, cleaning, feature engineering, and vocab creation.

        Community mapping (community_neighbors, n_communities) must already be
        set — call _map_communities() first, or pass a dataframe that already
        has a 'community_neighbors' column with 0-based indices.

        Parameters:
            market_indicator_cache_path: Optional path to a cached indicator CSV.
        """

        df = self.dataframe

        self._add_market_indicators(df, market_indicator_cache_path=market_indicator_cache_path)

        # --- Data Cleaning & Filtering ---
        df['sale_date'] = pd.to_datetime(df['sale_date'])
        df = df.sort_values('sale_date')

        df['sale_nbr'] = pd.to_numeric(df['sale_nbr'], errors='coerce')

        df = df.dropna(subset=['sale_price', 'lat', 'lng', 'sqft', 'sale_nbr', 'sale_date', 'sqft_lot'])
        df = df[df['sale_price'] > 0]
        df = df[df['sqft'] > 0]
        df = df[df['sale_nbr'] > 0]
        df = df[df['sqft_lot'] > 0]

        # Store reference date for continuous time features
        if self.reference_date is None:
            self.reference_date = df['sale_date'].min()

        # --- Feature Engineering ---
        df['price_per_sqft'] = df['sale_price'] / df['sqft']
        df['month']          = df['sale_date'].dt.month
        df['year']           = df['sale_date'].dt.isocalendar().year
        df['week']           = df['sale_date'].dt.isocalendar().week
        df['log_price']      = np.log(df['sale_price'])

        # --- Continuous Time Features ---
        df['time_trend']  = (df['sale_date'] - self.reference_date).dt.days / 365.25
        df['day_of_year'] = df['sale_date'].dt.dayofyear
        df['sin_day']     = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        df['cos_day']     = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
        df['sin_month']   = np.sin(2 * np.pi * df['month'] / 12)
        df['cos_month']   = np.cos(2 * np.pi * df['month'] / 12)
        df['quarter']     = df['sale_date'].dt.quarter

        # --- Year Vocabulary ---
        min_year = int(df['year'].min())
        max_year = int(df['year'].max())
        year_range = range(min_year, max_year + 1)
        self.year_vocab = {int(y): idx for idx, y in enumerate(year_range)}
        self.year_vocab["unknown"] = len(self.year_vocab)
        print(f"Year vocabulary: {min_year} to {max_year} (+ unknown)")

        # --- Week Vocabulary (weeks 1–53 + unknown) ---
        self.week_vocab = {value: index for index, value in enumerate(range(1, 54))}
        self.week_vocab["unknown"] = len(self.week_vocab)

        # --- Dataset Dimensions ---
        self.length      = df.shape[0]
        self.year_length = len(self.year_vocab)       # all years + unknown
        self.week_length = len(self.week_vocab) - 1   # exclude unknown

        self.dataframe = df.reset_index(drop=True)

        return self


class EnhancedEmbeddingModel(nn.Module):
    """
    Transformer-based house price model with H3 L9 neighborhood-aware community embeddings.
    Always builds the uncertainty head (uncertainty_layer); use return_uncertainty=True
    in forward() to get log-variance output at inference time.
    """
    def __init__(self, embedding_dim, hidden_dim, property_dim,
                 continuous_time_dim, market_dim,
                 community_embedding_length, 
                 year_length, week_length, 
                 dropout_rate=0.1,
                 estimate_uncertainty=False,   # kept for API compat, no longer gates uncertainty_layer
                 use_neighborhood_pooling=True,
                 pooling_strategy='mean'):
        super().__init__()
        
        self.use_neighborhood_pooling = use_neighborhood_pooling
        
        # --- Embedding Layers (Categorical) ---
        if use_neighborhood_pooling:
            self.community_embedding = H3CommunityEmbedding(
                num_communities=int(community_embedding_length),
                embedding_dim=embedding_dim,
                pooling_strategy=pooling_strategy
            )
        else:
            self.community_embedding = nn.Embedding(int(community_embedding_length), embedding_dim)
        
        self.year_embedding = nn.Embedding(int(year_length), embedding_dim)
        self.week_embedding = nn.Embedding(int(week_length), embedding_dim)
        
        # --- Feature Projection Layers ---
        self.property_feature_layer = nn.Linear(property_dim, embedding_dim)
        self.time_feature_layer = nn.Linear(continuous_time_dim, embedding_dim)
        self.market_feature_layer = nn.Linear(market_dim, embedding_dim)
        
        # --- Learnable CLS Token ---
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        
        # --- Attention Mechanism ---
        self.attention_layer = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=4,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # --- Regressor Head ---
        self.dropout = nn.Dropout(dropout_rate)
        self.hidden_layer1 = nn.Linear(embedding_dim, hidden_dim)
        self.hidden_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_layer3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.output_layer  = nn.Linear(hidden_dim // 2, 1)
        
        # Uncertainty head — always built, activated via return_uncertainty at inference
        self.uncertainty_layer = nn.Linear(hidden_dim // 2, 1)
        
        self.relu = nn.ReLU()
        self.last_attention_weights = None
        self.last_cls_attention = None

    def forward(self, community_indices, year, week, property_features, 
                time_features, market_features, return_uncertainty=False):
        """
        Forward pass with continuous time and market features
        
        Args:
            community_indices: If use_neighborhood_pooling=True, shape (batch, 7)
                             Otherwise, shape (batch,)
            year, week: Shape (batch,)
            property_features, time_features, market_features: Shape (batch, feature_dim)
        """
        # --- Safety: Clamp indices to valid range ---
        if self.use_neighborhood_pooling:
            # Clamp each of the 7 neighbor indices
            community_indices = torch.clamp(
                community_indices, 0, 
                self.community_embedding.vocab_size - 1
            )
            # community_embedding expects (batch, 7) and returns (batch, embedding_dim)
            community_embeddings = self.community_embedding(community_indices)
        else:
            # Legacy single index
            community_indices = torch.clamp(
                community_indices, 0, 
                self.community_embedding.num_embeddings - 1
            )
            community_embeddings = self.community_embedding(community_indices)
        
        year = torch.clamp(year, 0, self.year_embedding.num_embeddings - 1)
        week = torch.clamp(week, 0, self.week_embedding.num_embeddings - 1)
        
        # --- Embed Categorical Inputs ---
        year_embeddings = self.year_embedding(year)
        week_embeddings = self.week_embedding(week)
        
        # --- Process Continuous Features ---
        processed_property = self.relu(self.property_feature_layer(property_features))
        processed_time = self.relu(self.time_feature_layer(time_features))
        processed_market = self.relu(self.market_feature_layer(market_features))
        
        # --- Stack Sequence ---
        # [community, year, week, property, time, market]
        tokens = torch.stack([
            community_embeddings, 
            year_embeddings, 
            week_embeddings, 
            processed_property,
            processed_time,
            processed_market
        ], dim=1)
        
        # --- Prepend CLS Token ---
        B = tokens.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        seq = torch.cat([cls, tokens], dim=1)
        
        # --- Self-Attention ---
        attention_output, attention_weights = self.attention_layer(
            seq, seq, seq,
            need_weights=True, average_attn_weights=False
        )
        
        cls_out = attention_output[:, 0, :]
        
        # Store attention weights
        self.last_attention_weights = attention_weights.detach()
        self.last_cls_attention = attention_weights[:, :, 0, 1:].detach()
        
        # --- MLP Head with Dropout ---
        h1 = self.dropout(self.relu(self.hidden_layer1(cls_out)))
        h2 = self.dropout(self.relu(self.hidden_layer2(h1)))
        h3 = self.relu(self.hidden_layer3(h2))
        
        # Mean prediction
        output = self.output_layer(h3)
        
        if return_uncertainty:
            # Log variance for uncertainty estimation (always available)
            log_var = self.uncertainty_layer(h3)
            return output, log_var
        
        return output


class price_predictor:
    """
    Wraps EnhancedEmbeddingModel for training and inference.
    
    estimate_uncertainty controls the TRAINING LOSS only:
      - False (default): plain MSE — stable, comparable train/val curves
      - True: NLL loss — can destabilise training but theoretically optimal
    
    Uncertainty estimates are always available at inference via return_uncertainty=True
    in forward(), regardless of how the model was trained.
    """
    
    def __init__(self, device, embedding_dim, hidden_dim, property_dim,
                 continuous_time_dim, market_dim,
                 community_embedding_length,
                 year_length, week_length, learning_rate,
                 dropout_rate=0.1, estimate_uncertainty=False,
                 use_neighborhood_pooling=False, pooling_strategy='mean'):
        self.device = device
        self.estimate_uncertainty = estimate_uncertainty  # training loss only
        self.use_neighborhood_pooling = use_neighborhood_pooling
        
        self.model = EnhancedEmbeddingModel(
            embedding_dim, hidden_dim, property_dim,
            continuous_time_dim, market_dim,
            community_embedding_length, 
            year_length, week_length,
            dropout_rate=dropout_rate,
            use_neighborhood_pooling=use_neighborhood_pooling,
            pooling_strategy=pooling_strategy
        ).to(device)
        
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
    
    def eval(self):
        self.model.eval()
    
    def train_step(self, batch):
        """Single training step with optional uncertainty loss"""
        batch = tuple(t.to(self.device) for t in batch)
        community, year, week, property_feat, time_feat, market_feat, targets = batch
        
        self.optimizer.zero_grad()
        
        if self.estimate_uncertainty:
            predictions, log_var = self.model(
                community, year, week, property_feat, time_feat, market_feat,
                return_uncertainty=True
            )
            # Negative log-likelihood (Gaussian).
            # Clamp log_var to a safe range to prevent:
            #   - log_var → -∞: precision → ∞, loss explodes
            #   - log_var →  ∞: loss dominated by variance term, ignores fit
            log_var = torch.clamp(log_var, min=-6.0, max=6.0)
            precision = torch.exp(-log_var)
            loss = torch.mean(precision * (predictions.squeeze() - targets)**2 + log_var.squeeze())
        else:
            predictions = self.model(community, year, week, property_feat, time_feat, market_feat)
            loss = self.criterion(predictions.squeeze(), targets)
        
        if torch.isnan(loss):
            print("NaN loss detected, skipping batch")
            return 0.0
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, train_loader, val_loader, epochs, patience=10):
        """Training loop with early stopping"""
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            for batch in train_loader:
                train_loss += self.train_step(batch)
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = tuple(t.to(self.device) for t in batch)
                    community, year, week, property_feat, time_feat, market_feat, targets = batch
                    if self.estimate_uncertainty:
                        predictions, log_var = self.model(
                            community, year, week, property_feat, time_feat, market_feat,
                            return_uncertainty=True
                        )
                        log_var = torch.clamp(log_var, min=-6.0, max=6.0)
                        precision = torch.exp(-log_var)
                        loss = torch.mean(precision * (predictions.squeeze() - targets)**2 + log_var.squeeze())
                    else:
                        predictions = self.model(community, year, week, property_feat, time_feat, market_feat)
                        loss = self.criterion(predictions.squeeze(), targets)
                    val_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        return train_losses, val_losses


def create_vocab(df, column):
    """Creates a dictionary mapping unique values in a DF column to integers."""
    ids = sorted(df[column].unique())
    vocab = {int(id): index for index, id in enumerate(ids)}
    return vocab


def vocab_replace_tensor(tensor, vocab):
    """Replaces values in a tensor with their indices from vocabulary"""
    # Get the unknown index (should be the last index)
    unknown_idx = vocab.get('unknown', len(vocab) - 1)
    
    replaced = []
    for value in tensor:
        val = value.item()
        # Try to get the vocab index, use unknown if not found
        if val in vocab:
            replaced.append(vocab[val])
        else:
            replaced.append(unknown_idx)
    
    return torch.tensor(replaced, dtype=torch.int)
