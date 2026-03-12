"""
Enhanced Embedding Model V2
Improvements:
- Continuous time features alongside embeddings
- Market indicator integration
- Uncertainty estimation
- Better temporal extrapolation
- H3 L9 neighborhood-aware community embeddings
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

# Handle both relative and absolute imports
try:
    from .h3_community_embedding import H3CommunityEmbedding
    from .h3_neighbor_mapper import ensure_h3_l9_mappings
except ImportError:
    from h3_community_embedding import H3CommunityEmbedding
    from h3_neighbor_mapper import ensure_h3_l9_mappings


class dataset:
    """
    Enhanced dataset handler with market indicators and continuous time features
    """
    def __init__(self):
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
        self.continuous_time_features = torch.empty(0)
        self.market_features = torch.empty(0)
        self.target = torch.empty(0)
        self.reference_date = None

    def _prepare_data(self, dataframe, include_market_indicators=True, future_year_buffer=5):
        """
        Enhanced data preparation with continuous time and market features
        Now includes H3 L9 neighborhood-aware community embeddings
        
        Parameters:
            dataframe: Input sales data
            include_market_indicators: Whether to add market indicators
            future_year_buffer: Number of future years to pre-allocate in vocabulary
                               (default: 5, allows predictions up to 5 years ahead)
        """
        df = dataframe.copy()
        
        # --- Data Cleaning & Filtering ---
        df['sale_date'] = pd.to_datetime(df['sale_date'])
        df = df.sort_values('sale_date')
        
        # Convert sale_nbr to numeric (handle string values from RentCast data)
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
        df['month'] = df['sale_date'].dt.month
        df['year'] = df['sale_date'].dt.isocalendar().year
        df['week'] = df['sale_date'].dt.isocalendar().week
        df['log_price'] = np.log(df['sale_price'])
        
        # --- Continuous Time Features ---
        df['time_trend'] = (df['sale_date'] - self.reference_date).dt.days / 365.25
        df['day_of_year'] = df['sale_date'].dt.dayofyear
        df['sin_day'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        df['cos_day'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
        df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
        df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
        df['quarter'] = df['sale_date'].dt.quarter
        
        # --- Market Indicators (if available) ---
        if include_market_indicators:
            # Check if market indicators already exist in dataframe
            if 'mortgage_rate' not in df.columns:
                print("Warning: mortgage_rate not found. Add market indicators using MarketIndicatorFetcher")
                df['mortgage_rate'] = 6.5  # Default fallback
            if 'unemployment_rate' not in df.columns:
                print("Warning: unemployment_rate not found. Add market indicators using MarketIndicatorFetcher")
                df['unemployment_rate'] = 4.0  # Default fallback
        
        # --- H3 L9 Index Generation (if needed) ---
        # Ensure h3_09 column exists before attempting neighborhood mapping
        if 'h3_09' not in df.columns:
            if 'lat' in df.columns and 'lng' in df.columns:
                print("  Generating H3 L9 indices from lat/lng...")
                import h3
                df['h3_09'] = df.apply(
                    lambda row: h3.latlng_to_cell(row['lat'], row['lng'], 9) 
                    if pd.notna(row['lat']) and pd.notna(row['lng']) 
                    else None,
                    axis=1
                )
                print(f"  ✓ Generated H3 L9 indices for {df['h3_09'].notna().sum()} properties")
            else:
                print("  Warning: No h3_09 column and no lat/lng columns found")
                df['h3_09'] = None
        
        # --- H3 L9 Neighborhood Community Mapping ---
        # Automatically compute neighbor mappings if they don't exist
        neighbor_map_path = 'data/h3_l9_neighbor_communities.json'
        
        # Only attempt neighborhood mapping if h3_09 column exists and has values
        if 'h3_09' in df.columns and df['h3_09'].notna().any():
            if not os.path.exists(neighbor_map_path):
                print("\n  H3 L9 neighbor mappings not found, computing automatically...")
                try:
                    l9_to_community, h3_neighbor_map, vocab_data = ensure_h3_l9_mappings(
                        df,
                        community_map_path='data/community_map.json',
                        output_dir='data'
                    )
                    print("  ✓ H3 L9 neighbor mappings computed and cached")
                except Exception as e:
                    print(f"  Warning: Could not compute H3 L9 mappings: {e}")
                    print("  Falling back to single community index (no neighborhood pooling)")
                    df['community_neighbors'] = None
                    h3_neighbor_map = None
            else:
                print("  Loading H3 L9 neighbor community mappings...")
                with open(neighbor_map_path, 'r') as f:
                    h3_neighbor_map = json.load(f)
            
            if h3_neighbor_map is not None:
                # Map each h3_09 to its 7 community indices (center + 6 neighbors)
                df['community_neighbors'] = df['h3_09'].map(h3_neighbor_map)
                
                # Handle missing mappings (use UNKNOWN community for all 7 positions)
                missing_mask = df['community_neighbors'].isna()
                if missing_mask.any():
                    print(f"  Warning: {missing_mask.sum()} rows have no H3 L9 neighbor mapping")
                    # Will be filled with UNKNOWN index later
                
                print(f"  ✓ Mapped {(~missing_mask).sum()} properties to H3 L9 neighborhoods")
            else:
                print("  ✗ No neighborhood mapping available (will use single community index)")
                df['community_neighbors'] = None
        else:
            print("  ✗ No H3 L9 indices available (will use single community index)")
            df['community_neighbors'] = None
        
        # --- Vocabulary Creation ---
        # Year vocab with future years pre-allocated
        min_year = int(df['year'].min())
        max_year = int(df['year'].max())
        
        # Extend vocab to include future years beyond training data
        year_range = range(min_year, max_year + future_year_buffer + 1)
        
        self.year_vocab = {int(year): idx for idx, year in enumerate(year_range)}
        self.year_vocab["unknown"] = len(self.year_vocab)  # For years beyond buffer
        
        print(f"Year vocabulary: {min_year} to {max_year + future_year_buffer} (+ unknown)")
        print(f"  Training data years: {min_year} to {max_year}")
        print(f"  Future buffer: {future_year_buffer} years")
        
        # Week vocab (1-53)
        self.week_vocab = {value: index for index, value in enumerate(range(1, 54))}
        self.week_vocab["unknown"] = len(self.week_vocab)
        
        # Community vocab - load from precomputed L9 vocab
        community_vocab_path = 'data/community_vocab_l9.json'
        if os.path.exists(community_vocab_path):
            with open(community_vocab_path, 'r') as f:
                vocab_data = json.load(f)
            # Convert string keys to integers
            self.community_vocab = {int(k): v for k, v in vocab_data['community_to_idx'].items()}
            self.community_vocab["unknown"] = vocab_data['unknown_idx']
            self.n_communities = vocab_data['num_communities']
            print(f"Loaded H3 L9 community vocabulary: {self.n_communities} communities")
        else:
            # Fallback to old method
            print("Warning: community_vocab_l9.json not found, using legacy community mapping")
            self.community_vocab = create_vocab(df, 'community')
            self.community_vocab["unknown"] = len(self.community_vocab)
            self.n_communities = len(self.community_vocab)
        
        # Map communities for backward compatibility (single index)
        df['community_index'] = df['community'].map(self.community_vocab).fillna(self.community_vocab["unknown"]).astype(int)
        
        # Store dataset dimensions
        self.length = df.shape[0]
        self.year_length = len(self.year_vocab)  # Include all years + unknown
        self.week_length = len(self.week_vocab) - 1  # Exclude unknown for weeks
        
        self.dataframe = df.reset_index(drop=True)
        
        return self


class EnhancedEmbeddingModel(nn.Module):
    """
    Enhanced model with continuous time features and uncertainty estimation
    Now uses H3 L9 neighborhood-aware community embeddings
    """
    def __init__(self, embedding_dim, hidden_dim, property_dim,
                 continuous_time_dim, market_dim,
                 community_embedding_length, 
                 year_length, week_length, 
                 dropout_rate=0.1,
                 estimate_uncertainty=True,
                 use_neighborhood_pooling=True,
                 pooling_strategy='mean'):
        super().__init__()
        
        self.estimate_uncertainty = estimate_uncertainty
        self.use_neighborhood_pooling = use_neighborhood_pooling
        
        # --- Embedding Layers (Categorical) ---
        if use_neighborhood_pooling:
            # Use H3CommunityEmbedding with neighborhood pooling
            # community_embedding_length is num_communities (not including unknown)
            self.community_embedding = H3CommunityEmbedding(
                num_communities=int(community_embedding_length),
                embedding_dim=embedding_dim,
                pooling_strategy=pooling_strategy
            )
        else:
            # Legacy single community embedding
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
        
        # --- Regressor Head (MLP) with Dropout ---
        self.dropout = nn.Dropout(dropout_rate)
        self.hidden_layer1 = nn.Linear(embedding_dim, hidden_dim)
        self.hidden_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_layer3 = nn.Linear(hidden_dim, hidden_dim // 2)
        
        # Mean prediction
        self.output_layer = nn.Linear(hidden_dim // 2, 1)
        
        # Uncertainty estimation (log variance)
        if estimate_uncertainty:
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
        
        if return_uncertainty and self.estimate_uncertainty:
            # Log variance (for numerical stability)
            log_var = self.uncertainty_layer(h3)
            return output, log_var
        
        return output


class price_predictor:
    """Enhanced predictor with uncertainty estimation and neighborhood pooling"""
    
    def __init__(self, device, embedding_dim, hidden_dim, property_dim,
                 continuous_time_dim, market_dim,
                 community_embedding_length,
                 year_length, week_length, learning_rate,
                 dropout_rate=0.1, estimate_uncertainty=True,
                 use_neighborhood_pooling=True, pooling_strategy='mean'):
        self.device = device
        self.estimate_uncertainty = estimate_uncertainty
        self.use_neighborhood_pooling = use_neighborhood_pooling
        
        self.model = EnhancedEmbeddingModel(
            embedding_dim, hidden_dim, property_dim,
            continuous_time_dim, market_dim,
            community_embedding_length, 
            year_length, week_length,
            dropout_rate=dropout_rate,
            estimate_uncertainty=estimate_uncertainty,
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
            # Negative log likelihood loss (accounts for uncertainty)
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
