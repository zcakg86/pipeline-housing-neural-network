"""
Model Manager
Handles training, prediction, and retraining pipeline.
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
from typing import Optional, Tuple, Dict, List

# Handle both relative and absolute imports
try:
    from .embedding_model import dataset, price_predictor, vocab_replace_tensor
except ImportError:
    from embedding_model import dataset, price_predictor, vocab_replace_tensor


class modelmanager:
    """
    Enhanced model manager with retraining pipeline and uncertainty estimation
    """
    def __init__(self, model_name="property_model"):
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
            'timestamp': self.timestamp
        }
        self.scalers = {}
        self.predictor = None
        self.reference_date = None
        
        # Model architecture params
        self.embedding_dim = None
        self.hidden_dim = None
        self.property_dim = None
        self.continuous_time_dim = None
        self.market_dim = None
        self.n_communities = None
        self.year_length = None
        self.week_length = None
        self.learning_rate = None
        self.dropout_rate = None
        self.epochs = None
        self.pooling_strategy = 'mean'
        self.use_neighborhood_pooling = False
        
        # Vocabularies
        self.year_vocab = None
        self.week_vocab = None
        
        os.makedirs(self.directory, exist_ok=True)
    
    # Features that need scaling — shared across processor and scaler helpers
    _PROPERTY_FEATURES = ['sqft', 'sqft_lot', 'beds']
    _TIME_FEATURES     = ['time_trend']
    _MARKET_FEATURES   = ['mortgage_rate', 'unemployment_rate']
    _TARGET_FEATURE    = 'log_price'

    def processor(self, data, scale_mode = ''):
        """
        Enhanced processor with continuous time and market features.
        Now handles H3 L9 neighborhood community tensors (batch, 7).

        Scalers are NOT fitted here.  They are fitted exclusively on the
        training split after split_data() is called.  If pre-fitted scalers
        already exist in self.scalers (e.g. loaded from disk for inference),
        they are applied immediately so predictions work without re-splitting.
        Only if scale_mode is 'force_new', then new scalers are calculated. 
        """
        property_features = self._PROPERTY_FEATURES
        time_features     = self._TIME_FEATURES
        market_features   = self._MARKET_FEATURES
        target_feature    = self._TARGET_FEATURE
        all_features = property_features + time_features + market_features + [target_feature]

        # Store metadata from the dataset object
        self.dataframe    = data.dataframe
        self.year_length  = data.year_length
        self.week_length  = data.week_length
        self.n_communities = data.n_communities
        self.data_length  = data.length
        self.week_vocab   = data.week_vocab
        self.year_vocab   = data.year_vocab
        self.reference_date = data.reference_date

        # Remember scale_mode so split_data knows what to do
        self._scale_mode = scale_mode

        # If scalers already exist (inference / reuse path), apply them now so
        # the tensor build below has scaled columns available.
        if self.scalers:
            self._apply_scalers(self.dataframe)

        # Build community tensor
        # community_neighbors values are already 0-based indices; n_communities is the unknown slot.
        if 'community_neighbors' in data.dataframe.columns and data.dataframe['community_neighbors'].notna().any():
            unknown_idx = data.n_communities
            community_neighbors_list = []
            for neighbors in data.dataframe['community_neighbors']:
                if isinstance(neighbors, list) and len(neighbors) == 7:
                    community_neighbors_list.append(neighbors)
                else:
                    community_neighbors_list.append([unknown_idx] * 7)
            community_tensor = torch.tensor(community_neighbors_list, dtype=torch.long)
            self.use_neighborhood_pooling = True
            print(f"Using H3 L9 neighborhood pooling: community tensor shape {community_tensor.shape}")
        else:
            # Fallback: single community index per row (legacy path)
            community_tensor = torch.tensor(data.dataframe['community'].values, dtype=torch.long)
            self.use_neighborhood_pooling = False
            print(f"Using single community index: community tensor shape {community_tensor.shape}")

        # Store unscaled feature arrays so _fit_scalers_on_train can build tensors
        # after fitting.  We build placeholder scaled columns with zeros for now;
        # they are overwritten in split_data → _fit_and_apply_scalers_on_split.
        for feature in all_features:
            if feature not in self.dataframe.columns:
                print(f"Warning: {feature} not found in dataframe")
            elif f"{feature}_scaled" not in self.dataframe.columns:
                # Temporary placeholder — replaced after scaler fitting in split_data
                self.dataframe[f"{feature}_scaled"] = 0.0

        self._community_tensor = community_tensor
        # Defer full TensorDataset build until scalers are fitted (split_data)
        # but if scalers were already applied above, build now.
        if self.scalers:
            self._build_tensor_dataset()

        return self

    def _apply_scalers(self, df):
        """Apply already-fitted scalers to scaled columns in df (in-place)."""
        all_features = (self._PROPERTY_FEATURES + self._TIME_FEATURES +
                        self._MARKET_FEATURES + [self._TARGET_FEATURE])
        for feature in all_features:
            if feature in self.scalers and feature in df.columns:
                df[f"{feature}_scaled"] = self.scalers[feature].transform(df[[feature]])

    def _fit_and_apply_scalers_on_split(self, train_indices):
        """
        Fit scalers on training rows only, then apply to the full dataframe.
        Called from split_data() so scalers never see validation data.
        Saves scaler files to self.directory as before.
        """
        scale_mode = getattr(self, '_scale_mode', 'fit')
        all_features = (self._PROPERTY_FEATURES + self._TIME_FEATURES +
                        self._MARKET_FEATURES + [self._TARGET_FEATURE])

        train_df = self.dataframe.iloc[train_indices]

        for feature in all_features:
            if feature not in self.dataframe.columns:
                continue

            scaler_path = os.path.join(self.directory, f"{feature}_scaler.pkl")
            scaler = None

            # Check memory first
            if feature in self.scalers:
                scaler = self.scalers[feature]
            # Then disk (skip for force_new)
            elif os.path.exists(scaler_path) and scale_mode != "force_new":
                scaler = joblib.load(scaler_path)
                self.scalers[feature] = scaler
                print(f"  Loaded existing scaler for {feature}")

            if scaler is None:
                # Fit on training rows only — no leakage into validation
                scaler = StandardScaler()
                scaler.fit(train_df[[feature]])
                self.scalers[feature] = scaler
                joblib.dump(scaler, scaler_path)
                print(f"  Fitted new scaler for {feature} on {len(train_indices)} training rows")

            # Apply to the full dataframe (train + val)
            self.dataframe[f"{feature}_scaled"] = scaler.transform(self.dataframe[[feature]])

        # Rebuild the TensorDataset now that scaled columns are correct
        self._build_tensor_dataset()

    def _build_tensor_dataset(self):
        """Construct self.tensors from the current scaled columns."""
        property_features = self._PROPERTY_FEATURES
        time_features     = self._TIME_FEATURES
        market_features   = self._MARKET_FEATURES

        self.tensors = TensorDataset(
            self._community_tensor,
            torch.tensor(self.dataframe['year'].values, dtype=torch.long),
            torch.tensor(self.dataframe['week'].values, dtype=torch.long),
            torch.tensor(self.dataframe[[f'{f}_scaled' for f in property_features]].values, dtype=torch.float32),
            torch.tensor(self.dataframe[[f'{f}_scaled' for f in time_features]].values, dtype=torch.float32),
            torch.tensor(self.dataframe[[f'{f}_scaled' for f in market_features]].values, dtype=torch.float32),
            torch.tensor(self.dataframe['log_price_scaled'].values, dtype=torch.float32)
        )
    
    def split_data(self, train_ratio=0.8, temporal_split=False):
        """
        Split data into train/val sets, then fit scalers on train rows only.

        Scaler fitting is deferred to here so that validation rows are never
        seen during fit — preventing data leakage into evaluation metrics.

        temporal_split: If True, uses chronological split instead of random.
        """
        total_length = len(self.dataframe)
        train_size   = int(train_ratio * total_length)

        if temporal_split:
            # Chronological split: earlier rows → train
            train_indices = list(range(train_size))
            val_indices   = list(range(train_size, total_length))
        else:
            # Random split — derive indices from a random permutation
            perm          = torch.randperm(total_length).tolist()
            train_indices = perm[:train_size]
            val_indices   = perm[train_size:]

        # --- Fit scalers on training rows only, then apply to full dataframe ---
        print(f"\nFitting scalers on {len(train_indices)} training rows "
              f"(holding out {len(val_indices)} validation rows)...")
        self._fit_and_apply_scalers_on_split(train_indices)
        # self.tensors is rebuilt inside _fit_and_apply_scalers_on_split

        # --- Build train / val Subsets ---
        if temporal_split:
            self.train_dataset = Subset(self.tensors, train_indices)
            self.val_dataset   = Subset(self.tensors, val_indices)
        else:
            self.train_dataset = Subset(self.tensors, train_indices)
            self.val_dataset   = Subset(self.tensors, val_indices)

        self.tensor_length = total_length

        # --- Vocab replacement (year / week raw → vocab index) ---
        def _replace_vocab_in_subset(subset, indices):
            year_tensor = vocab_replace_tensor(
                self.tensors.tensors[1][indices], self.year_vocab
            )
            week_tensor = vocab_replace_tensor(
                self.tensors.tensors[2][indices], self.week_vocab
            )
            new_tensors = list(self.tensors.tensors)
            # Build a mini-TensorDataset for this split with correct vocab indices
            split_tensors = [t[indices] for t in new_tensors]
            split_tensors[1] = year_tensor
            split_tensors[2] = week_tensor
            return TensorDataset(*split_tensors)

        train_td = _replace_vocab_in_subset(self.train_dataset, train_indices)
        val_td   = _replace_vocab_in_subset(self.val_dataset,   val_indices)

        # Subsets with contiguous indices into their own TensorDatasets
        self.train_dataset = Subset(train_td, list(range(len(train_indices))))
        self.val_dataset   = Subset(val_td,   list(range(len(val_indices))))

        print(f"Split complete — train: {len(train_indices)}, val: {len(val_indices)}")

        return self
    
    def train_model(self, embedding_dim=16, hidden_dim=32, 
                   property_dim=3, continuous_time_dim=1, market_dim=2,
                   epochs=50, batch=256, learning_rate=0.0003,
                   dropout_rate=0.1, estimate_uncertainty=True,
                   pooling_strategy='mean', patience=20):
        """
        Train the enhanced model with H3 L9 neighborhood pooling
        
        Args:
            pooling_strategy: 'mean', 'center_weighted', or 'learnable'
        """
        train_loader = DataLoader(self.train_dataset, batch_size=batch, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=batch, drop_last=False)
        
        # Store architecture params
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.property_dim = property_dim
        self.continuous_time_dim = continuous_time_dim
        self.market_dim = market_dim
        self.learning_rate = learning_rate
        self.pooling_strategy = pooling_strategy
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.estimate_uncertainty = estimate_uncertainty
        # Initialize or update predictor. The manager owns the flag so the
        # predictor always receives the same value that processor() derived.
        if self.predictor is None:
            self.predictor = price_predictor(
                self.device, embedding_dim, hidden_dim, 
                property_dim, continuous_time_dim, market_dim,
                self.n_communities + 1,   # +1 to include the unknown index slot (= n_communities)
                self.year_length, self.week_length,
                learning_rate, dropout_rate, estimate_uncertainty,
                use_neighborhood_pooling=self.use_neighborhood_pooling,
                pooling_strategy=pooling_strategy
            )
        else:
            # Update learning rate for continued training
            for param_group in self.predictor.optimizer.param_groups:
                param_group['lr'] = learning_rate
        
        start_time = datetime.now()
        train_losses, val_losses = self.predictor.train(
            train_loader, val_loader, epochs, patience
        )
        end_time = datetime.now()
        
        print(f'{(end_time-start_time).total_seconds():.2f} seconds to train')
        
        self.results['train_losses'] = train_losses
        self.results['val_losses'] = val_losses
        
        return self
    
    def add_predictions_to_data(self, return_uncertainty=True):
        """
        Add predictions with uncertainty estimates and CLS attention weights to dataframe.
        Uncertainty is always available regardless of training loss used.
        """
        year_tensor = vocab_replace_tensor(self.tensors.tensors[1], self.year_vocab)
        week_tensor = vocab_replace_tensor(self.tensors.tensors[2], self.week_vocab)
        
        new_dataset = list(self.tensors.tensors)
        new_dataset[1] = year_tensor
        new_dataset[2] = week_tensor
        new_dataset = TensorDataset(*new_dataset)
        
        loader = DataLoader(new_dataset, batch_size=256)
        self.predictor.eval()
        
        predictions  = []
        uncertainties = []
        targets       = []
        cls_attentions = []
        
        with torch.no_grad():
            for batch in loader:
                batch = tuple(t.to(self.predictor.device) for t in batch)
                community, year, week, property_feat, time_feat, market_feat, target = batch
                
                if return_uncertainty:
                    # Uncertainty head is always built — available regardless of training loss
                    pred, log_var = self.predictor.model(
                        community, year, week, property_feat, time_feat, market_feat,
                        return_uncertainty=True
                    )
                    uncertainties.extend(torch.exp(log_var / 2).cpu().numpy())
                else:
                    pred = self.predictor.model(
                        community, year, week, property_feat, time_feat, market_feat
                    )
                
                predictions.extend(pred.cpu().numpy())
                targets.extend(target.cpu().numpy())
                
                if self.predictor.model.last_cls_attention is not None:
                    cls_attn = self.predictor.model.last_cls_attention.mean(dim=1).cpu().numpy()
                    cls_attentions.extend(cls_attn)
        
        # Reshape
        predictions = np.array(predictions).reshape(-1, 1)
        targets = np.array(targets).reshape(-1, 1)
        
        # Inverse transform
        scaler = self.scalers['log_price']
        predicted_log_price = scaler.inverse_transform(predictions).ravel()
        target_log_price = scaler.inverse_transform(targets).ravel()
        
        # Add to dataframe
        self.dataframe['predicted_log_price'] = predicted_log_price
        self.dataframe['predicted_price'] = np.exp(predicted_log_price)
        self.dataframe['target_log'] = target_log_price
        self.dataframe['target'] = np.exp(target_log_price)
        
        # Add CLS attention weights
        # Tokens: [community, year, week, property, time, market]
        if len(cls_attentions) > 0:
            cls_attentions = np.array(cls_attentions)
            self.dataframe['cls_attn_community'] = cls_attentions[:, 0]
            self.dataframe['cls_attn_year'] = cls_attentions[:, 1]
            self.dataframe['cls_attn_week'] = cls_attentions[:, 2]
            self.dataframe['cls_attn_property'] = cls_attentions[:, 3]
            self.dataframe['cls_attn_time'] = cls_attentions[:, 4]
            self.dataframe['cls_attn_market'] = cls_attentions[:, 5]
            
            print(f"\nCLS Attention Weights (average across all predictions):")
            print(f"  Community: {self.dataframe['cls_attn_community'].mean():.3f}")
            print(f"  Year:      {self.dataframe['cls_attn_year'].mean():.3f}")
            print(f"  Week:      {self.dataframe['cls_attn_week'].mean():.3f}")
            print(f"  Property:  {self.dataframe['cls_attn_property'].mean():.3f}")
            print(f"  Time:      {self.dataframe['cls_attn_time'].mean():.3f}")
            print(f"  Market:    {self.dataframe['cls_attn_market'].mean():.3f}")
        
        # Uncertainty (in log space, then convert to price space)
        if len(uncertainties) > 0:
            uncertainties = np.array(uncertainties).reshape(-1, 1)
            uncertainty_unscaled = scaler.scale_[0] * np.array(uncertainties).ravel()
            self.dataframe['prediction_std_log'] = uncertainty_unscaled
            # Approximate std in price space using delta method
            self.dataframe['prediction_std_price'] = self.dataframe['predicted_price'] * uncertainty_unscaled
            # 95% confidence interval
            self.dataframe['price_lower_95'] = np.exp(predicted_log_price - 1.96 * uncertainty_unscaled)
            self.dataframe['price_upper_95'] = np.exp(predicted_log_price + 1.96 * uncertainty_unscaled)
        
        # Error metrics
        self.dataframe['price_error'] = self.dataframe['predicted_price'] - self.dataframe['sale_price']
        self.dataframe['pct_error'] = 100 * (self.dataframe['price_error'] / self.dataframe['sale_price'])
        
        print(f'\nMean absolute percentage error: {self.dataframe["pct_error"].abs().mean():.2f}%')
        
        if len(uncertainties) > 0:
            # Check calibration: what % of actual prices fall within 95% CI
            in_ci = ((self.dataframe['sale_price'] >= self.dataframe['price_lower_95']) & 
                    (self.dataframe['sale_price'] <= self.dataframe['price_upper_95']))
            print(f'95% CI coverage: {in_ci.mean()*100:.1f}% (should be ~95%)')
        
        return self
    
    def extend_year_vocab(self, new_years):
        """
        Extend the year vocabulary and grow the year_embedding weight matrix to
        accommodate additional years without discarding learned weights.

        Parameters
        ----------
        new_years : iterable of int
            Calendar years to add (years already in the vocab are silently skipped).

        Returns
        -------
        self

        What this does
        --------------
        1. Adds each new year to self.year_vocab with the next available index.
        2. Updates self.year_length to reflect the new vocab size.
        3. Replaces self.predictor.model.year_embedding with a new nn.Embedding
           that has one extra row per new year.  Existing weight rows are copied
           verbatim; new rows are randomly initialised (same distribution as
           torch default — N(0,1)).
        4. Saves the updated year_vocab.json to self.directory.

        Notes
        -----
        * The "unknown" token always stays at the last index.  New years are
          inserted before it, and the unknown row is carried over correctly.
        * After extending you should fine-tune the model on data containing the
          new years so the new embedding rows learn meaningful representations.
        """
        if self.predictor is None:
            raise RuntimeError("No model loaded — call train_model() or load_model() first.")

        # Determine which years are genuinely new
        # year_vocab keys are ints (plus the "unknown" string key)
        existing_years = {k for k in self.year_vocab if isinstance(k, int)}
        years_to_add   = [int(y) for y in sorted(new_years) if int(y) not in existing_years]

        if not years_to_add:
            print("extend_year_vocab: all supplied years are already in the vocab — nothing to do.")
            return self

        # Current unknown index (always the last row)
        old_unknown_idx = self.year_vocab["unknown"]
        old_size        = old_unknown_idx + 1  # total rows including unknown

        # Remap: new years go between the last known year and the unknown token
        # i.e. insert them before the existing unknown row.
        new_year_indices = {}
        for i, year in enumerate(years_to_add):
            new_year_indices[year] = old_unknown_idx + i  # push unknown further out

        new_unknown_idx = old_unknown_idx + len(years_to_add)
        new_size        = new_unknown_idx + 1

        # Update vocab in-place
        for year, idx in new_year_indices.items():
            self.year_vocab[year] = idx
        self.year_vocab["unknown"] = new_unknown_idx
        self.year_length = new_size

        # --- Grow the embedding weight matrix ---
        old_embedding = self.predictor.model.year_embedding
        embedding_dim = old_embedding.embedding_dim

        new_embedding = nn.Embedding(new_size, embedding_dim)
        with torch.no_grad():
            # Copy existing rows (all rows up to and including old unknown)
            new_embedding.weight[:old_size] = old_embedding.weight

            # The unknown row moved from old_unknown_idx → new_unknown_idx;
            # rows old_unknown_idx .. new_unknown_idx-1 are new years —
            # initialise them from N(0,1) (already done by default __init__).
            # Overwrite the new unknown position with the old unknown embedding
            # so it retains its learned representation.
            new_embedding.weight[new_unknown_idx] = old_embedding.weight[old_unknown_idx]

        new_embedding = new_embedding.to(self.predictor.device)
        self.predictor.model.year_embedding = new_embedding

        print(f"extend_year_vocab: added {len(years_to_add)} year(s): {years_to_add}")
        print(f"  year_embedding size: {old_size} → {new_size} rows  "
              f"(embedding_dim={embedding_dim})")
        print(f"  unknown token index: {old_unknown_idx} → {new_unknown_idx}")

        # Persist updated vocab
        vocab_path = os.path.join(self.directory, "year_vocab.json")
        with open(vocab_path, "w") as f:
            json.dump(self.year_vocab, f)
        print(f"  Updated year_vocab saved to {vocab_path}")

        return self

    def save_model(self):
        """Save model, scalers, and vocabularies"""
        os.makedirs(self.directory, exist_ok=True)
        
        # Save model checkpoint
        torch.save({
            'model_state_dict': self.predictor.model.state_dict(),
            'optimizer_state_dict': self.predictor.optimizer.state_dict(),
            'results': self.results,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'property_dim': self.property_dim,
            'continuous_time_dim': self.continuous_time_dim,
            'market_dim': self.market_dim,
            'n_communities': self.n_communities,
            # community_embedding_size = n_communities + 1 (includes the unknown index slot)
            'community_embedding_size': self.n_communities + 1,
            'year_length': self.year_length,
            'week_length': self.week_length,
            'learning_rate': self.learning_rate,
            'pooling_strategy': self.pooling_strategy,
            'use_neighborhood_pooling': self.use_neighborhood_pooling,
            'reference_date': self.reference_date.isoformat() if self.reference_date else None
        }, f'{self.directory}/model.pth')
        
        # Save vocabularies
        with open(f'{self.directory}/year_vocab.json', 'w') as f:
            json.dump(self.year_vocab, f)
        with open(f'{self.directory}/week_vocab.json', 'w') as f:
            json.dump(self.week_vocab, f)
        
        # Save results
        with open(f'{self.directory}/results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Model saved to {self.directory}")
        print(f"  Using neighborhood pooling: {self.use_neighborhood_pooling}")
        if self.use_neighborhood_pooling:
            print(f"  Pooling strategy: {self.pooling_strategy}")
        
        return self
    
    def load_model(self, directory):
        """Load saved model and artifacts"""
        from pathlib import Path
        directory = Path(directory)
        self.directory = directory
        
        # Load checkpoint
        ckpt = torch.load(directory / "model.pth", map_location=self.device)
        
        # Restore architecture params
        self.embedding_dim = ckpt['embedding_dim']
        self.hidden_dim = ckpt['hidden_dim']
        self.property_dim = ckpt['property_dim']
        self.continuous_time_dim = ckpt['continuous_time_dim']
        self.market_dim = ckpt['market_dim']
        self.n_communities = ckpt['n_communities']
        self.year_length = ckpt['year_length']
        self.week_length = ckpt['week_length']
        self.learning_rate = ckpt['learning_rate']
        self.pooling_strategy = ckpt.get('pooling_strategy', 'mean')
        self.use_neighborhood_pooling = ckpt.get('use_neighborhood_pooling', False)
        
        if ckpt.get('reference_date'):
            self.reference_date = pd.to_datetime(ckpt['reference_date'])
        
        # Recreate predictor
        # community_embedding_size = n_communities + 1 (includes unknown slot).
        # Fall back to n_communities + 1 for older checkpoints that didn't save this key.
        community_embedding_size = ckpt.get('community_embedding_size', self.n_communities + 1)
        self.predictor = price_predictor(
            self.device, self.embedding_dim, self.hidden_dim,
            self.property_dim, self.continuous_time_dim, self.market_dim,
            community_embedding_size, self.year_length, self.week_length,
            self.learning_rate,
            use_neighborhood_pooling=self.use_neighborhood_pooling,
            pooling_strategy=self.pooling_strategy
        )
        
        # Load weights
        self.predictor.model.load_state_dict(ckpt['model_state_dict'])
        self.predictor.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.predictor.eval()
        
        # Load scalers
        for feature in ['sqft', 'sqft_lot', 'beds', 'time_trend', 'sin_day', 'cos_day', 
                       'sin_month', 'cos_month', 'mortgage_rate', 'unemployment_rate', 'log_price']:
            scaler_path = directory / f"{feature}_scaler.pkl"
            if scaler_path.exists():
                self.scalers[feature] = joblib.load(scaler_path)
        
        # Load vocabularies
        with open(directory / "year_vocab.json", "r") as f:
            self.year_vocab = json.load(f)
        with open(directory / "week_vocab.json", "r") as f:
            self.week_vocab = json.load(f)
        
        # Load results
        self.results = ckpt.get("results", {})
        
        print(f"Model loaded from {directory}")
        print(f"  Using neighborhood pooling: {self.use_neighborhood_pooling}")
        if self.use_neighborhood_pooling:
            print(f"  Pooling strategy: {self.pooling_strategy}")
        
        return self
