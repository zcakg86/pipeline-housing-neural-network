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

from .data_pipeline import vocab_replace_tensor
from .feature_contract import (
    LOCAL_FEATURES, MARKET_FEATURES, PROPERTY_FEATURES, TIME_FEATURES
)
from .trainer import PriceTrainer
from .reproducibility import seed_everything


class ModelManager:
    """Coordinate tensors, splits, training, and high-level model lifecycle.

    Evaluation/report construction and checkpoint serialization are delegated to
    focused modules. The manager retains orchestration state so existing training
    scripts and checkpoint formats remain compatible.
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
        self.local_feature_dim = len(self._LOCAL_FEATURES)
        self.local_market_snapshot = None
        self.neighbor_cells_map = None
        self.train_community_loss_weights = None
        self.val_community_loss_weights = None
        
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
        self.global_aux_weight = 0.5
        self.residual_penalty = 1e-2
        self.lr_plateau_factor = 0.5
        self.lr_plateau_patience = 3
        self.min_learning_rate = 1e-6
        self.uncertainty_calibration_epochs = 10
        self.uncertainty_patience = 3
        self.random_seed = 42
        self.train_indices = None
        self.val_indices = None
        
        # Vocabularies
        self.year_vocab = None
        self.week_vocab = None

    
    # Features that need scaling — shared across processor and scaler helpers
    _PROPERTY_FEATURES = list(PROPERTY_FEATURES)
    _TIME_FEATURES     = list(TIME_FEATURES)
    _MARKET_FEATURES   = list(MARKET_FEATURES)
    _LOCAL_FEATURES    = list(LOCAL_FEATURES)
    _TARGET_FEATURE    = 'log_price'

    def processor(self, data, scale_mode = ''):
        """
        Enhanced processor with continuous time and market features.
        Handles H3 L8 neighborhood community tensors (batch, 7).

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
        self.local_market_snapshot = getattr(data, 'local_market_snapshot', None)
        self.neighbor_cells_map = getattr(data, 'neighbor_cells_map', None)

        local_market_features = getattr(data, 'local_market_features', None)
        if self.local_feature_dim > 0:
            if local_market_features is None:
                raise ValueError(
                    "Local residual model requires dataset.local_market_features; "
                    "call dataset._map_communities() before _prepare_data()."
                )
            self._local_market_raw = np.asarray(local_market_features, dtype=np.float32)
            expected_shape = (len(self.dataframe), 7, self.local_feature_dim)
            if self._local_market_raw.shape != expected_shape:
                raise ValueError(
                    f"Expected local market tensor {expected_shape}, got "
                    f"{self._local_market_raw.shape}"
                )

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
            print(f"Using H3 L8 neighborhood pooling: community tensor shape {community_tensor.shape}")
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
        if self.local_feature_dim > 0 and hasattr(self, '_local_market_raw'):
            scaled = np.empty_like(self._local_market_raw, dtype=np.float32)
            for feature_idx, feature in enumerate(self._LOCAL_FEATURES):
                if feature not in self.scalers:
                    raise ValueError(f"Loaded checkpoint is missing local scaler '{feature}'")
                channel = self._local_market_raw[:, :, feature_idx].reshape(-1, 1)
                scaled[:, :, feature_idx] = self.scalers[feature].transform(channel).reshape(
                    len(df), 7
                )
            self._local_market_scaled = scaled

    def _fit_and_apply_scalers_on_split(self, train_indices):
        """
        Fit scalers on training rows only, then apply to the full dataframe.
        Called from split_data() so scalers never see validation data.
        Saves scaler files to self.directory as before.
        """
        # split_data() runs before train_model(), so this fresh timestamped run
        # directory does not exist yet. Scaler persistence is the first write in
        # a new training run and must create it itself.
        os.makedirs(self.directory, exist_ok=True)

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

        if self.local_feature_dim > 0:
            self._local_market_scaled = np.empty_like(self._local_market_raw, dtype=np.float32)
            for feature_idx, feature in enumerate(self._LOCAL_FEATURES):
                scaler_path = os.path.join(self.directory, f"{feature}_scaler.pkl")
                scaler = self.scalers.get(feature)
                train_values = self._local_market_raw[train_indices, :, feature_idx].reshape(-1, 1)
                if scaler is None and os.path.exists(scaler_path) and scale_mode != "force_new":
                    scaler = joblib.load(scaler_path)
                if scaler is None:
                    scaler = StandardScaler().fit(train_values)
                    joblib.dump(scaler, scaler_path)
                self.scalers[feature] = scaler
                all_values = self._local_market_raw[:, :, feature_idx].reshape(-1, 1)
                self._local_market_scaled[:, :, feature_idx] = scaler.transform(
                    all_values
                ).reshape(len(self.dataframe), 7)
                print(f"  Fitted/applied local scaler for {feature}")

        # Rebuild the TensorDataset now that scaled columns are correct
        self._build_tensor_dataset()

    def _build_tensor_dataset(self):
        """Construct self.tensors from the current scaled columns."""
        property_features = self._PROPERTY_FEATURES
        time_features     = self._TIME_FEATURES
        market_features   = self._MARKET_FEATURES

        tensors = [
            self._community_tensor,
            torch.tensor(self.dataframe['year'].values, dtype=torch.long),
            torch.tensor(self.dataframe['week'].values, dtype=torch.long),
            torch.tensor(self.dataframe[[f'{f}_scaled' for f in property_features]].values, dtype=torch.float32),
            torch.tensor(self.dataframe[[f'{f}_scaled' for f in time_features]].values, dtype=torch.float32),
            torch.tensor(self.dataframe[[f'{f}_scaled' for f in market_features]].values, dtype=torch.float32),
        ]
        if self.local_feature_dim > 0:
            tensors.append(torch.tensor(self._local_market_scaled, dtype=torch.float32))
        tensors.append(torch.tensor(self.dataframe['log_price_scaled'].values, dtype=torch.float32))
        self.tensors = TensorDataset(*tensors)
    
    def split_data(self, train_ratio=0.8, temporal_split=False):
        """
        Split data into train/val sets, then fit scalers on train rows only.

        Scaler fitting is deferred to here so that validation rows are never
        seen during fit — preventing data leakage into evaluation metrics.

        temporal_split: If True, uses chronological split instead of random.
        """
        if self.local_feature_dim > 0 and not temporal_split:
            print(
                "Local market features require rolling chronological evaluation; "
                "overriding temporal_split=True to prevent cross-split target leakage."
            )
            temporal_split = True

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

        # Retain original dataframe positions for split-specific diagnostics.
        # add_predictions_to_data() predicts in this same row order.
        self.train_indices = np.asarray(train_indices, dtype=np.int64)
        self.val_indices = np.asarray(val_indices, dtype=np.int64)

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

        if self.local_feature_dim > 0:
            def _dataset_level_community_weights(indices):
                communities = self._community_tensor[indices]
                centers = communities[:, 0] if communities.ndim == 2 else communities
                counts = torch.bincount(
                    centers,
                    minlength=self.n_communities + 1,
                ).to(torch.float32)
                active = counts > 0
                weights = torch.zeros_like(counts)
                # N / (G * n_c) gives every active community equal total
                # influence while keeping the dataset-wide mean weight at 1.
                weights[active] = (
                    len(indices) /
                    (active.sum().to(torch.float32) * counts[active])
                )
                return weights

            self.train_community_loss_weights = _dataset_level_community_weights(
                train_indices
            )
            self.val_community_loss_weights = _dataset_level_community_weights(
                val_indices
            )
            print(
                "Precomputed dataset-level community loss weights "
                f"(train groups: "
                f"{int((self.train_community_loss_weights > 0).sum())}, "
                f"val groups: {int((self.val_community_loss_weights > 0).sum())})"
            )

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
                   property_dim=None, continuous_time_dim=1, market_dim=2,
                   epochs=50, batch=256, learning_rate=0.0003,
                   dropout_rate=0.1, estimate_uncertainty=True,
                   pooling_strategy='mean', patience=20,
                   global_aux_weight=0.5, residual_penalty=1e-2,
                   lr_plateau_factor=0.5, lr_plateau_patience=3,
                   min_learning_rate=1e-6,
                   uncertainty_calibration_epochs=10,
                   uncertainty_patience=3, random_seed=42):
        """
        Train the enhanced model with H3 neighborhood pooling
        
        Args:
            pooling_strategy: 'mean', 'center_weighted', or 'learnable'
        """
        os.makedirs(self.directory, exist_ok=True)

        # Seed before constructing either the shuffled loader or model weights.
        # Supplying the generator explicitly prevents unrelated torch calls from
        # changing the training-row order.
        data_loader_generator = seed_everything(random_seed)

        valid_pooling_strategies = {'mean', 'center_weighted', 'learnable'}
        if pooling_strategy not in valid_pooling_strategies:
            raise ValueError(
                f"Unknown pooling strategy '{pooling_strategy}'; expected one of "
                f"{sorted(valid_pooling_strategies)}"
            )
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch,
            shuffle=True,
            generator=data_loader_generator,
        )
        val_loader = DataLoader(self.val_dataset, batch_size=batch, drop_last=False)
        
        # Store architecture params
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        if property_dim is None:
            property_dim = len(self._PROPERTY_FEATURES)
        if property_dim != len(self._PROPERTY_FEATURES):
            raise ValueError(
                f"property_dim={property_dim} but the configured property feature "
                f"contract has {len(self._PROPERTY_FEATURES)} fields: {self._PROPERTY_FEATURES}"
            )
        self.property_dim = property_dim
        self.continuous_time_dim = continuous_time_dim
        self.market_dim = market_dim
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.estimate_uncertainty = estimate_uncertainty
        self.global_aux_weight = global_aux_weight
        self.residual_penalty = residual_penalty
        self.lr_plateau_factor = lr_plateau_factor
        self.lr_plateau_patience = lr_plateau_patience
        self.min_learning_rate = min_learning_rate
        self.uncertainty_calibration_epochs = uncertainty_calibration_epochs
        self.uncertainty_patience = uncertainty_patience
        self.random_seed = random_seed
        # Initialize or update predictor. The manager owns the flag so the
        # predictor always receives the same value that processor() derived.
        if self.predictor is None:
            community_embedding_length = (
                self.n_communities if self.use_neighborhood_pooling
                else self.n_communities + 1
            )
            self.predictor = PriceTrainer(
                self.device, embedding_dim, hidden_dim, 
                property_dim, continuous_time_dim, market_dim,
                community_embedding_length,
                self.year_length, self.week_length,
                learning_rate, epochs, len(train_loader),
                dropout_rate, estimate_uncertainty,
                use_neighborhood_pooling=self.use_neighborhood_pooling,
                pooling_strategy=pooling_strategy,
                local_feature_dim=self.local_feature_dim,
                global_aux_weight=global_aux_weight,
                residual_penalty=residual_penalty,
                lr_plateau_factor=lr_plateau_factor,
                lr_plateau_patience=lr_plateau_patience,
                min_learning_rate=min_learning_rate,
                train_community_loss_weights=self.train_community_loss_weights,
                val_community_loss_weights=self.val_community_loss_weights,
            )
        else:
            if self.use_neighborhood_pooling:
                embedding = self.predictor.model.community_embedding
                current_strategy = embedding.pooling_strategy
                if pooling_strategy != current_strategy:
                    if 'learnable' in {pooling_strategy, current_strategy}:
                        raise ValueError(
                            "Switching to or from learnable pooling changes model "
                            "parameters; create a new model instead of continuing "
                            "training an existing predictor."
                        )
                    embedding.pooling_strategy = pooling_strategy
                    print(
                        f"Updated pooling strategy: {current_strategy} -> "
                        f"{pooling_strategy}"
                    )
            # Update learning rate for continued training
            for param_group in self.predictor.optimizer.param_groups:
                param_group['lr'] = learning_rate
            self.predictor.global_aux_weight = global_aux_weight
            self.predictor.residual_penalty = residual_penalty
            self.predictor.configure_scheduler(
                factor=lr_plateau_factor,
                patience=lr_plateau_patience,
                min_lr=min_learning_rate,
            )
            self.predictor.set_community_loss_weights(
                self.train_community_loss_weights,
                self.val_community_loss_weights,
            )

        if self.use_neighborhood_pooling:
            # The live module is authoritative, preventing checkpoint metadata
            # from drifting away from the behavior actually used in forward().
            self.pooling_strategy = (
                self.predictor.model.community_embedding.pooling_strategy
            )
        else:
            self.pooling_strategy = pooling_strategy

        start_time = datetime.now()
        if estimate_uncertainty:
            train_losses, val_losses = self.predictor.train_two_stage(
                train_loader,
                val_loader,
                mean_epochs=epochs,
                mean_patience=patience,
                uncertainty_epochs=uncertainty_calibration_epochs,
                uncertainty_patience=uncertainty_patience,
                learning_rate=learning_rate,
            )
        else:
            train_losses, val_losses = self.predictor.train(
                train_loader, val_loader, epochs, patience
            )
        end_time = datetime.now()
        
        print(f'{(end_time-start_time).total_seconds():.2f} seconds to train')
        
        self.results['train_losses'] = train_losses
        self.results['val_losses'] = val_losses
        self.results['learning_rates'] = self.predictor.learning_rates
        self.results['best_epoch'] = self.predictor.best_epoch
        self.results['best_val_loss'] = self.predictor.best_val_loss
        self.results['training_strategy'] = (
            'mse_then_frozen_uncertainty_nll' if estimate_uncertainty else 'mse'
        )
        if estimate_uncertainty:
            self.results['diagnostic_history'] = self.predictor.diagnostic_history
            self.results['best_uncertainty_epoch'] = self.predictor.best_uncertainty_epoch
            self.results['best_val_nll'] = self.predictor.best_val_nll
        
        return self
    
    def add_predictions_to_data(
        self,
        return_uncertainty=True,
        community_min_support=20,
        community_shrinkage_strength=20.0,
    ):
        """Attach prediction/report columns through the evaluation component."""
        from .evaluation import add_predictions_to_data
        return add_predictions_to_data(
            self,
            return_uncertainty=return_uncertainty,
            community_min_support=community_min_support,
            community_shrinkage_strength=community_shrinkage_strength,
        )

    def _calculate_validation_community_metrics(
        self,
        min_support=20,
        shrinkage_strength=20.0,
    ):
        """Compatibility wrapper for validation-only community evaluation."""
        from .evaluation import calculate_validation_community_metrics
        return calculate_validation_community_metrics(
            self,
            min_support=min_support,
            shrinkage_strength=shrinkage_strength,
        )
    
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

    def _synchronize_device(self):
        """Finish queued accelerator work before checkpoint operations."""
        from .checkpoint_io import synchronize_device
        return synchronize_device(self)

    def save_and_reload_for_evaluation(self):
        """Persist once and verify the exact state used for evaluation."""
        from .checkpoint_io import save_and_reload_for_evaluation
        return save_and_reload_for_evaluation(self)

    def save_model(self):
        """Atomically persist weights and immutable deployment artifacts."""
        from .checkpoint_io import save_model
        return save_model(self)

    def save_results(self):
        """Persist updated metrics without rewriting model weights."""
        from .checkpoint_io import save_results
        return save_results(self)

    def load_model(self, directory):
        """Restore weights and supporting artifacts from a checkpoint directory."""
        from .checkpoint_io import load_model
        return load_model(self, directory)


# Compatibility alias for existing imports while callers migrate.
modelmanager = ModelManager
