"""
Enhanced Model Manager V2
Handles training, prediction, and retraining pipeline
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

from embedding_model_v2 import dataset, price_predictor, vocab_replace_tensor


class modelmanager:
    """
    Enhanced model manager with retraining pipeline and uncertainty estimation
    """
    def __init__(self, model_name="property_model_v2"):
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
        
        # Vocabularies
        self.community_vocab = None
        self.year_vocab = None
        self.week_vocab = None
        
        os.makedirs(self.directory, exist_ok=True)
    
    def processor(self, data, scale_mode="fit"):
        """
        Enhanced processor with continuous time and market features
        """
        # Features to scale
        property_features = ['sqft', 'sqft_lot', 'beds']
        time_features = ['time_trend', 'sin_day', 'cos_day', 'sin_month', 'cos_month']
        market_features = ['mortgage_rate', 'unemployment_rate']
        target_feature = 'log_price'
        
        all_features = property_features + time_features + market_features + [target_feature]
        
        # Scale features
        for feature in all_features:
            if feature not in data.dataframe.columns:
                print(f"Warning: {feature} not found in dataframe")
                continue
            
            scaler_path = os.path.join(self.directory, f"{feature}_scaler.pkl")
            scaler = None
            
            # Check memory first
            if feature in self.scalers:
                scaler = self.scalers[feature]
            # Then disk
            elif os.path.exists(scaler_path) and scale_mode != "force_new":
                scaler = joblib.load(scaler_path)
                self.scalers[feature] = scaler
            
            # Apply or fit
            if scaler is not None:
                data.dataframe[f"{feature}_scaled"] = scaler.transform(data.dataframe[[feature]])
            else:
                scaler = StandardScaler()
                data.dataframe[f"{feature}_scaled"] = scaler.fit_transform(data.dataframe[[feature]])
                self.scalers[feature] = scaler
                joblib.dump(scaler, scaler_path)
        
        # Create tensors
        self.tensors = TensorDataset(
            torch.tensor(data.dataframe['community_index'].values, dtype=torch.int),
            torch.tensor(data.dataframe['year'].values, dtype=torch.int),
            torch.tensor(data.dataframe['week'].values, dtype=torch.int),
            torch.tensor(data.dataframe[[f'{f}_scaled' for f in property_features]].values, dtype=torch.float32),
            torch.tensor(data.dataframe[[f'{f}_scaled' for f in time_features]].values, dtype=torch.float32),
            torch.tensor(data.dataframe[[f'{f}_scaled' for f in market_features]].values, dtype=torch.float32),
            torch.tensor(data.dataframe['log_price_scaled'].values, dtype=torch.float32)
        )
        
        self.dataframe = data.dataframe
        self.year_length = data.year_length
        self.week_length = data.week_length
        self.n_communities = data.n_communities
        self.data_length = data.length
        self.community_vocab = data.community_vocab
        self.week_vocab = data.week_vocab
        self.year_vocab = data.year_vocab
        self.reference_date = data.reference_date
        
        return self
    
    def split_data(self, train_ratio=0.8, temporal_split=False):
        """
        Split data into train/val sets
        temporal_split: If True, uses chronological split instead of random
        """
        self.tensor_length = self.tensors.tensors[0].shape[0]
        train_size = int(train_ratio * self.tensor_length)
        val_size = self.tensor_length - train_size
        
        if temporal_split:
            # Chronological split (earlier data for training)
            train_indices = list(range(train_size))
            val_indices = list(range(train_size, self.tensor_length))
            self.train_dataset = Subset(self.tensors, train_indices)
            self.val_dataset = Subset(self.tensors, val_indices)
        else:
            # Random split
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                self.tensors, [train_size, val_size]
            )
        
        # Handle vocab consistency
        year_train_tensor = vocab_replace_tensor(
            self.train_dataset.dataset.tensors[1] if hasattr(self.train_dataset, 'dataset') else self.tensors.tensors[1],
            self.year_vocab
        )
        week_train_tensor = vocab_replace_tensor(
            self.train_dataset.dataset.tensors[2] if hasattr(self.train_dataset, 'dataset') else self.tensors.tensors[2],
            self.week_vocab
        )
        
        # Update tensors
        new_train_dataset = list(self.train_dataset.dataset.tensors if hasattr(self.train_dataset, 'dataset') else self.tensors.tensors)
        new_train_dataset[1] = year_train_tensor
        new_train_dataset[2] = week_train_tensor
        new_train_dataset = TensorDataset(*new_train_dataset)
        self.train_dataset = Subset(new_train_dataset, self.train_dataset.indices if hasattr(self.train_dataset, 'indices') else list(range(train_size)))
        
        # Same for validation
        year_val_tensor = vocab_replace_tensor(
            self.val_dataset.dataset.tensors[1] if hasattr(self.val_dataset, 'dataset') else self.tensors.tensors[1],
            self.year_vocab
        )
        week_val_tensor = vocab_replace_tensor(
            self.val_dataset.dataset.tensors[2] if hasattr(self.val_dataset, 'dataset') else self.tensors.tensors[2],
            self.week_vocab
        )
        
        new_val_dataset = list(self.val_dataset.dataset.tensors if hasattr(self.val_dataset, 'dataset') else self.tensors.tensors)
        new_val_dataset[1] = year_val_tensor
        new_val_dataset[2] = week_val_tensor
        new_val_dataset = TensorDataset(*new_val_dataset)
        self.val_dataset = Subset(new_val_dataset, self.val_dataset.indices if hasattr(self.val_dataset, 'indices') else list(range(train_size, self.tensor_length)))
        
        return self
    
    def train_model(self, embedding_dim=16, hidden_dim=32, 
                   property_dim=3, continuous_time_dim=5, market_dim=2,
                   epochs=50, batch=256, learning_rate=0.001,
                   dropout_rate=0.1, estimate_uncertainty=True,
                   patience=10):
        """
        Train the enhanced model
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
        
        # Initialize or update predictor
        if self.predictor is None:
            self.predictor = price_predictor(
                self.device, embedding_dim, hidden_dim, 
                property_dim, continuous_time_dim, market_dim,
                self.n_communities, self.year_length, self.week_length,
                learning_rate, dropout_rate, estimate_uncertainty
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
        Add predictions with uncertainty estimates and CLS attention weights to dataframe
        """
        # Prepare tensors with vocab replacement
        year_tensor = vocab_replace_tensor(self.tensors.tensors[1], self.year_vocab)
        week_tensor = vocab_replace_tensor(self.tensors.tensors[2], self.week_vocab)
        
        new_dataset = list(self.tensors.tensors)
        new_dataset[1] = year_tensor
        new_dataset[2] = week_tensor
        new_dataset = TensorDataset(*new_dataset)
        
        loader = DataLoader(new_dataset, batch_size=256)
        
        self.predictor.eval()
        
        predictions = []
        uncertainties = []
        targets = []
        cls_attentions = []  # Store CLS attention weights
        
        with torch.no_grad():
            for batch in loader:
                batch = tuple(t.to(self.predictor.device) for t in batch)
                community, year, week, property_feat, time_feat, market_feat, target = batch
                
                if return_uncertainty and self.predictor.estimate_uncertainty:
                    pred, log_var = self.predictor.model(
                        community, year, week, property_feat, time_feat, market_feat,
                        return_uncertainty=True
                    )
                    uncertainties.extend(torch.exp(log_var / 2).cpu().numpy())  # Convert log_var to std
                else:
                    pred = self.predictor.model(community, year, week, property_feat, time_feat, market_feat)
                
                predictions.extend(pred.cpu().numpy())
                targets.extend(target.cpu().numpy())
                
                # Capture CLS attention weights (averaged across attention heads)
                # Shape: (batch_size, num_heads, num_tokens)
                # We average across heads to get (batch_size, num_tokens)
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
            'year_length': self.year_length,
            'week_length': self.week_length,
            'learning_rate': self.learning_rate,
            'reference_date': self.reference_date.isoformat() if self.reference_date else None
        }, f'{self.directory}/model.pth')
        
        # Save vocabularies
        with open(f'{self.directory}/community_vocab.json', 'w') as f:
            json.dump(self.community_vocab, f)
        with open(f'{self.directory}/year_vocab.json', 'w') as f:
            json.dump(self.year_vocab, f)
        with open(f'{self.directory}/week_vocab.json', 'w') as f:
            json.dump(self.week_vocab, f)
        
        # Save results
        with open(f'{self.directory}/results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Model saved to {self.directory}")
        
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
        
        if ckpt.get('reference_date'):
            self.reference_date = pd.to_datetime(ckpt['reference_date'])
        
        # Recreate predictor
        self.predictor = price_predictor(
            self.device, self.embedding_dim, self.hidden_dim,
            self.property_dim, self.continuous_time_dim, self.market_dim,
            self.n_communities, self.year_length, self.week_length,
            self.learning_rate
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
        with open(directory / "community_vocab.json", "r") as f:
            self.community_vocab = json.load(f)
        with open(directory / "year_vocab.json", "r") as f:
            self.year_vocab = json.load(f)
        with open(directory / "week_vocab.json", "r") as f:
            self.week_vocab = json.load(f)
        
        # Load results
        self.results = ckpt.get("results", {})
        
        print(f"Model loaded from {directory}")
        
        return self
