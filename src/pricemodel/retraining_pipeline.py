"""
Automated Retraining Pipeline
Handles incremental model updates as new data arrives
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import os
import sys

sys.path.insert(0, os.getcwd())
from market_indicators import MarketIndicatorFetcher, TimeFeatureEngineer
from embedding_model_v2 import dataset
from model_manager_v2 import modelmanager


class RetrainingPipeline:
    """
    Manages model retraining with sliding windows and performance monitoring
    """
    
    def __init__(self, base_data_path='data/sales_202025.csv',
                 community_map_path='data/community_map.json',
                 model_registry_path='outputs/model_registry.json'):
        self.base_data_path = base_data_path
        self.community_map_path = community_map_path
        self.model_registry_path = model_registry_path
        self.model_registry = self._load_registry()
        
    def _load_registry(self):
        """Load model registry tracking all trained models"""
        if os.path.exists(self.model_registry_path):
            with open(self.model_registry_path, 'r') as f:
                return json.load(f)
        return {'models': [], 'active_model': None}
    
    def _save_registry(self):
        """Save model registry"""
        os.makedirs(os.path.dirname(self.model_registry_path), exist_ok=True)
        with open(self.model_registry_path, 'w') as f:
            json.dump(self.model_registry, f, indent=2)
    
    def load_and_prepare_data(self, data_path=None, sliding_window_years=5):
        """
        Load data with optional sliding window
        """
        if data_path is None:
            data_path = self.base_data_path
        
        print(f"Loading data from {data_path}...")
        df = pd.read_csv(data_path)
        df['sale_date'] = pd.to_datetime(df['sale_date'])
        
        # Apply sliding window if specified
        if sliding_window_years:
            cutoff_date = df['sale_date'].max() - timedelta(days=365 * sliding_window_years)
            df = df[df['sale_date'] >= cutoff_date]
            print(f"Applied {sliding_window_years}-year sliding window: {len(df)} records from {df['sale_date'].min()} to {df['sale_date'].max()}")
        
        # Add market indicators
        market_fetcher = MarketIndicatorFetcher()
        df = market_fetcher.merge_indicators_to_sales(df)
        df = market_fetcher.add_local_inventory(df)
        
        # Add time features
        time_engineer = TimeFeatureEngineer()
        df = time_engineer.add_continuous_time_features(df)
        df = time_engineer.add_market_momentum_features(df)
        
        # Add community mapping
        with open(self.community_map_path, 'r') as f:
            community_map = json.load(f)
        df['community'] = df['h3_07'].map(community_map).astype(str)
        
        return df
    
    def should_retrain(self, performance_threshold=15.0, time_threshold_days=30):
        """
        Determine if model should be retrained based on:
        1. Performance degradation
        2. Time since last training
        3. Availability of new data
        """
        if not self.model_registry['active_model']:
            print("No active model found. Retraining required.")
            return True, "No active model"
        
        active_model = self.model_registry['active_model']
        
        # Check time since last training
        last_train_date = datetime.fromisoformat(active_model['trained_date'])
        days_since_training = (datetime.now() - last_train_date).days
        
        if days_since_training > time_threshold_days:
            print(f"Model is {days_since_training} days old (threshold: {time_threshold_days})")
            return True, f"Time threshold exceeded: {days_since_training} days"
        
        # Check performance
        if 'validation_error' in active_model:
            if active_model['validation_error'] > performance_threshold:
                print(f"Model performance degraded: {active_model['validation_error']:.2f}% error")
                return True, f"Performance threshold exceeded: {active_model['validation_error']:.2f}%"
        
        print("Model is current. No retraining needed.")
        return False, "Model is current"
    
    def train_new_model(self, df, model_config=None, continue_from_checkpoint=None):
        """
        Train a new model or continue training from checkpoint
        """
        if model_config is None:
            model_config = {
                'embedding_dim': 16,
                'hidden_dim': 32,
                'property_dim': 3,
                'continuous_time_dim': 5,
                'market_dim': 2,
                'epochs': 50,
                'batch': 256,
                'learning_rate': 0.001,
                'dropout_rate': 0.1,
                'patience': 10
            }
        
        # Prepare dataset
        data = dataset()
        data = data._prepare_data(df, include_market_indicators=True)
        
        # Initialize or load model
        if continue_from_checkpoint:
            print(f"Loading checkpoint from {continue_from_checkpoint}")
            model = modelmanager()
            model.load_model(continue_from_checkpoint)
            # Process new data with existing scalers
            model.processor(data, scale_mode="transform")
        else:
            print("Training new model from scratch")
            model = modelmanager()
            model.processor(data)
        
        # Split and train
        model.split_data(train_ratio=0.8, temporal_split=False)
        model.train_model(**model_config)
        
        # Evaluate
        model.add_predictions_to_data(return_uncertainty=True)
        
        # Calculate metrics
        metrics = {
            'mean_abs_pct_error': float(model.dataframe['pct_error'].abs().mean()),
            'median_abs_pct_error': float(model.dataframe['pct_error'].abs().median()),
            'train_loss': float(model.results['train_losses'][-1]),
            'val_loss': float(model.results['val_losses'][-1])
        }
        
        if 'price_lower_95' in model.dataframe.columns:
            in_ci = ((model.dataframe['sale_price'] >= model.dataframe['price_lower_95']) & 
                    (model.dataframe['sale_price'] <= model.dataframe['price_upper_95']))
            metrics['ci_coverage'] = float(in_ci.mean() * 100)
        
        # Save model
        model.save_model()
        
        # Update registry
        model_entry = {
            'model_path': str(model.directory),
            'trained_date': datetime.now().isoformat(),
            'data_range': {
                'start': str(df['sale_date'].min()),
                'end': str(df['sale_date'].max())
            },
            'n_records': len(df),
            'metrics': metrics,
            'config': model_config
        }
        
        self.model_registry['models'].append(model_entry)
        self.model_registry['active_model'] = model_entry
        self._save_registry()
        
        print(f"\nModel trained and registered:")
        print(f"  Path: {model.directory}")
        print(f"  Mean Abs % Error: {metrics['mean_abs_pct_error']:.2f}%")
        print(f"  Validation Loss: {metrics['val_loss']:.4f}")
        
        return model
    
    def incremental_retrain(self, new_data_path=None, epochs=10):
        """
        Perform incremental retraining on new data
        """
        if not self.model_registry['active_model']:
            print("No active model for incremental training. Use train_new_model() instead.")
            return None
        
        # Load new data
        df = self.load_and_prepare_data(new_data_path)
        
        # Continue from last checkpoint
        checkpoint_path = self.model_registry['active_model']['model_path']
        
        # Reduced epochs for incremental update
        config = self.model_registry['active_model']['config'].copy()
        config['epochs'] = epochs
        config['learning_rate'] = config['learning_rate'] * 0.1  # Lower LR for fine-tuning
        
        print(f"Performing incremental retraining from {checkpoint_path}")
        model = self.train_new_model(df, model_config=config, continue_from_checkpoint=checkpoint_path)
        
        return model
    
    def get_active_model(self):
        """Load the currently active model"""
        if not self.model_registry['active_model']:
            raise ValueError("No active model found. Train a model first.")
        
        model_path = self.model_registry['active_model']['model_path']
        print(f"Loading active model from {model_path}")
        
        model = modelmanager()
        model.load_model(model_path)
        
        return model
    
    def compare_models(self, n_recent=5):
        """Compare performance of recent models"""
        if not self.model_registry['models']:
            print("No models in registry")
            return None
        
        recent_models = self.model_registry['models'][-n_recent:]
        
        comparison = []
        for m in recent_models:
            comparison.append({
                'trained_date': m['trained_date'],
                'mean_error': m['metrics']['mean_abs_pct_error'],
                'val_loss': m['metrics']['val_loss'],
                'n_records': m['n_records'],
                'path': m['model_path']
            })
        
        df = pd.DataFrame(comparison)
        print("\nRecent Model Comparison:")
        print(df.to_string(index=False))
        
        return df


def main():
    """Example usage of retraining pipeline"""
    pipeline = RetrainingPipeline()
    
    # Check if retraining is needed
    should_retrain, reason = pipeline.should_retrain(
        performance_threshold=15.0,
        time_threshold_days=30
    )
    
    if should_retrain:
        print(f"\nRetraining triggered: {reason}")
        
        # Load and prepare data
        df = pipeline.load_and_prepare_data(sliding_window_years=5)
        
        # Train new model
        model = pipeline.train_new_model(df)
        
        print("\nRetraining complete!")
    else:
        print("\nNo retraining needed. Loading active model...")
        model = pipeline.get_active_model()
    
    # Compare recent models
    pipeline.compare_models(n_recent=5)
    
    return pipeline, model


if __name__ == "__main__":
    pipeline, model = main()
