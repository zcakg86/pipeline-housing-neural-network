import torch
import torch.nn as nn
import numpy as np

class ModelAnalyzer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.attention_history = []
        self.feature_gradients = []

    def hook_attention_weights(self):
        """Setup hook to capture attention weights during training/inference"""

        def attention_hook(module, input, output):
            if isinstance(output, tuple) and len(output) > 1:
                self.attention_history.append(output[1].detach().cpu())

        # Register hook on attention layer
        self.attention_hook_handle = self.model.attention_layer.register_forward_hook(attention_hook)

    def remove_hooks(self):
        """Remove all hooks"""
        if hasattr(self, 'attention_hook_handle'):
            self.attention_hook_handle.remove()

    def analyze_attention_patterns(self, dataloader, num_batches=10):
        """Analyze attention patterns across multiple batches"""
        self.model.eval()
        self.model.save_intermediates = True
        attention_stats = {
            'mean_weights': [],
            'std_weights': [],
            'max_weights': [],
            'feature_attention_map': None
        }

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break

                community_indices, community_features, year, week, property_features, targets = batch

                # Move to device
                inputs = [
                    community_indices.to(self.device),
                    community_features.to(self.device),
                    year.to(self.device),
                    week.to(self.device),
                    property_features.to(self.device)
                ]

                # Forward pass
                _ = self.model(*inputs)

                # Collect attention weights
                if self.model.last_attention_weights is not None:
                    weights = self.model.last_attention_weights.cpu().numpy()
                    attention_stats['mean_weights'].append(float(np.mean(weights)))
                    attention_stats['std_weights'].append(float(np.std(weights)))
                    attention_stats['max_weights'].append(float(np.max(weights)))

                    if attention_stats['feature_attention_map'] is None:
                        attention_stats['feature_attention_map'] = weights
                    else:
                        attention_stats['feature_attention_map'] += weights

        # Average the feature attention map
        if attention_stats['feature_attention_map'] is not None:
            # x /= y is x = x / y: 
            # min as weights will only be for first num_batches 
            attention_stats['feature_attention_map'] /= min(num_batches, len(dataloader))

        self.model.save_intermediates = False
        return attention_stats

    def compute_feature_importance_gradients(self, dataloader, num_batches=10):
        """Compute feature importance using gradient-based methods"""
        self.model.eval()

        feature_importance = {
            'community_features': [],
            'property_features': [],
            'embeddings': {
                'community': [],
                'year': [],
                'week': []
            }
        }

        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break

            community_indices, community_features, year, week, property_features, targets = batch

            # Move to device and enable gradients
            community_indices = community_indices.to(self.device)
            community_features = community_features.to(self.device).requires_grad_(True)
            year = year.to(self.device)
            week = week.to(self.device)
            property_features = property_features.to(self.device).requires_grad_(True)
            targets = targets.to(self.device)

            # Forward pass
            self.model.zero_grad()
            outputs = self.model(community_indices, community_features, year, week, property_features)

            # Compute gradients
            loss = nn.functional.mse_loss(outputs.squeeze(), targets)
            loss.backward()

            # Collect gradient magnitudes
            if community_features.grad is not None:
                feature_importance['community_features'].append(
                    community_features.grad.abs().mean(dim=0).cpu().numpy()
                )
            if property_features.grad is not None:
                feature_importance['property_features'].append(
                    property_features.grad.abs().mean(dim=0).cpu().numpy()
                )
            # Collect gradient magnitudes for embeddings
            if self.model.community_embedding.weight.grad is not None:
                feature_importance['embeddings']['community'].append(
                    self.model.community_embedding.weight.grad.abs().mean().item()
                )
            if self.model.year_embedding.weight.grad is not None:
                feature_importance['embeddings']['year'].append(
                    self.model.year_embedding.weight.grad.abs().mean().item()
                )
            if self.model.week_embedding.weight.grad is not None:
                feature_importance['embeddings']['week'].append(
                    self.model.week_embedding.weight.grad.abs().mean().item()
                )

            # Clean up
            self.model.zero_grad()

        # Aggregate results
        for key in ['community_features', 'property_features']:
            if feature_importance[key]:
                feature_importance[key] = np.mean(feature_importance[key], axis=0)

        for key in ['community', 'year', 'week']:
            if feature_importance['embeddings'][key]:
                feature_importance['embeddings'][key] = float(np.mean(feature_importance['embeddings'][key]))

        # convert property_feature array to list
        feature_importance['property_features']=list(map(float,feature_importance['property_features']))
        return feature_importance

    def visualize_attention_analysis(self, attention_stats):
        """Create comprehensive visualization of attention patterns"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Attention weight distribution
        axes[0, 0].hist(attention_stats['mean_weights'], bins=30, alpha=0.7, color='blue')
        axes[0, 0].set_xlabel('Mean Attention Weight')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Mean Attention Weights')

        # 2. Attention variability
        axes[0, 1].scatter(attention_stats['mean_weights'], attention_stats['std_weights'], alpha=0.6)
        axes[0, 1].set_xlabel('Mean Attention Weight')
        axes[0, 1].set_ylabel('Std of Attention Weight')
        axes[0, 1].set_title('Attention Weight Mean vs Variability')

        # 3. Feature attention heatmap
        if attention_stats['feature_attention_map'] is not None:
            im = axes[1, 0].imshow(attention_stats['feature_attention_map'].squeeze(),
                                   cmap='YlOrRd', aspect='auto')
            axes[1, 0].set_title('Attention Weight Heatmap')
            axes[1, 0].set_xlabel('Feature Dimension')
            axes[1, 0].set_ylabel('Batch Sample')
            plt.colorbar(im, ax=axes[1, 0])

        # 4. Max attention weights over time
        axes[1, 1].plot(attention_stats['max_weights'])
        axes[1, 1].set_xlabel('Batch Number')
        axes[1, 1].set_ylabel('Max Attention Weight')
        axes[1, 1].set_title('Maximum Attention Weights Across Batches')

        plt.tight_layout()
        plt.show()