class LayerAblation:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.original_state = model.state_dict()

    def create_skip_connections(self, layers_to_skip):
        """Create a modified forward function with skip connections"""
        original_forward = self.model.forward

        def modified_forward(community_indices, community_features, year, week, property_features, targets=None):
            # Embeddings (always kept)
            community_embeddings = self.model.community_embedding(community_indices)
            year_embeddings = self.model.year_embedding(year)
            week_embeddings = self.model.week_embedding(week)
            combined_embeddings = torch.cat([community_embeddings, year_embeddings, week_embeddings], dim=-1)

            # Feature Processing (can be skipped)
            if 'community_feature_layer' not in layers_to_skip:
                processed_community_features = self.model.relu(
                    self.model.community_feature_layer(community_features)
                )
            else:
                # Create zero features of appropriate size
                processed_community_features = torch.zeros(
                    community_features.size(0),
                    self.model.hidden_dim
                ).to(self.device)

            if 'property_feature_layer' not in layers_to_skip:
                processed_property_features = self.model.relu(
                    self.model.property_feature_layer(property_features)
                )
            else:
                processed_property_features = torch.zeros(
                    property_features.size(0),
                    self.model.hidden_dim
                ).to(self.device)

            # Combine features
            combined_features = torch.cat([
                combined_embeddings,
                processed_community_features,
                processed_property_features
            ], dim=-1)

            # Attention (can be skipped)
            if 'attention' not in layers_to_skip:
                combined_features = combined_features.unsqueeze(1)
                attention_output, _ = self.model.attention_layer(
                    combined_features, combined_features, combined_features
                )
                features = attention_output.squeeze(1)
            else:
                features = combined_features
                # If skipping attention, need to ensure correct dimension for next layer
                if features.size(-1) != self.model.embed_dim_attention:
                    # Create a projection layer to match dimensions
                    projection = nn.Linear(features.size(-1), self.model.embed_dim_attention).to(self.device)
                    features = projection(features)

            # Hidden layers (can be individually skipped)
            if 'hidden1' not in layers_to_skip:
                features = self.model.relu(self.model.hidden_layer1(features))
            else:
                # If skipping hidden1, need to project to correct dimension
                if features.size(-1) != self.model.hidden_dim:
                    projection = nn.Linear(features.size(-1), self.model.hidden_dim).to(self.device)
                    features = projection(features)

            if 'hidden2' not in layers_to_skip:
                features = self.model.relu(self.model.hidden_layer2(features))

            # Output layer (always kept)
            output = self.model.output_layer(features)

            # Return output and shape to match original signature
            return output, features.shape

        return modified_forward

    def run_ablation_study(self, val_loader, layers_to_test=None):
        """Run systematic ablation study on specified layers"""
        if layers_to_test is None:
            layers_to_test = [
                'community_feature_layer',
                'property_feature_layer',
                'attention',
                'hidden1',
                'hidden2'
            ]

        results = {}

        # Baseline performance
        print("Evaluating baseline model...")
        baseline_results = self.evaluate_model(self.model, val_loader)
        results['baseline'] = baseline_results

        # Test each layer
        for layer_name in layers_to_test:
            print(f"\nTesting model without {layer_name}...")

            # Create modified forward function
            modified_forward = self.create_skip_connections([layer_name])

            # Temporarily replace forward method
            original_forward = self.model.forward
            self.model.forward = modified_forward

            # Evaluate
            layer_results = self.evaluate_model(self.model, val_loader)

            # Restore original forward
            self.model.forward = original_forward

            # Calculate performance drop
            performance_drop = {
                'mse_increase': (layer_results['mse'] - baseline_results['mse']) / baseline_results['mse'] * 100,
                'mae_increase': (layer_results['mae'] - baseline_results['mae']) / baseline_results['mae'] * 100,
                'loss_increase': (layer_results['loss'] - baseline_results['loss']) / baseline_results['loss'] * 100
            }

            layer_results['performance_drop'] = performance_drop
            results[f'without_{layer_name}'] = layer_results

        # Test combinations of layers
        print("\nTesting layer combinations...")
        combinations = [
            ['community_feature_layer', 'property_feature_layer'],
            ['hidden1', 'hidden2'],
            ['attention', 'hidden1'],
            ['attention', 'hidden2']
        ]

        for combo in combinations:
            combo_name = '_'.join(combo)
            print(f"\nTesting without {combo_name}...")

            modified_forward = self.create_skip_connections(combo)
            original_forward = self.model.forward
            self.model.forward = modified_forward

            combo_results = self.evaluate_model(self.model, val_loader)
            self.model.forward = original_forward

            performance_drop = {
                'mse_increase': (combo_results['mse'] - baseline_results['mse']) / baseline_results['mse'] * 100,
                'mae_increase': (combo_results['mae'] - baseline_results['mae']) / baseline_results['mae'] * 100,
                'loss_increase': (combo_results['loss'] - baseline_results['loss']) / baseline_results['loss'] * 100
            }

            combo_results['performance_drop'] = performance_drop
            results[f'without_{combo_name}'] = combo_results

        return results

    def evaluate_model(self, model, dataloader):
        """Evaluate model performance"""
        model.eval()
        total_loss = 0
        predictions = []
        actuals = []
        criterion = nn.MSELoss()

        with torch.no_grad():
            for batch in dataloader:
                # Move to device
                community_indices = batch[0].to(self.device)
                community_features = batch[1].to(self.device)
                year = batch[2].to(self.device)
                week = batch[3].to(self.device)
                property_features = batch[4].to(self.device)
                targets = batch[5].to(self.device)

                # Forward pass
                outputs, _ = model(community_indices, community_features, year,
                                   week, property_features, targets)
                loss = criterion(outputs.squeeze(), targets)

                total_loss += loss.item()
                predictions.extend(outputs.squeeze().cpu().numpy())
                actuals.extend(targets.cpu().numpy())

        # Calculate metrics
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        mse = np.mean((predictions - actuals) ** 2)
        mae = np.mean(np.abs(predictions - actuals))
        mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
        r2 = 1 - (np.sum((actuals - predictions) ** 2) / np.sum((actuals - np.mean(actuals)) ** 2))

        return {
            'loss': total_loss / len(dataloader),
            'mse': mse,
            'mae': mae,
            'mape': mape,
            'r2': r2,
            'predictions': predictions,
            'actuals': actuals
        }

    def visualize_ablation_results(self, results):
        """Create comprehensive visualization of ablation study results"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd

        # Prepare data for visualization
        data = []
        for config, result in results.items():
            if config != 'baseline':
                layer_removed = config.replace('without_', '')
                data.append({
                    'Configuration': layer_removed,
                    'MSE Increase (%)': result['performance_drop']['mse_increase'],
                    'MAE Increase (%)': result['performance_drop']['mae_increase'],
                    'Loss Increase (%)': result['performance_drop']['loss_increase'],
                    'R² Score': result['r2']
                })

        df = pd.DataFrame(data)
        df = df.sort_values('MSE Increase (%)', ascending=False)

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Bar plot of performance drops
        ax1 = axes[0, 0]
        x = np.arange(len(df))
        width = 0.25

        ax1.bar(x - width, df['MSE Increase (%)'], width, label='MSE', alpha=0.8)
        ax1.bar(x, df['MAE Increase (%)'], width, label='MAE', alpha=0.8)
        ax1.bar(x + width, df['Loss Increase (%)'], width, label='Loss', alpha=0.8)

        ax1.set_xlabel('Layer(s) Removed')
        ax1.set_ylabel('Performance Drop (%)')
        ax1.set_title('Impact of Layer Removal on Model Performance')
        ax1.set_xticks(x)
        ax1.set_xticklabels(df['Configuration'], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Heatmap of metrics
        ax2 = axes[0, 1]
        metrics_df = df.set_index('Configuration')[['MSE Increase (%)', 'MAE Increase (%)', 'Loss Increase (%)']].T
        sns.heatmap(metrics_df, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax2, cbar_kws={'label': 'Increase (%)'})
        ax2.set_title('Performance Drop Heatmap')
        ax2.set_xlabel('Layer(s) Removed')

        # 3. R² scores comparison
        ax3 = axes[1, 0]
        baseline_r2 = results['baseline']['r2']

        # Add baseline to dataframe for comparison
        r2_data = pd.DataFrame([{'Configuration': 'baseline', 'R² Score': baseline_r2}])
        r2_data = pd.concat([r2_data, df[['Configuration', 'R² Score']]], ignore_index=True)

        ax3.bar(r2_data['Configuration'], r2_data['R² Score'])
        ax3.axhline(y=baseline_r2, color='r', linestyle='--', label='Baseline R²')
        ax3.set_xlabel('Configuration')
        ax3.set_ylabel('R² Score')
        ax3.set_title('R² Score Comparison')
        ax3.set_xticklabels(r2_data['Configuration'], rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Scatter plot of predictions vs actuals for worst performing config
        ax4 = axes[1, 1]
        worst_config = df.iloc[0]['Configuration']
        worst_results = results[f'without_{worst_config}']

        ax4.scatter(worst_results['actuals'][:1000], worst_results['predictions'][:1000],
                    alpha=0.5, label=f'Without {worst_config}')
        ax4.scatter(results['baseline']['actuals'][:1000], results['baseline']['predictions'][:1000],
                    alpha=0.5, label='Baseline')

        # Add diagonal line
        min_val = min(np.min(worst_results['actuals']), np.min(worst_results['predictions']))
        max_val = max(np.max(worst_results['actuals']), np.max(worst_results['predictions']))
        ax4.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)

        ax4.set_xlabel('Actual Values')
        ax4.set_ylabel('Predicted Values')
        ax4.set_title('Predictions vs Actuals (Sample of 1000 points)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Print summary statistics
        print("\nAblation Study Summary:")
        print("-" * 60)
        print(f"{'Configuration':<30} {'MSE Inc %':<12} {'MAE Inc %':<12} {'R²':<10}")
        print("-" * 60)
        print(f"{'Baseline':<30} {'0.0':<12} {'0.0':<12} {baseline_r2:<10.4f}")
        for _, row in df.iterrows():
            print(f"{row['Configuration']:<30} {row['MSE Increase (%)']:<12.2f} "
                  f"{row['MAE Increase (%)']:<12.2f} {row['R² Score']:<10.4f}")

        return df