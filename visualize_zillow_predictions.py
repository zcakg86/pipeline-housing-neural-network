"""
Visualize Zillow price predictions
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def create_visualizations():
    """Create visualizations of Zillow predictions"""
    
    print("Creating visualizations...")
    
    # Load predictions
    df = pd.read_csv('data/zillow_with_predictions.csv')
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Predicted vs Actual scatter plot
    ax1 = plt.subplot(2, 3, 1)
    ax1.scatter(df['sale_price']/1e6, df['predicted_price']/1e6, alpha=0.6, s=100)
    
    # Add diagonal line (perfect prediction)
    max_price = max(df['sale_price'].max(), df['predicted_price'].max()) / 1e6
    ax1.plot([0, max_price], [0, max_price], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax1.set_xlabel('Zillow Listing Price ($M)', fontsize=11)
    ax1.set_ylabel('Model Predicted Price ($M)', fontsize=11)
    ax1.set_title('Predicted vs Listed Price', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Error distribution
    ax2 = plt.subplot(2, 3, 2)
    ax2.hist(df['pct_error'], bins=20, edgecolor='black', alpha=0.7)
    ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax2.axvline(df['pct_error'].median(), color='green', linestyle='--', linewidth=2, 
                label=f'Median: {df["pct_error"].median():.1f}%')
    ax2.set_xlabel('Prediction Error (%)', fontsize=11)
    ax2.set_ylabel('Count', fontsize=11)
    ax2.set_title('Distribution of Prediction Errors', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Error by price range
    ax3 = plt.subplot(2, 3, 3)
    
    # Create price bins
    df['price_bin'] = pd.cut(df['sale_price'], bins=[0, 500e3, 1e6, 2e6, 5e6], 
                              labels=['<$500K', '$500K-$1M', '$1M-$2M', '>$2M'])
    
    error_by_bin = df.groupby('price_bin')['pct_error'].apply(lambda x: x.abs().mean())
    
    ax3.bar(range(len(error_by_bin)), error_by_bin.values, edgecolor='black', alpha=0.7)
    ax3.set_xticks(range(len(error_by_bin)))
    ax3.set_xticklabels(error_by_bin.index, rotation=45, ha='right')
    ax3.set_ylabel('Mean Absolute % Error', fontsize=11)
    ax3.set_title('Error by Price Range', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for i, v in enumerate(error_by_bin.values):
        ax3.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 4. Error by property size
    ax4 = plt.subplot(2, 3, 4)
    
    # Create sqft bins
    df['sqft_bin'] = pd.cut(df['sqft'], bins=[0, 1000, 2000, 3000, 10000], 
                             labels=['<1K', '1K-2K', '2K-3K', '>3K'])
    
    error_by_sqft = df.groupby('sqft_bin')['pct_error'].apply(lambda x: x.abs().mean())
    
    ax4.bar(range(len(error_by_sqft)), error_by_sqft.values, edgecolor='black', alpha=0.7, color='orange')
    ax4.set_xticks(range(len(error_by_sqft)))
    ax4.set_xticklabels([f'{label} sqft' for label in error_by_sqft.index], rotation=45, ha='right')
    ax4.set_ylabel('Mean Absolute % Error', fontsize=11)
    ax4.set_title('Error by Property Size', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for i, v in enumerate(error_by_sqft.values):
        ax4.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 5. Uncertainty intervals
    ax5 = plt.subplot(2, 3, 5)
    
    # Sort by listing price for better visualization
    df_sorted = df.sort_values('sale_price').reset_index(drop=True)
    
    # Plot every 2nd property to avoid clutter
    indices = range(0, len(df_sorted), max(1, len(df_sorted)//20))
    
    for idx in indices:
        row = df_sorted.iloc[idx]
        ax5.plot([idx, idx], [row['price_lower_95']/1e6, row['price_upper_95']/1e6], 
                'b-', alpha=0.3, linewidth=2)
        ax5.scatter(idx, row['predicted_price']/1e6, color='blue', s=50, zorder=3)
        ax5.scatter(idx, row['sale_price']/1e6, color='red', s=50, marker='x', zorder=3)
    
    ax5.set_xlabel('Property Index (sorted by price)', fontsize=11)
    ax5.set_ylabel('Price ($M)', fontsize=11)
    ax5.set_title('Prediction Uncertainty (95% CI)', fontsize=12, fontweight='bold')
    ax5.legend(['95% CI', 'Predicted', 'Listed'], loc='upper left')
    ax5.grid(True, alpha=0.3)
    
    # 6. Model attention weights
    ax6 = plt.subplot(2, 3, 6)
    
    attention_features = ['Community', 'Year', 'Week', 'Property', 'Time', 'Market']
    attention_cols = ['cls_attn_community', 'cls_attn_year', 'cls_attn_week', 
                     'cls_attn_property', 'cls_attn_time', 'cls_attn_market']
    
    attention_values = [df[col].mean() for col in attention_cols]
    
    colors = plt.cm.Set3(range(len(attention_features)))
    bars = ax6.bar(range(len(attention_features)), attention_values, 
                   edgecolor='black', alpha=0.8, color=colors)
    ax6.set_xticks(range(len(attention_features)))
    ax6.set_xticklabels(attention_features, rotation=45, ha='right')
    ax6.set_ylabel('Average Attention Weight', fontsize=11)
    ax6.set_title('Model Feature Attention', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for i, v in enumerate(attention_values):
        ax6.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    
    # Save figure
    output_file = 'outputs/zillow_predictions_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved visualization to {output_file}")
    
    plt.close()
    
    # Create a second figure for geographic analysis
    fig2, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Map of predictions
    ax1 = axes[0]
    scatter = ax1.scatter(df['lng'], df['lat'], 
                         c=df['pct_error'], 
                         s=df['sqft']/10,
                         cmap='RdYlGn_r', 
                         vmin=-80, vmax=80,
                         alpha=0.7, 
                         edgecolors='black', 
                         linewidth=0.5)
    ax1.set_xlabel('Longitude', fontsize=11)
    ax1.set_ylabel('Latitude', fontsize=11)
    ax1.set_title('Geographic Distribution of Prediction Errors', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Prediction Error (%)', fontsize=10)
    
    # Map of prices
    ax2 = axes[1]
    scatter2 = ax2.scatter(df['lng'], df['lat'], 
                          c=df['sale_price']/1e6, 
                          s=df['sqft']/10,
                          cmap='viridis', 
                          alpha=0.7, 
                          edgecolors='black', 
                          linewidth=0.5)
    ax2.set_xlabel('Longitude', fontsize=11)
    ax2.set_ylabel('Latitude', fontsize=11)
    ax2.set_title('Geographic Distribution of Listing Prices', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('Listing Price ($M)', fontsize=10)
    
    plt.tight_layout()
    
    output_file2 = 'outputs/zillow_predictions_geographic.png'
    plt.savefig(output_file2, dpi=300, bbox_inches='tight')
    print(f"✓ Saved geographic visualization to {output_file2}")
    
    plt.close()


if __name__ == "__main__":
    create_visualizations()
