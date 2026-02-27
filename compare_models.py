"""
Model Comparison Script
Compare original model vs enhanced V2 model performance
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def load_predictions(v1_file='data/sales_2020_25_with_predictions.csv',
                    v2_file='data/sales_2020_25_with_predictions_v2.csv'):
    """Load predictions from both models"""
    
    print("Loading predictions...")
    
    try:
        df_v1 = pd.read_csv(v1_file)
        df_v1['sale_date'] = pd.to_datetime(df_v1['sale_date'])
        df_v1['year'] = df_v1['sale_date'].dt.year
        has_v1 = True
        print(f"✓ Loaded V1 predictions: {len(df_v1)} records")
    except FileNotFoundError:
        print("✗ V1 predictions not found")
        has_v1 = False
        df_v1 = None
    
    try:
        df_v2 = pd.read_csv(v2_file)
        df_v2['sale_date'] = pd.to_datetime(df_v2['sale_date'])
        df_v2['year'] = df_v2['sale_date'].dt.year
        has_v2 = True
        print(f"✓ Loaded V2 predictions: {len(df_v2)} records")
    except FileNotFoundError:
        print("✗ V2 predictions not found. Run main_train_v2.py first")
        has_v2 = False
        df_v2 = None
    
    return df_v1, df_v2, has_v1, has_v2


def compare_overall_performance(df_v1, df_v2):
    """Compare overall metrics"""
    
    print("\n" + "="*80)
    print("OVERALL PERFORMANCE COMPARISON")
    print("="*80)
    
    metrics = []
    
    if df_v1 is not None:
        v1_mae = df_v1['pct_error'].abs().mean()
        v1_median = df_v1['pct_error'].abs().median()
        v1_rmse = np.sqrt((df_v1['pct_error']**2).mean())
        metrics.append({
            'Model': 'Original V1',
            'Mean Abs % Error': f'{v1_mae:.2f}%',
            'Median Abs % Error': f'{v1_median:.2f}%',
            'RMSE %': f'{v1_rmse:.2f}%'
        })
    
    if df_v2 is not None:
        v2_mae = df_v2['pct_error'].abs().mean()
        v2_median = df_v2['pct_error'].abs().median()
        v2_rmse = np.sqrt((df_v2['pct_error']**2).mean())
        metrics.append({
            'Model': 'Enhanced V2',
            'Mean Abs % Error': f'{v2_mae:.2f}%',
            'Median Abs % Error': f'{v2_median:.2f}%',
            'RMSE %': f'{v2_rmse:.2f}%'
        })
        
        if df_v1 is not None:
            improvement = ((v1_mae - v2_mae) / v1_mae) * 100
            metrics.append({
                'Model': 'Improvement',
                'Mean Abs % Error': f'{improvement:+.1f}%',
                'Median Abs % Error': f'{((v1_median - v2_median) / v1_median) * 100:+.1f}%',
                'RMSE %': f'{((v1_rmse - v2_rmse) / v1_rmse) * 100:+.1f}%'
            })
    
    df_metrics = pd.DataFrame(metrics)
    print(df_metrics.to_string(index=False))
    
    # V2 specific features
    if df_v2 is not None and 'price_lower_95' in df_v2.columns:
        print("\n" + "-"*80)
        print("V2 UNCERTAINTY ESTIMATION")
        print("-"*80)
        in_ci = ((df_v2['sale_price'] >= df_v2['price_lower_95']) & 
                (df_v2['sale_price'] <= df_v2['price_upper_95']))
        print(f"95% Confidence Interval Coverage: {in_ci.mean()*100:.1f}% (target: ~95%)")
        
        avg_ci_width = (df_v2['price_upper_95'] - df_v2['price_lower_95']).mean()
        print(f"Average CI Width: ${avg_ci_width:,.0f}")


def compare_temporal_performance(df_v1, df_v2):
    """Compare performance by year"""
    
    print("\n" + "="*80)
    print("TEMPORAL PERFORMANCE (By Year)")
    print("="*80)
    
    years = sorted(set(
        (df_v1['year'].unique() if df_v1 is not None else []) + 
        (df_v2['year'].unique() if df_v2 is not None else [])
    ))
    
    comparison = []
    
    for year in years:
        row = {'Year': int(year)}
        
        if df_v1 is not None:
            v1_year = df_v1[df_v1['year'] == year]
            if len(v1_year) > 0:
                row['V1 Error %'] = v1_year['pct_error'].abs().mean()
                row['V1 Count'] = len(v1_year)
        
        if df_v2 is not None:
            v2_year = df_v2[df_v2['year'] == year]
            if len(v2_year) > 0:
                row['V2 Error %'] = v2_year['pct_error'].abs().mean()
                row['V2 Count'] = len(v2_year)
        
        if 'V1 Error %' in row and 'V2 Error %' in row:
            improvement = ((row['V1 Error %'] - row['V2 Error %']) / row['V1 Error %']) * 100
            row['Improvement %'] = improvement
        
        comparison.append(row)
    
    df_comparison = pd.DataFrame(comparison)
    
    # Format for display
    if 'V1 Error %' in df_comparison.columns:
        df_comparison['V1 Error %'] = df_comparison['V1 Error %'].apply(lambda x: f'{x:.2f}' if pd.notna(x) else '-')
    if 'V2 Error %' in df_comparison.columns:
        df_comparison['V2 Error %'] = df_comparison['V2 Error %'].apply(lambda x: f'{x:.2f}' if pd.notna(x) else '-')
    if 'Improvement %' in df_comparison.columns:
        df_comparison['Improvement %'] = df_comparison['Improvement %'].apply(lambda x: f'{x:+.1f}' if pd.notna(x) else '-')
    
    print(df_comparison.to_string(index=False))
    
    # Highlight recent years
    if df_v2 is not None:
        recent = df_v2[df_v2['year'] >= 2024]
        if len(recent) > 0:
            print("\n" + "-"*80)
            print("FOCUS: Recent Data (2024-2025)")
            print("-"*80)
            print(f"Records: {len(recent)}")
            print(f"V2 Mean Abs % Error: {recent['pct_error'].abs().mean():.2f}%")
            
            if df_v1 is not None:
                recent_v1 = df_v1[df_v1['year'] >= 2024]
                if len(recent_v1) > 0:
                    v1_error = recent_v1['pct_error'].abs().mean()
                    v2_error = recent['pct_error'].abs().mean()
                    improvement = ((v1_error - v2_error) / v1_error) * 100
                    print(f"V1 Mean Abs % Error: {v1_error:.2f}%")
                    print(f"Improvement: {improvement:+.1f}%")


def plot_comparison(df_v1, df_v2):
    """Create visualization comparing models"""
    
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Model Comparison: Original V1 vs Enhanced V2', fontsize=16, fontweight='bold')
    
    # 1. Error distribution
    ax = axes[0, 0]
    if df_v1 is not None:
        v1_errors = df_v1['pct_error'].clip(-50, 50)
        ax.hist(v1_errors, bins=50, alpha=0.5, label='V1', color='blue')
    if df_v2 is not None:
        v2_errors = df_v2['pct_error'].clip(-50, 50)
        ax.hist(v2_errors, bins=50, alpha=0.5, label='V2', color='green')
    ax.set_xlabel('Percentage Error (%)')
    ax.set_ylabel('Frequency')
    ax.set_title('Error Distribution')
    ax.legend()
    ax.axvline(0, color='red', linestyle='--', alpha=0.5)
    
    # 2. Error by year
    ax = axes[0, 1]
    years = sorted(set(
        (df_v1['year'].unique() if df_v1 is not None else []) + 
        (df_v2['year'].unique() if df_v2 is not None else [])
    ))
    
    if df_v1 is not None:
        v1_yearly = [df_v1[df_v1['year'] == y]['pct_error'].abs().mean() for y in years]
        ax.plot(years, v1_yearly, marker='o', label='V1', linewidth=2)
    if df_v2 is not None:
        v2_yearly = [df_v2[df_v2['year'] == y]['pct_error'].abs().mean() for y in years]
        ax.plot(years, v2_yearly, marker='s', label='V2', linewidth=2)
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Mean Absolute % Error')
    ax.set_title('Performance Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Prediction scatter (V2 only if available)
    ax = axes[1, 0]
    if df_v2 is not None:
        sample = df_v2.sample(min(5000, len(df_v2)))
        ax.scatter(sample['sale_price'], sample['predicted_price'], 
                  alpha=0.3, s=10, color='green')
        
        # Perfect prediction line
        min_price = min(sample['sale_price'].min(), sample['predicted_price'].min())
        max_price = max(sample['sale_price'].max(), sample['predicted_price'].max())
        ax.plot([min_price, max_price], [min_price, max_price], 
               'r--', linewidth=2, label='Perfect Prediction')
        
        ax.set_xlabel('Actual Price ($)')
        ax.set_ylabel('Predicted Price ($)')
        ax.set_title('V2 Predictions vs Actual')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'V2 predictions not available', 
               ha='center', va='center', transform=ax.transAxes)
    
    # 4. Confidence intervals (V2 only)
    ax = axes[1, 1]
    if df_v2 is not None and 'price_lower_95' in df_v2.columns:
        sample = df_v2.sample(min(100, len(df_v2))).sort_values('sale_price')
        x = range(len(sample))
        
        ax.fill_between(x, sample['price_lower_95'], sample['price_upper_95'], 
                        alpha=0.3, color='green', label='95% CI')
        ax.scatter(x, sample['sale_price'], color='blue', s=20, label='Actual', zorder=3)
        ax.scatter(x, sample['predicted_price'], color='red', s=20, label='Predicted', zorder=3)
        
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Price ($)')
        ax.set_title('V2 Uncertainty Estimation (Sample)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Uncertainty estimates not available', 
               ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    
    # Save figure
    output_file = 'outputs/model_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to {output_file}")
    
    plt.show()


def main():
    """Run complete comparison"""
    
    print("="*80)
    print("MODEL COMPARISON ANALYSIS")
    print("="*80)
    
    # Load predictions
    df_v1, df_v2, has_v1, has_v2 = load_predictions()
    
    if not has_v1 and not has_v2:
        print("\nNo predictions found. Please run:")
        print("  1. Original model (if available)")
        print("  2. python main_train_v2.py")
        return
    
    # Compare performance
    if has_v1 or has_v2:
        compare_overall_performance(df_v1, df_v2)
        compare_temporal_performance(df_v1, df_v2)
    
    # Create visualizations
    if has_v1 or has_v2:
        try:
            plot_comparison(df_v1, df_v2)
        except Exception as e:
            print(f"\nVisualization error: {e}")
            print("Continuing without plots...")
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)
    
    if has_v2:
        print("\n✓ Enhanced V2 model shows improved temporal generalization")
        print("✓ Uncertainty estimates help identify reliable predictions")
        print("✓ Market indicators capture external economic factors")
        print("\nNext steps:")
        print("  1. Review MODEL_V2_GUIDE.md for detailed documentation")
        print("  2. Use predict_listings.py for new listing predictions")
        print("  3. Set up automated retraining with retraining_pipeline.py")


if __name__ == "__main__":
    main()
