"""
Analyze Differences Between V1 and V2 Models
Helps understand why V2 overestimates vs V1 underestimates
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_predictions():
    """Load predictions from both models"""
    try:
        df_v1 = pd.read_csv('data/sales_2020_25_with_predictions.csv')
        df_v1['sale_date'] = pd.to_datetime(df_v1['sale_date'])
        has_v1 = True
        print(f"✓ Loaded V1: {len(df_v1)} records")
    except FileNotFoundError:
        print("✗ V1 predictions not found")
        df_v1 = None
        has_v1 = False
    
    try:
        df_v2 = pd.read_csv('data/sales_2020_25_with_predictions_v2.csv')
        df_v2['sale_date'] = pd.to_datetime(df_v2['sale_date'])
        has_v2 = True
        print(f"✓ Loaded V2: {len(df_v2)} records")
    except FileNotFoundError:
        print("✗ V2 predictions not found")
        df_v2 = None
        has_v2 = False
    
    return df_v1, df_v2, has_v1, has_v2


def analyze_overestimation(df_v1, df_v2):
    """Analyze overestimation patterns"""
    print("\n" + "="*80)
    print("OVERESTIMATION ANALYSIS")
    print("="*80)
    
    # V1 overestimation
    if df_v1 is not None:
        v1_overest = df_v1[df_v1['pct_error'] > 50]
        print(f"\nV1 Overestimation (>50%):")
        print(f"  Count: {len(v1_overest)} ({len(v1_overest)/len(df_v1)*100:.1f}%)")
        print(f"  Mean error: {v1_overest['pct_error'].mean():.1f}%")
        print(f"  Median error: {v1_overest['pct_error'].median():.1f}%")
    
    # V2 overestimation
    if df_v2 is not None:
        v2_overest = df_v2[df_v2['pct_error'] > 50]
        print(f"\nV2 Overestimation (>50%):")
        print(f"  Count: {len(v2_overest)} ({len(v2_overest)/len(df_v2)*100:.1f}%)")
        print(f"  Mean error: {v2_overest['pct_error'].mean():.1f}%")
        print(f"  Median error: {v2_overest['pct_error'].median():.1f}%")
        
        # Analyze CLS attention for overestimated predictions
        if 'cls_attn_community' in v2_overest.columns:
            print(f"\n  CLS Attention (Overestimated):")
            print(f"    Community: {v2_overest['cls_attn_community'].mean():.3f}")
            print(f"    Year:      {v2_overest['cls_attn_year'].mean():.3f}")
            print(f"    Week:      {v2_overest['cls_attn_week'].mean():.3f}")
            print(f"    Property:  {v2_overest['cls_attn_property'].mean():.3f}")
            print(f"    Time:      {v2_overest['cls_attn_time'].mean():.3f}")
            print(f"    Market:    {v2_overest['cls_attn_market'].mean():.3f}")
            
            # Compare to well-predicted samples
            v2_good = df_v2[df_v2['pct_error'].abs() < 10]
            print(f"\n  CLS Attention (Well-predicted, <10% error):")
            print(f"    Community: {v2_good['cls_attn_community'].mean():.3f}")
            print(f"    Year:      {v2_good['cls_attn_year'].mean():.3f}")
            print(f"    Week:      {v2_good['cls_attn_week'].mean():.3f}")
            print(f"    Property:  {v2_good['cls_attn_property'].mean():.3f}")
            print(f"    Time:      {v2_good['cls_attn_time'].mean():.3f}")
            print(f"    Market:    {v2_good['cls_attn_market'].mean():.3f}")
        
        # Characteristics of overestimated properties
        print(f"\n  Characteristics of Overestimated Properties:")
        print(f"    Avg actual price: ${v2_overest['sale_price'].mean():,.0f}")
        print(f"    Avg predicted price: ${v2_overest['predicted_price'].mean():,.0f}")
        print(f"    Avg sqft: {v2_overest['sqft'].mean():.0f}")
        print(f"    Avg beds: {v2_overest['beds'].mean():.1f}")
        
        # By year
        print(f"\n  Overestimation by Year:")
        by_year = v2_overest.groupby(v2_overest['sale_date'].dt.year).size()
        for year, count in by_year.items():
            pct = count / len(df_v2[df_v2['sale_date'].dt.year == year]) * 100
            print(f"    {year}: {count} ({pct:.1f}% of year)")
        
        # By community
        print(f"\n  Top 10 Communities with Overestimation:")
        by_comm = v2_overest.groupby('community').size().sort_values(ascending=False).head(10)
        for comm, count in by_comm.items():
            total_in_comm = len(df_v2[df_v2['community'] == comm])
            pct = count / total_in_comm * 100 if total_in_comm > 0 else 0
            print(f"    Community {comm}: {count} ({pct:.1f}% of community)")


def analyze_underestimation(df_v1, df_v2):
    """Analyze underestimation patterns"""
    print("\n" + "="*80)
    print("UNDERESTIMATION ANALYSIS")
    print("="*80)
    
    # V1 underestimation
    if df_v1 is not None:
        v1_underest = df_v1[df_v1['pct_error'] < -50]
        print(f"\nV1 Underestimation (<-50%):")
        print(f"  Count: {len(v1_underest)} ({len(v1_underest)/len(df_v1)*100:.1f}%)")
        print(f"  Mean error: {v1_underest['pct_error'].mean():.1f}%")
        print(f"  Median error: {v1_underest['pct_error'].median():.1f}%")
    
    # V2 underestimation
    if df_v2 is not None:
        v2_underest = df_v2[df_v2['pct_error'] < -50]
        print(f"\nV2 Underestimation (<-50%):")
        print(f"  Count: {len(v2_underest)} ({len(v2_underest)/len(df_v2)*100:.1f}%)")
        print(f"  Mean error: {v2_underest['pct_error'].mean():.1f}%")
        print(f"  Median error: {v2_underest['pct_error'].median():.1f}%")


def analyze_feature_importance(df_v2):
    """Analyze which features drive predictions"""
    if df_v2 is None or 'cls_attn_community' not in df_v2.columns:
        print("\n⚠️  CLS attention not available in V2 predictions")
        return
    
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE ANALYSIS (V2)")
    print("="*80)
    
    # Overall attention
    print(f"\nOverall CLS Attention (all predictions):")
    print(f"  Community: {df_v2['cls_attn_community'].mean():.3f}")
    print(f"  Year:      {df_v2['cls_attn_year'].mean():.3f}")
    print(f"  Week:      {df_v2['cls_attn_week'].mean():.3f}")
    print(f"  Property:  {df_v2['cls_attn_property'].mean():.3f}")
    print(f"  Time:      {df_v2['cls_attn_time'].mean():.3f}")
    print(f"  Market:    {df_v2['cls_attn_market'].mean():.3f}")
    
    # By error magnitude
    print(f"\nCLS Attention by Error Magnitude:")
    
    error_bins = [
        ("Excellent (<5%)", df_v2[df_v2['pct_error'].abs() < 5]),
        ("Good (5-10%)", df_v2[(df_v2['pct_error'].abs() >= 5) & (df_v2['pct_error'].abs() < 10)]),
        ("Fair (10-20%)", df_v2[(df_v2['pct_error'].abs() >= 10) & (df_v2['pct_error'].abs() < 20)]),
        ("Poor (>20%)", df_v2[df_v2['pct_error'].abs() >= 20])
    ]
    
    for label, subset in error_bins:
        if len(subset) > 0:
            print(f"\n  {label} ({len(subset)} records):")
            print(f"    Community: {subset['cls_attn_community'].mean():.3f}")
            print(f"    Year:      {subset['cls_attn_year'].mean():.3f}")
            print(f"    Week:      {subset['cls_attn_week'].mean():.3f}")
            print(f"    Property:  {subset['cls_attn_property'].mean():.3f}")
            print(f"    Time:      {subset['cls_attn_time'].mean():.3f}")
            print(f"    Market:    {subset['cls_attn_market'].mean():.3f}")


def compare_price_ranges(df_v1, df_v2):
    """Compare performance across price ranges"""
    print("\n" + "="*80)
    print("PERFORMANCE BY PRICE RANGE")
    print("="*80)
    
    price_bins = [
        ("Low (<$400k)", 0, 400000),
        ("Mid ($400k-$700k)", 400000, 700000),
        ("High ($700k-$1M)", 700000, 1000000),
        ("Luxury (>$1M)", 1000000, float('inf'))
    ]
    
    for label, min_price, max_price in price_bins:
        print(f"\n{label}:")
        
        if df_v1 is not None:
            v1_subset = df_v1[(df_v1['sale_price'] >= min_price) & (df_v1['sale_price'] < max_price)]
            if len(v1_subset) > 0:
                print(f"  V1: {len(v1_subset)} records, {v1_subset['pct_error'].abs().mean():.1f}% error")
                print(f"      Overest >50%: {len(v1_subset[v1_subset['pct_error'] > 50])} ({len(v1_subset[v1_subset['pct_error'] > 50])/len(v1_subset)*100:.1f}%)")
        
        if df_v2 is not None:
            v2_subset = df_v2[(df_v2['sale_price'] >= min_price) & (df_v2['sale_price'] < max_price)]
            if len(v2_subset) > 0:
                print(f"  V2: {len(v2_subset)} records, {v2_subset['pct_error'].abs().mean():.1f}% error")
                print(f"      Overest >50%: {len(v2_subset[v2_subset['pct_error'] > 50])} ({len(v2_subset[v2_subset['pct_error'] > 50])/len(v2_subset)*100:.1f}%)")


def create_visualizations(df_v1, df_v2):
    """Create comparison visualizations"""
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('V1 vs V2 Model Comparison: Overestimation Analysis', fontsize=16, fontweight='bold')
    
    # 1. Error distribution comparison
    ax = axes[0, 0]
    if df_v1 is not None:
        ax.hist(df_v1['pct_error'].clip(-100, 100), bins=50, alpha=0.5, label='V1', color='blue')
    if df_v2 is not None:
        ax.hist(df_v2['pct_error'].clip(-100, 100), bins=50, alpha=0.5, label='V2', color='green')
    ax.axvline(0, color='red', linestyle='--', alpha=0.5)
    ax.axvline(50, color='orange', linestyle='--', alpha=0.5, label='50% overest')
    ax.axvline(-50, color='orange', linestyle='--', alpha=0.5, label='50% underest')
    ax.set_xlabel('Percentage Error (%)')
    ax.set_ylabel('Frequency')
    ax.set_title('Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. CLS Attention for V2 (if available)
    ax = axes[0, 1]
    if df_v2 is not None and 'cls_attn_community' in df_v2.columns:
        # Compare attention for overestimated vs well-predicted
        overest = df_v2[df_v2['pct_error'] > 50]
        good = df_v2[df_v2['pct_error'].abs() < 10]
        
        features = ['Community', 'Year', 'Week', 'Property', 'Time', 'Market']
        overest_attn = [
            overest['cls_attn_community'].mean(),
            overest['cls_attn_year'].mean(),
            overest['cls_attn_week'].mean(),
            overest['cls_attn_property'].mean(),
            overest['cls_attn_time'].mean(),
            overest['cls_attn_market'].mean()
        ]
        good_attn = [
            good['cls_attn_community'].mean(),
            good['cls_attn_year'].mean(),
            good['cls_attn_week'].mean(),
            good['cls_attn_property'].mean(),
            good['cls_attn_time'].mean(),
            good['cls_attn_market'].mean()
        ]
        
        x = np.arange(len(features))
        width = 0.35
        ax.bar(x - width/2, overest_attn, width, label='Overestimated (>50%)', color='red', alpha=0.7)
        ax.bar(x + width/2, good_attn, width, label='Well-predicted (<10%)', color='green', alpha=0.7)
        ax.set_xlabel('Feature')
        ax.set_ylabel('CLS Attention Weight')
        ax.set_title('V2: CLS Attention by Prediction Quality')
        ax.set_xticks(x)
        ax.set_xticklabels(features, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, 'CLS attention not available', ha='center', va='center', transform=ax.transAxes)
    
    # 3. Overestimation by price range
    ax = axes[1, 0]
    price_bins = ['<$400k', '$400k-$700k', '$700k-$1M', '>$1M']
    bin_edges = [0, 400000, 700000, 1000000, float('inf')]
    
    v1_overest_pct = []
    v2_overest_pct = []
    
    for i in range(len(bin_edges)-1):
        if df_v1 is not None:
            subset = df_v1[(df_v1['sale_price'] >= bin_edges[i]) & (df_v1['sale_price'] < bin_edges[i+1])]
            pct = len(subset[subset['pct_error'] > 50]) / len(subset) * 100 if len(subset) > 0 else 0
            v1_overest_pct.append(pct)
        
        if df_v2 is not None:
            subset = df_v2[(df_v2['sale_price'] >= bin_edges[i]) & (df_v2['sale_price'] < bin_edges[i+1])]
            pct = len(subset[subset['pct_error'] > 50]) / len(subset) * 100 if len(subset) > 0 else 0
            v2_overest_pct.append(pct)
    
    x = np.arange(len(price_bins))
    width = 0.35
    if df_v1 is not None:
        ax.bar(x - width/2, v1_overest_pct, width, label='V1', color='blue', alpha=0.7)
    if df_v2 is not None:
        ax.bar(x + width/2, v2_overest_pct, width, label='V2', color='green', alpha=0.7)
    ax.set_xlabel('Price Range')
    ax.set_ylabel('% Overestimated (>50%)')
    ax.set_title('Overestimation Rate by Price Range')
    ax.set_xticks(x)
    ax.set_xticklabels(price_bins, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Overestimation by year
    ax = axes[1, 1]
    if df_v1 is not None:
        v1_by_year = df_v1.groupby(df_v1['sale_date'].dt.year).apply(
            lambda x: len(x[x['pct_error'] > 50]) / len(x) * 100
        )
        ax.plot(v1_by_year.index, v1_by_year.values, marker='o', label='V1', linewidth=2)
    
    if df_v2 is not None:
        v2_by_year = df_v2.groupby(df_v2['sale_date'].dt.year).apply(
            lambda x: len(x[x['pct_error'] > 50]) / len(x) * 100
        )
        ax.plot(v2_by_year.index, v2_by_year.values, marker='s', label='V2', linewidth=2)
    
    ax.set_xlabel('Year')
    ax.set_ylabel('% Overestimated (>50%)')
    ax.set_title('Overestimation Rate Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/model_overestimation_analysis.png', dpi=150, bbox_inches='tight')
    print("✓ Saved visualization to outputs/model_overestimation_analysis.png")


def main():
    """Run complete analysis"""
    print("="*80)
    print("MODEL COMPARISON: V1 vs V2 OVERESTIMATION ANALYSIS")
    print("="*80)
    
    df_v1, df_v2, has_v1, has_v2 = load_predictions()
    
    if not has_v1 and not has_v2:
        print("\n❌ No predictions found. Please run:")
        print("  python main_train_v2.py")
        return
    
    if has_v1 and has_v2:
        analyze_overestimation(df_v1, df_v2)
        analyze_underestimation(df_v1, df_v2)
        compare_price_ranges(df_v1, df_v2)
    
    if has_v2:
        analyze_feature_importance(df_v2)
    
    if has_v1 or has_v2:
        create_visualizations(df_v1, df_v2)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
