"""
Compare V2 vs V3 Model Performance
V2: Without community mapping (985 communities from clustering)
V3: With community mapping from H3 level 7
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_predictions():
    """Load V2 and V3 predictions"""
    try:
        df_v2 = pd.read_csv('data/sales_2020_25_with_predictions_v2.csv')
        print(f"✓ Loaded V2: {len(df_v2)} records")
        has_v2 = True
    except FileNotFoundError:
        print("✗ V2 predictions not found")
        df_v2 = None
        has_v2 = False
    
    try:
        df_v3 = pd.read_csv('data/sales_2020_25_with_predictions_v3.csv')
        print(f"✓ Loaded V3: {len(df_v3)} records")
        has_v3 = True
    except FileNotFoundError:
        print("✗ V3 predictions not found")
        df_v3 = None
        has_v3 = False
    
    return df_v2, df_v3, has_v2, has_v3


def compare_overall_performance(df_v2, df_v3):
    """Compare overall performance metrics"""
    print("\n" + "="*80)
    print("OVERALL PERFORMANCE COMPARISON")
    print("="*80)
    
    metrics = []
    
    if df_v2 is not None:
        v2_metrics = {
            'Model': 'V2 (No H3 Mapping)',
            'Mean Abs Error %': df_v2['pct_error'].abs().mean(),
            'Median Abs Error %': df_v2['pct_error'].abs().median(),
            'Std Error %': df_v2['pct_error'].abs().std(),
            'Overest >50%': (df_v2['pct_error'] > 50).sum(),
            'Overest >50% %': (df_v2['pct_error'] > 50).mean() * 100,
            'Underest <-50%': (df_v2['pct_error'] < -50).sum(),
            'Underest <-50% %': (df_v2['pct_error'] < -50).mean() * 100,
        }
        metrics.append(v2_metrics)
    
    if df_v3 is not None:
        v3_metrics = {
            'Model': 'V3 (With H3 Mapping)',
            'Mean Abs Error %': df_v3['pct_error'].abs().mean(),
            'Median Abs Error %': df_v3['pct_error'].abs().median(),
            'Std Error %': df_v3['pct_error'].abs().std(),
            'Overest >50%': (df_v3['pct_error'] > 50).sum(),
            'Overest >50% %': (df_v3['pct_error'] > 50).mean() * 100,
            'Underest <-50%': (df_v3['pct_error'] < -50).sum(),
            'Underest <-50% %': (df_v3['pct_error'] < -50).mean() * 100,
        }
        metrics.append(v3_metrics)
    
    df_metrics = pd.DataFrame(metrics)
    print("\n", df_metrics.to_string(index=False))
    
    if len(metrics) == 2:
        improvement = metrics[0]['Mean Abs Error %'] - metrics[1]['Mean Abs Error %']
        print(f"\nImprovement: {improvement:+.2f} percentage points")
        if improvement > 0:
            print(f"✓ V3 is better by {improvement:.2f}%")
        else:
            print(f"✗ V2 is better by {-improvement:.2f}%")


def compare_attention_weights(df_v2, df_v3):
    """Compare CLS attention weights"""
    print("\n" + "="*80)
    print("CLS ATTENTION WEIGHTS COMPARISON")
    print("="*80)
    
    attention_cols = ['cls_attn_community', 'cls_attn_year', 'cls_attn_week', 
                     'cls_attn_property', 'cls_attn_time', 'cls_attn_market']
    
    if df_v2 is not None and all(col in df_v2.columns for col in attention_cols):
        print("\nV2 (No H3 Mapping):")
        for col in attention_cols:
            feature = col.replace('cls_attn_', '').capitalize()
            print(f"  {feature:12s}: {df_v2[col].mean():.4f}")
    
    if df_v3 is not None and all(col in df_v3.columns for col in attention_cols):
        print("\nV3 (With H3 Mapping):")
        for col in attention_cols:
            feature = col.replace('cls_attn_', '').capitalize()
            print(f"  {feature:12s}: {df_v3[col].mean():.4f}")
    
    if df_v2 is not None and df_v3 is not None and all(col in df_v2.columns for col in attention_cols):
        print("\nDifference (V3 - V2):")
        for col in attention_cols:
            feature = col.replace('cls_attn_', '').capitalize()
            diff = df_v3[col].mean() - df_v2[col].mean()
            print(f"  {feature:12s}: {diff:+.4f}")


def compare_by_community(df_v2, df_v3):
    """Compare performance by community"""
    print("\n" + "="*80)
    print("PERFORMANCE BY COMMUNITY")
    print("="*80)
    
    # Get top communities (by V3 volume)
    if df_v3 is not None:
        top_communities = df_v3['community'].value_counts().head(10).index
        
        print("\nTop 10 Communities (by volume):")
        for comm in top_communities:
            v2_comm = df_v2[df_v2['community'] == comm] if df_v2 is not None else None
            v3_comm = df_v3[df_v3['community'] == comm]
            
            print(f"\nCommunity {comm}:")
            print(f"  Records: {len(v3_comm)}")
            
            if v2_comm is not None and len(v2_comm) > 0:
                print(f"  V2 Error: {v2_comm['pct_error'].abs().mean():.1f}%")
            print(f"  V3 Error: {v3_comm['pct_error'].abs().mean():.1f}%")
            
            if v2_comm is not None and len(v2_comm) > 0:
                improvement = v2_comm['pct_error'].abs().mean() - v3_comm['pct_error'].abs().mean()
                print(f"  Improvement: {improvement:+.1f}%")


def create_visualizations(df_v2, df_v3):
    """Create comparison visualizations"""
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Error distribution comparison
    ax1 = fig.add_subplot(gs[0, 0])
    if df_v2 is not None:
        ax1.hist(df_v2['pct_error'].clip(-100, 100), bins=60, alpha=0.5, 
                label='V2 (No H3)', color='steelblue', density=True)
    if df_v3 is not None:
        ax1.hist(df_v3['pct_error'].clip(-100, 100), bins=60, alpha=0.5, 
                label='V3 (With H3)', color='coral', density=True)
    ax1.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_xlabel('Percentage Error (%)')
    ax1.set_ylabel('Density')
    ax1.set_title('Error Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. CLS Attention comparison
    ax2 = fig.add_subplot(gs[0, 1])
    attention_cols = ['cls_attn_community', 'cls_attn_year', 'cls_attn_week', 
                     'cls_attn_property', 'cls_attn_time', 'cls_attn_market']
    features = ['Community', 'Year', 'Week', 'Property', 'Time', 'Market']
    
    if df_v2 is not None and all(col in df_v2.columns for col in attention_cols):
        v2_attn = [df_v2[col].mean() for col in attention_cols]
        v3_attn = [df_v3[col].mean() for col in attention_cols] if df_v3 is not None else [0]*6
        
        x = np.arange(len(features))
        width = 0.35
        ax2.bar(x - width/2, v2_attn, width, label='V2', alpha=0.8, color='steelblue')
        ax2.bar(x + width/2, v3_attn, width, label='V3', alpha=0.8, color='coral')
        ax2.axhline(1/6, color='gray', linestyle='--', alpha=0.5, label='Equal (16.7%)')
        ax2.set_ylabel('Attention Weight')
        ax2.set_title('CLS Attention Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(features, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. MAE by community (top 15)
    ax3 = fig.add_subplot(gs[0, 2])
    if df_v3 is not None:
        top_comms = df_v3['community'].value_counts().head(15).index
        v2_errors = []
        v3_errors = []
        labels = []
        
        for comm in top_comms:
            v3_comm = df_v3[df_v3['community'] == comm]
            v3_errors.append(v3_comm['pct_error'].abs().mean())
            
            if df_v2 is not None:
                v2_comm = df_v2[df_v2['community'] == comm]
                v2_errors.append(v2_comm['pct_error'].abs().mean() if len(v2_comm) > 0 else 0)
            
            labels.append(f'C{comm}')
        
        x = np.arange(len(labels))
        width = 0.35
        if df_v2 is not None:
            ax3.barh(x - width/2, v2_errors, width, label='V2', alpha=0.8, color='steelblue')
        ax3.barh(x + width/2, v3_errors, width, label='V3', alpha=0.8, color='coral')
        ax3.set_yticks(x)
        ax3.set_yticklabels(labels, fontsize=8)
        ax3.set_xlabel('Mean Absolute % Error')
        ax3.set_title('Error by Community (Top 15)')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='x')
    
    # 4. Cumulative error distribution
    ax4 = fig.add_subplot(gs[1, 0])
    if df_v2 is not None:
        v2_sorted = np.sort(df_v2['pct_error'].abs())
        ax4.plot(v2_sorted, np.linspace(0, 100, len(v2_sorted)), 
                label='V2', linewidth=2, color='steelblue')
    if df_v3 is not None:
        v3_sorted = np.sort(df_v3['pct_error'].abs())
        ax4.plot(v3_sorted, np.linspace(0, 100, len(v3_sorted)), 
                label='V3', linewidth=2, color='coral')
    ax4.set_xlabel('Absolute % Error')
    ax4.set_ylabel('Cumulative %')
    ax4.set_title('Cumulative Error Distribution')
    ax4.set_xlim(0, 100)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Error by price range
    ax5 = fig.add_subplot(gs[1, 1])
    price_bins = ['<$400k', '$400k-$700k', '$700k-$1M', '>$1M']
    bin_edges = [0, 400000, 700000, 1000000, float('inf')]
    
    v2_errors_by_price = []
    v3_errors_by_price = []
    
    for i in range(len(bin_edges)-1):
        if df_v2 is not None:
            subset = df_v2[(df_v2['sale_price'] >= bin_edges[i]) & 
                          (df_v2['sale_price'] < bin_edges[i+1])]
            v2_errors_by_price.append(subset['pct_error'].abs().mean())
        
        if df_v3 is not None:
            subset = df_v3[(df_v3['sale_price'] >= bin_edges[i]) & 
                          (df_v3['sale_price'] < bin_edges[i+1])]
            v3_errors_by_price.append(subset['pct_error'].abs().mean())
    
    x = np.arange(len(price_bins))
    width = 0.35
    if df_v2 is not None:
        ax5.bar(x - width/2, v2_errors_by_price, width, label='V2', alpha=0.8, color='steelblue')
    if df_v3 is not None:
        ax5.bar(x + width/2, v3_errors_by_price, width, label='V3', alpha=0.8, color='coral')
    ax5.set_xticks(x)
    ax5.set_xticklabels(price_bins, rotation=45)
    ax5.set_ylabel('Mean Absolute % Error')
    ax5.set_title('Error by Price Range')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Error by year
    ax6 = fig.add_subplot(gs[1, 2])
    if df_v2 is not None:
        v2_by_year = df_v2.groupby('year')['pct_error'].apply(lambda x: x.abs().mean())
        ax6.plot(v2_by_year.index, v2_by_year.values, marker='o', 
                label='V2', linewidth=2, color='steelblue')
    if df_v3 is not None:
        v3_by_year = df_v3.groupby('year')['pct_error'].apply(lambda x: x.abs().mean())
        ax6.plot(v3_by_year.index, v3_by_year.values, marker='s', 
                label='V3', linewidth=2, color='coral')
    ax6.set_xlabel('Year')
    ax6.set_ylabel('Mean Absolute % Error')
    ax6.set_title('Error Over Time')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Scatter: V2 vs V3 error by community
    ax7 = fig.add_subplot(gs[2, 0])
    if df_v2 is not None and df_v3 is not None:
        communities = df_v3['community'].value_counts().head(30).index
        v2_comm_errors = []
        v3_comm_errors = []
        sizes = []
        
        for comm in communities:
            v2_comm = df_v2[df_v2['community'] == comm]
            v3_comm = df_v3[df_v3['community'] == comm]
            
            if len(v2_comm) > 0 and len(v3_comm) > 0:
                v2_comm_errors.append(v2_comm['pct_error'].abs().mean())
                v3_comm_errors.append(v3_comm['pct_error'].abs().mean())
                sizes.append(len(v3_comm) / 10)
        
        ax7.scatter(v2_comm_errors, v3_comm_errors, s=sizes, alpha=0.6, color='purple')
        max_val = max(max(v2_comm_errors), max(v3_comm_errors))
        ax7.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Equal')
        ax7.set_xlabel('V2 Error (%)')
        ax7.set_ylabel('V3 Error (%)')
        ax7.set_title('Community Error: V2 vs V3\n(size = volume)')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
    
    # 8. Community attention vs error (V3)
    ax8 = fig.add_subplot(gs[2, 1])
    if df_v3 is not None and 'cls_attn_community' in df_v3.columns:
        ax8.scatter(df_v3['cls_attn_community'], df_v3['pct_error'].abs(), 
                   alpha=0.05, s=1, color='coral')
        ax8.set_xlabel('Community Attention (V3)')
        ax8.set_ylabel('Absolute % Error')
        ax8.set_title('V3: Community Attention vs Error')
        ax8.grid(True, alpha=0.3)
    
    # 9. Summary statistics
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    summary_text = "SUMMARY\n" + "="*35 + "\n\n"
    
    if df_v2 is not None:
        summary_text += f"V2 (No H3 Mapping):\n"
        summary_text += f"  MAE: {df_v2['pct_error'].abs().mean():.2f}%\n"
        summary_text += f"  Median: {df_v2['pct_error'].abs().median():.2f}%\n"
        summary_text += f"  Communities: {df_v2['community'].nunique()}\n"
        if 'cls_attn_community' in df_v2.columns:
            summary_text += f"  Comm Attn: {df_v2['cls_attn_community'].mean():.3f}\n"
        summary_text += "\n"
    
    if df_v3 is not None:
        summary_text += f"V3 (With H3 Mapping):\n"
        summary_text += f"  MAE: {df_v3['pct_error'].abs().mean():.2f}%\n"
        summary_text += f"  Median: {df_v3['pct_error'].abs().median():.2f}%\n"
        summary_text += f"  Communities: {df_v3['community'].nunique()}\n"
        if 'cls_attn_community' in df_v3.columns:
            summary_text += f"  Comm Attn: {df_v3['cls_attn_community'].mean():.3f}\n"
        summary_text += "\n"
    
    if df_v2 is not None and df_v3 is not None:
        improvement = df_v2['pct_error'].abs().mean() - df_v3['pct_error'].abs().mean()
        summary_text += f"Improvement:\n"
        summary_text += f"  {improvement:+.2f} percentage points\n"
        if improvement > 0:
            summary_text += f"  ✓ V3 is better\n"
        else:
            summary_text += f"  ✗ V2 is better\n"
    
    ax9.text(0.1, 0.95, summary_text, transform=ax9.transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle('V2 vs V3 Model Comparison', fontsize=16, fontweight='bold')
    
    plt.savefig('outputs/v2_v3_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved visualization to outputs/v2_v3_comparison.png")


def main():
    """Run V2 vs V3 comparison"""
    print("="*80)
    print("V2 vs V3 MODEL COMPARISON")
    print("="*80)
    
    df_v2, df_v3, has_v2, has_v3 = load_predictions()
    
    if not has_v2 and not has_v3:
        print("\n❌ No predictions found. Please run:")
        print("  python main_train_v2.py")
        print("  python main_train_v3.py")
        return
    
    if has_v2 and has_v3:
        compare_overall_performance(df_v2, df_v3)
        compare_attention_weights(df_v2, df_v3)
        compare_by_community(df_v2, df_v3)
        create_visualizations(df_v2, df_v3)
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
