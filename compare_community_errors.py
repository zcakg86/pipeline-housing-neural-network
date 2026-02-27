"""
Compare V1 vs V2 Error by Community
Visualize how each model performs across different communities
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    """Load both model predictions"""
    try:
        df_v1 = pd.read_csv('data/sales_2020_25_with_predictions.csv')
        print(f"✓ Loaded V1: {len(df_v1)} records")
        has_v1 = True
    except FileNotFoundError:
        print("✗ V1 predictions not found")
        df_v1 = None
        has_v1 = False
    
    try:
        df_v2 = pd.read_csv('data/sales_2020_25_with_predictions_v2.csv')
        print(f"✓ Loaded V2: {len(df_v2)} records")
        has_v2 = True
    except FileNotFoundError:
        print("✗ V2 predictions not found")
        df_v2 = None
        has_v2 = False
    
    return df_v1, df_v2, has_v1, has_v2


def analyze_by_community(df_v1, df_v2):
    """Analyze error metrics by community"""
    
    results = []
    
    # Get all unique communities
    communities = set()
    if df_v1 is not None:
        communities.update(df_v1['community'].unique())
    if df_v2 is not None:
        communities.update(df_v2['community'].unique())
    
    communities = sorted(communities)
    
    for comm in communities:
        row = {'community': comm}
        
        # V1 stats
        if df_v1 is not None:
            v1_comm = df_v1[df_v1['community'] == comm]
            if len(v1_comm) > 0:
                row['v1_count'] = len(v1_comm)
                row['v1_mean_error'] = v1_comm['pct_error'].abs().mean()
                row['v1_median_error'] = v1_comm['pct_error'].abs().median()
                row['v1_overest_pct'] = (v1_comm['pct_error'] > 50).sum() / len(v1_comm) * 100
                row['v1_underest_pct'] = (v1_comm['pct_error'] < -50).sum() / len(v1_comm) * 100
                row['v1_avg_price'] = v1_comm['sale_price'].mean()
            else:
                row['v1_count'] = 0
                row['v1_mean_error'] = np.nan
                row['v1_median_error'] = np.nan
                row['v1_overest_pct'] = np.nan
                row['v1_underest_pct'] = np.nan
                row['v1_avg_price'] = np.nan
        
        # V2 stats
        if df_v2 is not None:
            v2_comm = df_v2[df_v2['community'] == comm]
            if len(v2_comm) > 0:
                row['v2_count'] = len(v2_comm)
                row['v2_mean_error'] = v2_comm['pct_error'].abs().mean()
                row['v2_median_error'] = v2_comm['pct_error'].abs().median()
                row['v2_overest_pct'] = (v2_comm['pct_error'] > 50).sum() / len(v2_comm) * 100
                row['v2_underest_pct'] = (v2_comm['pct_error'] < -50).sum() / len(v2_comm) * 100
                row['v2_avg_price'] = v2_comm['sale_price'].mean()
                
                # V2 specific: community attention
                if 'cls_attn_community' in v2_comm.columns:
                    row['v2_community_attn'] = v2_comm['cls_attn_community'].mean()
            else:
                row['v2_count'] = 0
                row['v2_mean_error'] = np.nan
                row['v2_median_error'] = np.nan
                row['v2_overest_pct'] = np.nan
                row['v2_underest_pct'] = np.nan
                row['v2_avg_price'] = np.nan
                row['v2_community_attn'] = np.nan
        
        results.append(row)
    
    return pd.DataFrame(results)


def create_visualizations(community_stats, df_v1, df_v2):
    """Create comprehensive community comparison visualizations"""
    
    # Filter to communities with sufficient data (at least 100 records in either model)
    min_records = 100
    community_stats['max_count'] = community_stats[['v1_count', 'v2_count']].max(axis=1)
    filtered = community_stats[community_stats['max_count'] >= min_records].copy()
    
    print(f"\nAnalyzing {len(filtered)} communities with ≥{min_records} records")
    
    # Sort by total volume
    filtered = filtered.sort_values('max_count', ascending=False)
    
    # Take top 20 for visualization
    top_n = min(20, len(filtered))
    top_communities = filtered.head(top_n)
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Mean Absolute Error by Community (Top 20)
    ax1 = fig.add_subplot(gs[0, :2])
    x = np.arange(len(top_communities))
    width = 0.35
    
    v1_errors = top_communities['v1_mean_error'].values
    v2_errors = top_communities['v2_mean_error'].values
    
    bars1 = ax1.bar(x - width/2, v1_errors, width, label='V1', alpha=0.8, color='steelblue')
    bars2 = ax1.bar(x + width/2, v2_errors, width, label='V2', alpha=0.8, color='coral')
    
    ax1.set_xlabel('Community', fontsize=11)
    ax1.set_ylabel('Mean Absolute % Error', fontsize=11)
    ax1.set_title(f'Mean Absolute Error by Community (Top {top_n} by Volume)', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'C{int(c)}' for c in top_communities['community']], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add difference indicators
    for i, (v1, v2) in enumerate(zip(v1_errors, v2_errors)):
        if not np.isnan(v1) and not np.isnan(v2):
            diff = v2 - v1
            color = 'red' if diff > 5 else 'green' if diff < -5 else 'gray'
            y_pos = max(v1, v2) + 2
            ax1.text(i, y_pos, f'{diff:+.0f}', ha='center', va='bottom', 
                    fontsize=8, color=color, fontweight='bold')
    
    # 2. Overestimation Rate (>50% error)
    ax2 = fig.add_subplot(gs[0, 2])
    v1_overest = top_communities['v1_overest_pct'].values
    v2_overest = top_communities['v2_overest_pct'].values
    
    ax2.scatter(v1_overest, v2_overest, s=100, alpha=0.6, c=range(len(top_communities)), cmap='viridis')
    
    # Add diagonal line
    max_val = max(np.nanmax(v1_overest), np.nanmax(v2_overest))
    ax2.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Equal')
    
    ax2.set_xlabel('V1 Overestimation Rate (%)', fontsize=10)
    ax2.set_ylabel('V2 Overestimation Rate (%)', fontsize=10)
    ax2.set_title('Overestimation Rate\n(>50% error)', fontsize=11, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Annotate worst offenders
    for idx, row in top_communities.iterrows():
        if row['v2_overest_pct'] > 20 or row['v1_overest_pct'] > 20:
            ax2.annotate(f"C{int(row['community'])}", 
                        (row['v1_overest_pct'], row['v2_overest_pct']),
                        fontsize=7, alpha=0.7)
    
    # 3. Error Distribution Histogram
    ax3 = fig.add_subplot(gs[1, 0])
    if df_v1 is not None:
        ax3.hist(df_v1['pct_error'].clip(-100, 100), bins=60, alpha=0.5, 
                label='V1', color='steelblue', density=True)
    if df_v2 is not None:
        ax3.hist(df_v2['pct_error'].clip(-100, 100), bins=60, alpha=0.5, 
                label='V2', color='coral', density=True)
    ax3.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax3.axvline(50, color='red', linestyle='--', linewidth=1, alpha=0.3)
    ax3.axvline(-50, color='red', linestyle='--', linewidth=1, alpha=0.3)
    ax3.set_xlabel('Percentage Error (%)', fontsize=10)
    ax3.set_ylabel('Density', fontsize=10)
    ax3.set_title('Overall Error Distribution', fontsize=11, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. V2 Community Attention vs Error
    ax4 = fig.add_subplot(gs[1, 1])
    if 'v2_community_attn' in filtered.columns:
        valid = filtered.dropna(subset=['v2_community_attn', 'v2_mean_error'])
        scatter = ax4.scatter(valid['v2_community_attn'], valid['v2_mean_error'], 
                            s=valid['v2_count']/10, alpha=0.6, c=valid['v2_mean_error'],
                            cmap='RdYlGn_r', vmin=0, vmax=50)
        
        # Trend line
        z = np.polyfit(valid['v2_community_attn'], valid['v2_mean_error'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(valid['v2_community_attn'].min(), 
                             valid['v2_community_attn'].max(), 100)
        ax4.plot(x_trend, p(x_trend), "r--", linewidth=2, alpha=0.7,
                label=f'Trend: {z[0]:.0f}x + {z[1]:.1f}')
        
        ax4.set_xlabel('V2 Community Attention', fontsize=10)
        ax4.set_ylabel('V2 Mean Absolute % Error', fontsize=10)
        ax4.set_title('V2: Community Attention vs Error\n(size = volume)', 
                     fontsize=11, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax4, label='Error %')
    else:
        ax4.text(0.5, 0.5, 'Community attention\nnot available', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
    
    # 5. Error Improvement (V1 to V2)
    ax5 = fig.add_subplot(gs[1, 2])
    filtered['error_improvement'] = filtered['v1_mean_error'] - filtered['v2_mean_error']
    improvement_sorted = filtered.sort_values('error_improvement', ascending=False).head(15)
    
    colors = ['green' if x > 0 else 'red' for x in improvement_sorted['error_improvement']]
    ax5.barh(range(len(improvement_sorted)), improvement_sorted['error_improvement'], 
            color=colors, alpha=0.7)
    ax5.set_yticks(range(len(improvement_sorted)))
    ax5.set_yticklabels([f'C{int(c)}' for c in improvement_sorted['community']], fontsize=9)
    ax5.axvline(0, color='black', linestyle='-', linewidth=1)
    ax5.set_xlabel('Error Change (V1 - V2)', fontsize=10)
    ax5.set_title('Error Improvement by Community\n(positive = V2 better)', 
                 fontsize=11, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='x')
    
    # 6. Communities with Worst V2 Performance
    ax6 = fig.add_subplot(gs[2, 0])
    worst_v2 = filtered.nlargest(15, 'v2_mean_error')
    ax6.barh(range(len(worst_v2)), worst_v2['v2_mean_error'], 
            color='darkred', alpha=0.7, label='V2')
    ax6.barh(range(len(worst_v2)), worst_v2['v1_mean_error'], 
            color='steelblue', alpha=0.5, label='V1')
    ax6.set_yticks(range(len(worst_v2)))
    ax6.set_yticklabels([f'C{int(c)}' for c in worst_v2['community']], fontsize=9)
    ax6.set_xlabel('Mean Absolute % Error', fontsize=10)
    ax6.set_title('Communities with Worst V2 Error', fontsize=11, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='x')
    
    # 7. Price Range vs Error
    ax7 = fig.add_subplot(gs[2, 1])
    valid = filtered.dropna(subset=['v2_avg_price', 'v2_mean_error'])
    scatter = ax7.scatter(valid['v2_avg_price']/1000, valid['v2_mean_error'], 
                         s=valid['v2_count']/10, alpha=0.6, c=valid['v2_mean_error'],
                         cmap='RdYlGn_r', vmin=0, vmax=50)
    ax7.set_xlabel('Average Price ($1000s)', fontsize=10)
    ax7.set_ylabel('V2 Mean Absolute % Error', fontsize=10)
    ax7.set_title('V2: Price vs Error by Community\n(size = volume)', 
                 fontsize=11, fontweight='bold')
    ax7.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax7, label='Error %')
    
    # 8. Summary Statistics Table
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    
    summary_text = "SUMMARY STATISTICS\n" + "="*30 + "\n\n"
    
    if df_v1 is not None:
        summary_text += f"V1 Model:\n"
        summary_text += f"  Mean Error: {df_v1['pct_error'].abs().mean():.1f}%\n"
        summary_text += f"  Median Error: {df_v1['pct_error'].abs().median():.1f}%\n"
        summary_text += f"  Overest >50%: {(df_v1['pct_error'] > 50).sum():,} "
        summary_text += f"({(df_v1['pct_error'] > 50).mean()*100:.1f}%)\n\n"
    
    if df_v2 is not None:
        summary_text += f"V2 Model:\n"
        summary_text += f"  Mean Error: {df_v2['pct_error'].abs().mean():.1f}%\n"
        summary_text += f"  Median Error: {df_v2['pct_error'].abs().median():.1f}%\n"
        summary_text += f"  Overest >50%: {(df_v2['pct_error'] > 50).sum():,} "
        summary_text += f"({(df_v2['pct_error'] > 50).mean()*100:.1f}%)\n\n"
        
        if 'cls_attn_community' in df_v2.columns:
            summary_text += f"  Avg Community Attn: {df_v2['cls_attn_community'].mean():.3f}\n"
            summary_text += f"  Avg Property Attn: {df_v2['cls_attn_property'].mean():.3f}\n\n"
    
    summary_text += f"Communities Analyzed: {len(filtered)}\n"
    summary_text += f"(≥{min_records} records)\n\n"
    
    # Communities where V2 is worse
    worse_in_v2 = filtered[filtered['v2_mean_error'] > filtered['v1_mean_error']]
    summary_text += f"V2 Worse than V1:\n"
    summary_text += f"  {len(worse_in_v2)} communities\n"
    summary_text += f"  ({len(worse_in_v2)/len(filtered)*100:.1f}%)\n"
    
    ax8.text(0.1, 0.95, summary_text, transform=ax8.transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle('V1 vs V2 Model: Community-Level Error Analysis', 
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig('outputs/community_error_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to outputs/community_error_comparison.png")


def print_detailed_stats(community_stats):
    """Print detailed statistics"""
    print("\n" + "="*80)
    print("DETAILED COMMUNITY STATISTICS")
    print("="*80)
    
    # Filter to communities with data
    community_stats['max_count'] = community_stats[['v1_count', 'v2_count']].max(axis=1)
    filtered = community_stats[community_stats['max_count'] >= 100].copy()
    
    print(f"\nTop 10 Communities by V2 Error:")
    top_error = filtered.nlargest(10, 'v2_mean_error')[
        ['community', 'v2_count', 'v2_mean_error', 'v1_mean_error', 
         'v2_overest_pct', 'v2_avg_price']
    ]
    print(top_error.to_string(index=False))
    
    print(f"\nTop 10 Communities where V2 is Worse than V1:")
    filtered['v2_worse_by'] = filtered['v2_mean_error'] - filtered['v1_mean_error']
    worst_v2 = filtered.nlargest(10, 'v2_worse_by')[
        ['community', 'v2_count', 'v1_mean_error', 'v2_mean_error', 'v2_worse_by']
    ]
    print(worst_v2.to_string(index=False))
    
    print(f"\nTop 10 Communities where V2 is Better than V1:")
    best_v2 = filtered.nsmallest(10, 'v2_worse_by')[
        ['community', 'v2_count', 'v1_mean_error', 'v2_mean_error', 'v2_worse_by']
    ]
    print(best_v2.to_string(index=False))
    
    if 'v2_community_attn' in filtered.columns:
        print(f"\nCommunities with Lowest Community Attention:")
        low_attn = filtered.nsmallest(10, 'v2_community_attn')[
            ['community', 'v2_count', 'v2_community_attn', 'v2_mean_error']
        ]
        print(low_attn.to_string(index=False))


def main():
    """Run community error comparison"""
    print("="*80)
    print("V1 vs V2: COMMUNITY ERROR COMPARISON")
    print("="*80)
    
    df_v1, df_v2, has_v1, has_v2 = load_data()
    
    if not has_v1 and not has_v2:
        print("\n❌ No predictions found")
        return
    
    print("\nAnalyzing error by community...")
    community_stats = analyze_by_community(df_v1, df_v2)
    
    print_detailed_stats(community_stats)
    
    create_visualizations(community_stats, df_v1, df_v2)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
