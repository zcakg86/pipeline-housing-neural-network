"""
Diagnose Community Attention Issues
Investigate why the model isn't paying attention to community features
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_community_attention():
    """Analyze community attention patterns"""
    
    print("="*80)
    print("COMMUNITY ATTENTION DIAGNOSTIC")
    print("="*80)
    
    # Load V2 predictions
    try:
        df = pd.read_csv('data/sales_2020_25_with_predictions_v2.csv')
        print(f"✓ Loaded {len(df)} predictions")
    except FileNotFoundError:
        print("✗ V2 predictions not found. Run training first.")
        return
    
    # Check if CLS attention exists
    if 'cls_attn_community' not in df.columns:
        print("✗ CLS attention not found in predictions. Retrain model.")
        return
    
    # Overall attention distribution
    print("\n" + "-"*80)
    print("OVERALL ATTENTION DISTRIBUTION")
    print("-"*80)
    
    attention_cols = ['cls_attn_community', 'cls_attn_year', 'cls_attn_week', 
                     'cls_attn_property', 'cls_attn_time', 'cls_attn_market']
    
    for col in attention_cols:
        feature = col.replace('cls_attn_', '').capitalize()
        mean_attn = df[col].mean()
        std_attn = df[col].std()
        min_attn = df[col].min()
        max_attn = df[col].max()
        
        print(f"\n{feature}:")
        print(f"  Mean:   {mean_attn:.4f}")
        print(f"  Std:    {std_attn:.4f}")
        print(f"  Min:    {min_attn:.4f}")
        print(f"  Max:    {max_attn:.4f}")
        print(f"  Range:  {max_attn - min_attn:.4f}")
    
    # Community-specific analysis
    print("\n" + "-"*80)
    print("COMMUNITY-SPECIFIC ANALYSIS")
    print("-"*80)
    
    # Price variance by community
    community_stats = df.groupby('community').agg({
        'sale_price': ['mean', 'std', 'count'],
        'cls_attn_community': 'mean',
        'pct_error': lambda x: x.abs().mean()
    }).round(2)
    
    community_stats.columns = ['avg_price', 'price_std', 'count', 'avg_community_attn', 'avg_error']
    community_stats['price_cv'] = community_stats['price_std'] / community_stats['avg_price']
    community_stats = community_stats.sort_values('count', ascending=False)
    
    print("\nTop 10 Communities by Volume:")
    print(community_stats.head(10).to_string())
    
    # Communities with high vs low attention
    print("\n" + "-"*80)
    print("HIGH VS LOW COMMUNITY ATTENTION")
    print("-"*80)
    
    high_attn_threshold = df['cls_attn_community'].quantile(0.75)
    low_attn_threshold = df['cls_attn_community'].quantile(0.25)
    
    high_attn = df[df['cls_attn_community'] >= high_attn_threshold]
    low_attn = df[df['cls_attn_community'] <= low_attn_threshold]
    
    print(f"\nHigh Community Attention (top 25%, attn >= {high_attn_threshold:.4f}):")
    print(f"  Count: {len(high_attn)}")
    print(f"  Avg error: {high_attn['pct_error'].abs().mean():.2f}%")
    print(f"  Avg price: ${high_attn['sale_price'].mean():,.0f}")
    print(f"  Top communities: {high_attn['community'].value_counts().head(5).to_dict()}")
    
    print(f"\nLow Community Attention (bottom 25%, attn <= {low_attn_threshold:.4f}):")
    print(f"  Count: {len(low_attn)}")
    print(f"  Avg error: {low_attn['pct_error'].abs().mean():.2f}%")
    print(f"  Avg price: ${low_attn['sale_price'].mean():,.0f}")
    print(f"  Top communities: {low_attn['community'].value_counts().head(5).to_dict()}")
    
    # Correlation analysis
    print("\n" + "-"*80)
    print("CORRELATION ANALYSIS")
    print("-"*80)
    
    correlations = df[attention_cols + ['pct_error']].corr()['pct_error'].drop('pct_error')
    print("\nCorrelation between Attention and Prediction Error:")
    for col in attention_cols:
        feature = col.replace('cls_attn_', '').capitalize()
        corr = correlations[col]
        print(f"  {feature:12s}: {corr:+.4f}")
    
    # Check if community embeddings are being used
    print("\n" + "-"*80)
    print("POTENTIAL ISSUES")
    print("-"*80)
    
    issues = []
    
    # Issue 1: Very low community attention
    if df['cls_attn_community'].mean() < 0.10:
        issues.append("⚠️  Community attention is very low (<10%)")
        issues.append("   → Model may not be learning location-specific patterns")
    
    # Issue 2: Low variance in community attention
    if df['cls_attn_community'].std() < 0.05:
        issues.append("⚠️  Community attention has low variance")
        issues.append("   → Model treats all communities similarly")
    
    # Issue 3: Property attention dominates
    if df['cls_attn_property'].mean() > 0.40:
        issues.append("⚠️  Property features dominate (>40% attention)")
        issues.append("   → Model may be over-relying on sqft/beds")
    
    # Issue 4: Time attention too high
    if df['cls_attn_time'].mean() > 0.30:
        issues.append("⚠️  Time features have high attention (>30%)")
        issues.append("   → Model may be over-extrapolating temporal trends")
    
    # Issue 5: Poor performance in diverse communities
    diverse_communities = community_stats[community_stats['price_cv'] > 0.5]
    if len(diverse_communities) > 0:
        avg_error_diverse = df[df['community'].isin(diverse_communities.index)]['pct_error'].abs().mean()
        if avg_error_diverse > df['pct_error'].abs().mean() * 1.2:
            issues.append(f"⚠️  High error in price-diverse communities ({avg_error_diverse:.1f}%)")
            issues.append("   → Community embeddings may not capture local variation")
    
    if issues:
        print("\nIdentified Issues:")
        for issue in issues:
            print(issue)
    else:
        print("\n✓ No major issues detected")
    
    # Recommendations
    print("\n" + "-"*80)
    print("RECOMMENDATIONS")
    print("-"*80)
    
    recommendations = []
    
    if df['cls_attn_community'].mean() < 0.15:
        recommendations.append("1. Increase community embedding dimension")
        recommendations.append("   → Try embedding_dim=32 instead of 16")
        recommendations.append("   → Add more attention heads (8 instead of 4)")
    
    if df['cls_attn_property'].mean() > 0.35:
        recommendations.append("2. Reduce property feature dominance")
        recommendations.append("   → Add more property features (age, condition, etc.)")
        recommendations.append("   → Use feature dropout during training")
    
    if df['cls_attn_time'].mean() > 0.25:
        recommendations.append("3. Balance temporal features")
        recommendations.append("   → Reduce weight on time_trend")
        recommendations.append("   → Add more community-time interactions")
    
    recommendations.append("4. Add community-specific features")
    recommendations.append("   → School ratings by community")
    recommendations.append("   → Crime rates by community")
    recommendations.append("   → Walkability scores by community")
    
    recommendations.append("5. Use community feature aggregation")
    recommendations.append("   → Add mean/median price by community")
    recommendations.append("   → Add price trends by community")
    
    if recommendations:
        print("\nSuggested Improvements:")
        for rec in recommendations:
            print(rec)
    
    return df


def create_visualizations(df):
    """Create diagnostic visualizations"""
    
    print("\n" + "-"*80)
    print("CREATING VISUALIZATIONS")
    print("-"*80)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Community Attention Diagnostic', fontsize=16, fontweight='bold')
    
    attention_cols = ['cls_attn_community', 'cls_attn_year', 'cls_attn_week', 
                     'cls_attn_property', 'cls_attn_time', 'cls_attn_market']
    
    # 1. Attention distribution
    ax = axes[0, 0]
    attention_means = [df[col].mean() for col in attention_cols]
    features = ['Community', 'Year', 'Week', 'Property', 'Time', 'Market']
    colors = ['red' if x < 0.15 else 'green' for x in attention_means]
    ax.bar(features, attention_means, color=colors, alpha=0.7)
    ax.axhline(1/6, color='gray', linestyle='--', label='Equal attention (16.7%)')
    ax.set_ylabel('Average Attention Weight')
    ax.set_title('Average CLS Attention by Feature')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # 2. Attention vs error
    ax = axes[0, 1]
    ax.scatter(df['cls_attn_community'], df['pct_error'].abs(), alpha=0.1, s=1)
    ax.set_xlabel('Community Attention')
    ax.set_ylabel('Absolute % Error')
    ax.set_title('Community Attention vs Prediction Error')
    ax.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(df['cls_attn_community'], df['pct_error'].abs(), 1)
    p = np.poly1d(z)
    x_trend = np.linspace(df['cls_attn_community'].min(), df['cls_attn_community'].max(), 100)
    ax.plot(x_trend, p(x_trend), "r--", linewidth=2, label=f'Trend: {z[0]:.1f}x + {z[1]:.1f}')
    ax.legend()
    
    # 3. Attention distribution histogram
    ax = axes[0, 2]
    ax.hist(df['cls_attn_community'], bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(df['cls_attn_community'].mean(), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {df["cls_attn_community"].mean():.3f}')
    ax.set_xlabel('Community Attention')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Community Attention')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Error by community attention quartile
    ax = axes[1, 0]
    df['attn_quartile'] = pd.qcut(df['cls_attn_community'], q=4, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])
    quartile_errors = df.groupby('attn_quartile')['pct_error'].apply(lambda x: x.abs().mean())
    ax.bar(range(len(quartile_errors)), quartile_errors.values, color='steelblue', alpha=0.7)
    ax.set_xticks(range(len(quartile_errors)))
    ax.set_xticklabels(quartile_errors.index)
    ax.set_ylabel('Mean Absolute % Error')
    ax.set_title('Prediction Error by Community Attention Quartile')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 5. Attention correlation heatmap
    ax = axes[1, 1]
    corr_matrix = df[attention_cols].corr()
    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xticks(range(len(features)))
    ax.set_yticks(range(len(features)))
    ax.set_xticklabels(features, rotation=45)
    ax.set_yticklabels(features)
    ax.set_title('Attention Correlation Matrix')
    
    # Add correlation values
    for i in range(len(features)):
        for j in range(len(features)):
            text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    plt.colorbar(im, ax=ax)
    
    # 6. Top communities by attention
    ax = axes[1, 2]
    top_communities = df.groupby('community')['cls_attn_community'].mean().sort_values(ascending=False).head(10)
    ax.barh(range(len(top_communities)), top_communities.values, color='green', alpha=0.7)
    ax.set_yticks(range(len(top_communities)))
    ax.set_yticklabels([f'Comm {c}' for c in top_communities.index])
    ax.set_xlabel('Average Community Attention')
    ax.set_title('Top 10 Communities by Attention')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('outputs/community_attention_diagnostic.png', dpi=150, bbox_inches='tight')
    print("✓ Saved visualization to outputs/community_attention_diagnostic.png")


def main():
    """Run diagnostic"""
    df = analyze_community_attention()
    
    if df is not None:
        create_visualizations(df)
    
    print("\n" + "="*80)
    print("DIAGNOSTIC COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. Review the recommendations above")
    print("2. Check outputs/community_attention_diagnostic.png")
    print("3. Consider retraining with adjusted architecture")


if __name__ == "__main__":
    main()
