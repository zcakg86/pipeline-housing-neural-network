"""
Map sales data points for the 10 biggest communities, colored by community
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print("Loading sales data...")

# Load historical sales
sales_df = pd.read_csv('data/sales_2020_25_with_predictions_v3.csv')
print(f"Loaded {len(sales_df)} historical sales")

# Find top 10 communities by number of sales
top_communities = sales_df['community'].value_counts().head(10)
print(f"\nTop 10 Communities by Sales Count:")
for comm, count in top_communities.items():
    print(f"  Community {comm}: {count} sales")

# Filter to top 10 communities
df = sales_df[sales_df['community'].isin(top_communities.index)].copy()
print(f"\nFiltered to {len(df)} sales in top 10 communities")

# Create figure
fig, ax = plt.subplots(figsize=(16, 12))

# Create color map for communities
communities = sorted(df['community'].unique())
colors = plt.cm.tab10(np.linspace(0, 1, len(communities)))
color_map = dict(zip(communities, colors))

# Plot each community
for i, comm in enumerate(communities):
    comm_data = df[df['community'] == comm]
    count = len(comm_data)
    
    ax.scatter(
        comm_data['lng'],
        comm_data['lat'],
        c=[color_map[comm]],
        label=f'Community {comm} (n={count:,})',
        alpha=0.6,
        s=20,
        edgecolors='black',
        linewidth=0.3
    )

# Styling
ax.set_xlabel('Longitude', fontsize=12, fontweight='bold')
ax.set_ylabel('Latitude', fontsize=12, fontweight='bold')
ax.set_title('Top 10 Communities by Sales Volume\nSeattle Area (2020-2025)', 
             fontsize=14, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, linestyle='--')

# Legend
ax.legend(
    loc='upper left',
    bbox_to_anchor=(1.02, 1),
    fontsize=10,
    framealpha=0.9,
    title='Communities',
    title_fontsize=11
)

# Add statistics box
stats_text = f"""Total Sales: {len(df):,}
Communities: {len(communities)}
Date Range: {df['sale_date'].min()} to {df['sale_date'].max()}
Avg Price: ${df['sale_price'].mean():,.0f}"""

ax.text(
    0.02, 0.98,
    stats_text,
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
)

plt.tight_layout()

# Save figure
output_file = 'outputs/top_10_communities_map.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved map to {output_file}")

plt.close()

# Create a second figure with community boundaries (convex hull)
print("\nCreating community boundary map...")

from scipy.spatial import ConvexHull

fig, ax = plt.subplots(figsize=(16, 12))

# Plot each community with boundary
for i, comm in enumerate(communities):
    comm_data = df[df['community'] == comm]
    count = len(comm_data)
    
    # Plot points
    ax.scatter(
        comm_data['lng'],
        comm_data['lat'],
        c=[color_map[comm]],
        label=f'Community {comm} (n={count:,})',
        alpha=0.6,
        s=20,
        edgecolors='black',
        linewidth=0.3,
        zorder=2
    )
    
    # Draw convex hull if enough points
    if len(comm_data) >= 3:
        try:
            points = comm_data[['lng', 'lat']].values
            hull = ConvexHull(points)
            
            # Plot hull
            for simplex in hull.simplices:
                ax.plot(
                    points[simplex, 0],
                    points[simplex, 1],
                    color=color_map[comm],
                    linewidth=2,
                    alpha=0.5,
                    zorder=1
                )
            
            # Fill hull
            hull_points = points[hull.vertices]
            ax.fill(
                hull_points[:, 0],
                hull_points[:, 1],
                color=color_map[comm],
                alpha=0.1,
                zorder=0
            )
            
        except Exception as e:
            print(f"  Warning: Could not create hull for community {comm}: {e}")

# Styling
ax.set_xlabel('Longitude', fontsize=12, fontweight='bold')
ax.set_ylabel('Latitude', fontsize=12, fontweight='bold')
ax.set_title('Top 10 Communities with Boundaries\nSeattle Area (2020-2025)', 
             fontsize=14, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, linestyle='--')

# Legend
ax.legend(
    loc='upper left',
    bbox_to_anchor=(1.02, 1),
    fontsize=10,
    framealpha=0.9,
    title='Communities',
    title_fontsize=11
)

# Add statistics box
ax.text(
    0.02, 0.98,
    stats_text,
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
)

plt.tight_layout()

# Save figure
output_file2 = 'outputs/top_10_communities_with_boundaries.png'
plt.savefig(output_file2, dpi=300, bbox_inches='tight')
print(f"✓ Saved boundary map to {output_file2}")

plt.close()

# Create summary statistics by community
print("\nCommunity Statistics:")
print("="*80)

summary = df.groupby('community').agg({
    'sale_price': ['count', 'mean', 'median', 'min', 'max'],
    'sqft': 'mean',
    'beds': 'mean',
    'bath_full': 'mean',
    'pct_error': lambda x: x.abs().mean()
}).round(2)

summary.columns = ['Count', 'Avg Price', 'Median Price', 'Min Price', 'Max Price', 
                   'Avg Sqft', 'Avg Beds', 'Avg Baths', 'Avg Error %']

# Sort by count
summary = summary.sort_values('Count', ascending=False)

print(summary.to_string())

# Save summary to CSV
summary_file = 'outputs/top_10_communities_summary.csv'
summary.to_csv(summary_file)
print(f"\n✓ Saved summary statistics to {summary_file}")

print("\n" + "="*80)
print("COMPLETE")
print("="*80)
