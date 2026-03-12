"""
Visualize predicted values for a standard house (3 bed, 1500 sqft) across all H3 indices
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h3
import sys
import os
from datetime import datetime

# Add src to path
sys.path.insert(0, os.getcwd() + '/src/pricemodel')

from model_manager_v2 import modelmanager
from embedding_model_v2 import dataset

print("="*80)
print("STANDARD HOUSE VALUE PREDICTION ACROSS H3 INDICES")
print("="*80)

# Load the V3 model
model_dir = 'outputs/models/20260226_180158'
print(f"\n[1/5] Loading V3 model from {model_dir}...")

model = modelmanager()
model.load_model(model_dir)

print(f"  ✓ Model loaded")
print(f"  Communities: {model.n_communities}")

# Load existing sales data to get H3 indices and their communities
print(f"\n[2/5] Loading existing sales data to get H3 indices...")

sales_df = pd.read_csv('data/sales_2020_25_with_predictions_v3.csv')
print(f"  Loaded {len(sales_df)} sales records")

# Get unique H3 level 7 indices and their communities
h3_data = sales_df[['h3_07', 'community', 'lat', 'lng']].drop_duplicates('h3_07')
print(f"  Found {len(h3_data)} unique H3 level 7 indices")

# Create synthetic data for standard house at each H3 location
print(f"\n[3/5] Creating synthetic data for standard house...")

standard_house = {
    'beds': 3,
    'sqft': 1500,
    'sqft_lot': 5000,  # Standard lot size
    'year_built': 1980,  # Median year
    'year_reno': None,
    'baths': 1.5,
    'sale_date': datetime.now().strftime('%Y-%m-%d'),
    'sale_price': 800000,  # Placeholder, will be predicted
    'sale_nbr': 1
}

print(f"  Standard house specs:")
print(f"    Beds: {standard_house['beds']}")
print(f"    Sqft: {standard_house['sqft']}")
print(f"    Baths: {standard_house['baths']}")
print(f"    Lot: {standard_house['sqft_lot']} sqft")
print(f"    Year Built: {standard_house['year_built']}")

# Create a row for each H3 index
synthetic_data = []
for idx, row in h3_data.iterrows():
    house = standard_house.copy()
    house['h3_07'] = row['h3_07']
    house['community'] = row['community']
    house['lat'] = row['lat']
    house['lng'] = row['lng']
    synthetic_data.append(house)

df = pd.DataFrame(synthetic_data)
print(f"  Created {len(df)} synthetic properties")

# Prepare data using model's processor
print(f"\n[4/5] Processing data through model...")

# Create dataset object with existing vocabularies
data = dataset()
data.reference_date = model.reference_date
data.community_vocab = model.community_vocab
data.year_vocab = model.year_vocab
data.week_vocab = model.week_vocab
data.n_communities = len(model.community_vocab)
data.year_length = len(model.year_vocab)
data.week_length = len(model.week_vocab)

# Prepare data
df['sale_date'] = pd.to_datetime(df['sale_date'])
data = data._prepare_data(df, include_market_indicators=True, future_year_buffer=5)

# Process through model
model.processor(data, scale_mode="transform")

# Generate predictions
print(f"\n[5/5] Generating predictions...")
model.add_predictions_to_data(return_uncertainty=True)

results_df = model.dataframe.copy()
print(f"  ✓ Generated {len(results_df)} predictions")

# Summary statistics
print(f"\nPrediction Summary:")
print(f"  Min: ${results_df['predicted_price'].min():,.0f}")
print(f"  Max: ${results_df['predicted_price'].max():,.0f}")
print(f"  Mean: ${results_df['predicted_price'].mean():,.0f}")
print(f"  Median: ${results_df['predicted_price'].median():,.0f}")
print(f"  Std Dev: ${results_df['predicted_price'].std():,.0f}")

# Save results
output_file = 'outputs/standard_house_predictions_by_h3.csv'
results_df.to_csv(output_file, index=False)
print(f"\n✓ Saved predictions to {output_file}")

# Create visualizations
print(f"\n{'='*80}")
print("CREATING VISUALIZATIONS")
print(f"{'='*80}")

# Figure 1: Scatter plot colored by predicted price
fig, axes = plt.subplots(2, 2, figsize=(18, 16))

# Plot 1: Predicted Price
ax1 = axes[0, 0]
scatter1 = ax1.scatter(
    results_df['lng'],
    results_df['lat'],
    c=results_df['predicted_price'],
    s=50,
    cmap='viridis',
    alpha=0.7,
    edgecolors='black',
    linewidth=0.5
)
ax1.set_xlabel('Longitude', fontsize=11)
ax1.set_ylabel('Latitude', fontsize=11)
ax1.set_title('Predicted Price for Standard House\n(3 bed, 1500 sqft, 1.5 bath)', 
              fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
cbar1 = plt.colorbar(scatter1, ax=ax1)
cbar1.set_label('Predicted Price ($)', fontsize=10)

# Plot 2: Price per sqft
results_df['price_per_sqft'] = results_df['predicted_price'] / results_df['sqft']
ax2 = axes[0, 1]
scatter2 = ax2.scatter(
    results_df['lng'],
    results_df['lat'],
    c=results_df['price_per_sqft'],
    s=50,
    cmap='plasma',
    alpha=0.7,
    edgecolors='black',
    linewidth=0.5
)
ax2.set_xlabel('Longitude', fontsize=11)
ax2.set_ylabel('Latitude', fontsize=11)
ax2.set_title('Predicted Price per Sqft', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
cbar2 = plt.colorbar(scatter2, ax=ax2)
cbar2.set_label('Price per Sqft ($/sqft)', fontsize=10)

# Plot 3: Uncertainty (95% CI width)
results_df['ci_width'] = results_df['price_upper_95'] - results_df['price_lower_95']
ax3 = axes[1, 0]
scatter3 = ax3.scatter(
    results_df['lng'],
    results_df['lat'],
    c=results_df['ci_width'],
    s=50,
    cmap='coolwarm',
    alpha=0.7,
    edgecolors='black',
    linewidth=0.5
)
ax3.set_xlabel('Longitude', fontsize=11)
ax3.set_ylabel('Latitude', fontsize=11)
ax3.set_title('Prediction Uncertainty (95% CI Width)', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
cbar3 = plt.colorbar(scatter3, ax=ax3)
cbar3.set_label('CI Width ($)', fontsize=10)

# Plot 4: Community attention
ax4 = axes[1, 1]
scatter4 = ax4.scatter(
    results_df['lng'],
    results_df['lat'],
    c=results_df['cls_attn_community'],
    s=50,
    cmap='RdYlGn',
    alpha=0.7,
    edgecolors='black',
    linewidth=0.5
)
ax4.set_xlabel('Longitude', fontsize=11)
ax4.set_ylabel('Latitude', fontsize=11)
ax4.set_title('Model Community Attention Weight', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)
cbar4 = plt.colorbar(scatter4, ax=ax4)
cbar4.set_label('Community Attention', fontsize=10)

plt.tight_layout()

output_fig1 = 'outputs/standard_house_value_map.png'
plt.savefig(output_fig1, dpi=300, bbox_inches='tight')
print(f"✓ Saved 4-panel map to {output_fig1}")
plt.close()

# Figure 2: Hexbin plot for smoother visualization
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Hexbin 1: Predicted Price
ax1 = axes[0]
hexbin1 = ax1.hexbin(
    results_df['lng'],
    results_df['lat'],
    C=results_df['predicted_price'],
    gridsize=30,
    cmap='viridis',
    reduce_C_function=np.mean,
    mincnt=1
)
ax1.set_xlabel('Longitude', fontsize=11)
ax1.set_ylabel('Latitude', fontsize=11)
ax1.set_title('Predicted Price (Hexbin Aggregation)\n3 bed, 1500 sqft, 1.5 bath', 
              fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
cbar1 = plt.colorbar(hexbin1, ax=ax1)
cbar1.set_label('Avg Predicted Price ($)', fontsize=10)

# Hexbin 2: Price per sqft
ax2 = axes[1]
hexbin2 = ax2.hexbin(
    results_df['lng'],
    results_df['lat'],
    C=results_df['price_per_sqft'],
    gridsize=30,
    cmap='plasma',
    reduce_C_function=np.mean,
    mincnt=1
)
ax2.set_xlabel('Longitude', fontsize=11)
ax2.set_ylabel('Latitude', fontsize=11)
ax2.set_title('Price per Sqft (Hexbin Aggregation)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
cbar2 = plt.colorbar(hexbin2, ax=ax2)
cbar2.set_label('Avg Price per Sqft ($/sqft)', fontsize=10)

plt.tight_layout()

output_fig2 = 'outputs/standard_house_value_hexbin.png'
plt.savefig(output_fig2, dpi=300, bbox_inches='tight')
print(f"✓ Saved hexbin map to {output_fig2}")
plt.close()

# Figure 3: Top 10 most/least expensive locations
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Top 10 most expensive
top_10 = results_df.nlargest(10, 'predicted_price')
ax1 = axes[0]
scatter_top = ax1.scatter(
    results_df['lng'],
    results_df['lat'],
    c='lightgray',
    s=20,
    alpha=0.3,
    label='All locations'
)
scatter_top_highlight = ax1.scatter(
    top_10['lng'],
    top_10['lat'],
    c=top_10['predicted_price'],
    s=200,
    cmap='Reds',
    alpha=0.9,
    edgecolors='black',
    linewidth=2,
    label='Top 10 most expensive'
)
ax1.set_xlabel('Longitude', fontsize=11)
ax1.set_ylabel('Latitude', fontsize=11)
ax1.set_title('Top 10 Most Expensive Locations', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()
cbar1 = plt.colorbar(scatter_top_highlight, ax=ax1)
cbar1.set_label('Predicted Price ($)', fontsize=10)

# Top 10 least expensive
bottom_10 = results_df.nsmallest(10, 'predicted_price')
ax2 = axes[1]
scatter_bottom = ax2.scatter(
    results_df['lng'],
    results_df['lat'],
    c='lightgray',
    s=20,
    alpha=0.3,
    label='All locations'
)
scatter_bottom_highlight = ax2.scatter(
    bottom_10['lng'],
    bottom_10['lat'],
    c=bottom_10['predicted_price'],
    s=200,
    cmap='Blues',
    alpha=0.9,
    edgecolors='black',
    linewidth=2,
    label='Top 10 least expensive'
)
ax2.set_xlabel('Longitude', fontsize=11)
ax2.set_ylabel('Latitude', fontsize=11)
ax2.set_title('Top 10 Least Expensive Locations', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()
cbar2 = plt.colorbar(scatter_bottom_highlight, ax=ax2)
cbar2.set_label('Predicted Price ($)', fontsize=10)

plt.tight_layout()

output_fig3 = 'outputs/standard_house_extremes.png'
plt.savefig(output_fig3, dpi=300, bbox_inches='tight')
print(f"✓ Saved extremes map to {output_fig3}")
plt.close()

# Print top/bottom locations
print(f"\n{'='*80}")
print("TOP 10 MOST EXPENSIVE LOCATIONS")
print(f"{'='*80}")
for idx, row in top_10.iterrows():
    print(f"  H3: {row['h3_07']} | Community: {row['community']} | "
          f"Price: ${row['predicted_price']:,.0f} | "
          f"$/sqft: ${row['price_per_sqft']:.0f}")

print(f"\n{'='*80}")
print("TOP 10 LEAST EXPENSIVE LOCATIONS")
print(f"{'='*80}")
for idx, row in bottom_10.iterrows():
    print(f"  H3: {row['h3_07']} | Community: {row['community']} | "
          f"Price: ${row['predicted_price']:,.0f} | "
          f"$/sqft: ${row['price_per_sqft']:.0f}")

print(f"\n{'='*80}")
print("COMPLETE")
print(f"{'='*80}")
print(f"\nGenerated files:")
print(f"  - {output_file}")
print(f"  - {output_fig1}")
print(f"  - {output_fig2}")
print(f"  - {output_fig3}")
