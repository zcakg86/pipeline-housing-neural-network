"""
Plot V2 vs V3 spatial comparison maps
4 maps: V2 community | V2 error | V3 community | V3 error
Aggregated by H3 level 7
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h3
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors
import contextily as ctx

def load_data():
    """Load V2 and V3 predictions"""
    df_v2 = pd.read_csv('data/sales_2020_25_with_predictions_v2.csv')
    df_v3 = pd.read_csv('data/sales_2020_25_with_predictions_v3.csv')
    print(f"✓ Loaded V2: {len(df_v2)} records")
    print(f"✓ Loaded V3: {len(df_v3)} records")
    return df_v2, df_v3


def lat_lng_to_h3(lat, lng, resolution=7):
    """Convert lat/lng to H3 index"""
    try:
        return h3.latlng_to_cell(lat, lng, resolution)
    except:
        return None


def h3_to_polygon(h3_index):
    """Convert H3 index to polygon coordinates"""
    boundary = h3.cell_to_boundary(h3_index)
    return [(lng, lat) for lat, lng in boundary]


def aggregate_by_h3(df, resolution=7):
    """Aggregate data by H3 hexagons"""
    print(f"  Aggregating {len(df)} records to H3 resolution {resolution}...")
    
    # Add H3 index if not present
    if 'h3_07' not in df.columns:
        df['h3_07'] = df.apply(lambda row: lat_lng_to_h3(row['lat'], row['lng'], resolution), axis=1)
    
    df = df.dropna(subset=['h3_07'])
    
    # Aggregate by H3
    h3_agg = df.groupby('h3_07').agg({
        'pct_error': 'mean',
        'community': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0],  # Most common community
        'sale_price': 'count',
        'lat': 'mean',
        'lng': 'mean'
    }).reset_index()
    
    h3_agg.columns = ['h3_index', 'mean_pct_error', 'community', 'count', 'lat', 'lng']
    
    print(f"  Created {len(h3_agg)} hexagons")
    print(f"  Communities: {h3_agg['community'].nunique()}")
    
    return h3_agg


def plot_h3_map(ax, h3_data, value_col, cmap, norm, title, add_basemap=True):
    """Plot H3 hexagons on map"""
    
    # Create polygons for each hexagon
    patches = []
    colors = []
    
    for _, row in h3_data.iterrows():
        h3_index = row['h3_index']
        value = row[value_col]
        
        # Get hexagon boundary
        polygon_coords = h3_to_polygon(h3_index)
        polygon = Polygon(polygon_coords, closed=True)
        patches.append(polygon)
        colors.append(value)
    
    # Create patch collection
    pc = PatchCollection(patches, cmap=cmap, norm=norm, 
                        edgecolors='white', linewidths=0.3, alpha=0.85, zorder=2)
    pc.set_array(np.array(colors))
    
    ax.add_collection(pc)
    
    # Set axis limits
    lngs = [coord[0] for h3_idx in h3_data['h3_index'] 
            for coord in h3_to_polygon(h3_idx)]
    lats = [coord[1] for h3_idx in h3_data['h3_index'] 
            for coord in h3_to_polygon(h3_idx)]
    
    margin = 0.02
    ax.set_xlim(min(lngs) - margin, max(lngs) + margin)
    ax.set_ylim(min(lats) - margin, max(lats) + margin)
    
    # Add basemap
    if add_basemap:
        try:
            ctx.add_basemap(ax, crs='EPSG:4326', source=ctx.providers.CartoDB.Positron, 
                           alpha=0.4, zorder=1)
        except Exception as e:
            print(f"  Warning: Could not add basemap: {e}")
    
    ax.set_xlabel('Longitude', fontsize=9)
    ax.set_ylabel('Latitude', fontsize=9)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, zorder=3)
    ax.tick_params(labelsize=8)
    
    return pc


def main():
    """Create 4-panel comparison map"""
    print("="*80)
    print("V2 vs V3 SPATIAL COMPARISON MAPS")
    print("="*80)
    
    df_v2, df_v3 = load_data()
    
    # Filter to geographic bounds
    lat_min, lat_max = 47.50, 47.70
    lng_min, lng_max = -122.50, -122.20
    
    print(f"\nFiltering to geographic bounds:")
    print(f"  Latitude: {lat_min} to {lat_max}")
    print(f"  Longitude: {lng_min} to {lng_max}")
    
    df_v2_filtered = df_v2[
        (df_v2['lat'] >= lat_min) & (df_v2['lat'] <= lat_max) &
        (df_v2['lng'] >= lng_min) & (df_v2['lng'] <= lng_max)
    ].copy()
    
    df_v3_filtered = df_v3[
        (df_v3['lat'] >= lat_min) & (df_v3['lat'] <= lat_max) &
        (df_v3['lng'] >= lng_min) & (df_v3['lng'] <= lng_max)
    ].copy()
    
    print(f"  V2: {len(df_v2_filtered)} records (from {len(df_v2)} total)")
    print(f"  V3: {len(df_v3_filtered)} records (from {len(df_v3)} total)")
    
    # Aggregate both datasets
    print("\nAggregating V2 data...")
    v2_h3 = aggregate_by_h3(df_v2_filtered, resolution=7)
    
    print("\nAggregating V3 data...")
    v3_h3 = aggregate_by_h3(df_v3_filtered, resolution=7)
    
    # Create figure with 2x2 layout
    fig = plt.figure(figsize=(20, 18))
    gs = fig.add_gridspec(2, 2, hspace=0.15, wspace=0.12,
                          left=0.05, right=0.95, top=0.94, bottom=0.08)
    
    # Color maps
    # For community IDs: use a categorical colormap
    n_communities_v2 = v2_h3['community'].nunique()
    n_communities_v3 = v3_h3['community'].nunique()
    print(f"\nV2 unique communities: {n_communities_v2}")
    print(f"V3 unique communities: {n_communities_v3}")
    
    # Community colormap - use tab20 repeated if needed
    comm_cmap = plt.cm.get_cmap('tab20')
    
    # For V2 communities
    v2_comm_norm = mcolors.Normalize(vmin=v2_h3['community'].min(), 
                                      vmax=v2_h3['community'].max())
    
    # For V3 communities
    v3_comm_norm = mcolors.Normalize(vmin=v3_h3['community'].min(), 
                                      vmax=v3_h3['community'].max())
    
    # For errors: symmetric around 0
    all_errors = pd.concat([v2_h3['mean_pct_error'], v3_h3['mean_pct_error']])
    vmax = max(abs(all_errors.min()), abs(all_errors.max()))
    vmax = min(vmax, 100)
    
    error_cmap = mcolors.LinearSegmentedColormap.from_list(
        'error_cmap',
        ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0', '#f7f7f7', 
         '#fddbc7', '#f4a582', '#d6604d', '#b2182b']
    )
    error_norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    
    print("\nCreating maps...")
    
    # Top left: V2 Community IDs
    ax1 = fig.add_subplot(gs[0, 0])
    plot_h3_map(ax1, v2_h3, 'community', comm_cmap, v2_comm_norm, 
                'V2: Community IDs', add_basemap=True)
    ax1.text(0.02, 0.98, f'{n_communities_v2} communities\n{len(v2_h3)} hexagons', 
            transform=ax1.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    
    # Top right: V2 Error
    ax2 = fig.add_subplot(gs[0, 1])
    pc2 = plot_h3_map(ax2, v2_h3, 'mean_pct_error', error_cmap, error_norm, 
                      'V2: Mean % Error', add_basemap=True)
    ax2.text(0.02, 0.98, f'MAE: {df_v2_filtered["pct_error"].abs().mean():.1f}%\nMedian: {df_v2_filtered["pct_error"].abs().median():.1f}%', 
            transform=ax2.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    
    # Bottom left: V3 Community IDs
    ax3 = fig.add_subplot(gs[1, 0])
    plot_h3_map(ax3, v3_h3, 'community', comm_cmap, v3_comm_norm, 
                'V3: Community IDs (H3 Mapped)', add_basemap=True)
    ax3.text(0.02, 0.98, f'{n_communities_v3} communities\n{len(v3_h3)} hexagons', 
            transform=ax3.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    
    # Bottom right: V3 Error
    ax4 = fig.add_subplot(gs[1, 1])
    pc4 = plot_h3_map(ax4, v3_h3, 'mean_pct_error', error_cmap, error_norm, 
                      'V3: Mean % Error', add_basemap=True)
    ax4.text(0.02, 0.98, f'MAE: {df_v3_filtered["pct_error"].abs().mean():.1f}%\nMedian: {df_v3_filtered["pct_error"].abs().median():.1f}%', 
            transform=ax4.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    
    # Add single colorbar for error maps at the bottom
    sm = plt.cm.ScalarMappable(cmap=error_cmap, norm=error_norm)
    sm.set_array([])
    
    cbar_ax = fig.add_axes([0.25, 0.03, 0.5, 0.012])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Mean % Error (Blue = Underestimate, White = Accurate, Red = Overestimate)', 
                  fontsize=11, fontweight='bold')
    cbar.ax.tick_params(labelsize=9)
    
    # Add title
    plt.suptitle('V2 vs V3 Spatial Comparison: Community Assignment and Error Distribution\n(Lat: 47.50-47.70, Lng: -122.50 to -122.20, H3 Level 7)', 
                fontsize=15, fontweight='bold', y=0.97)
    
    # Add comparison stats
    improvement = df_v2_filtered['pct_error'].abs().mean() - df_v3_filtered['pct_error'].abs().mean()
    stats_text = (
        f"Overall Improvement: {improvement:+.2f} percentage points | "
        f"V2: {n_communities_v2} communities (clustering) | V3: {n_communities_v3} communities (H3 mapping)"
    )
    fig.text(0.5, 0.005, stats_text, ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # Save
    output_file = 'outputs/v2_v3_spatial_maps_zoomed.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n{'='*80}")
    print(f"✓ Saved visualization to {output_file}")
    print(f"{'='*80}")
    
    # Print summary
    print("\nSummary:")
    print(f"  V2: {n_communities_v2} communities, {len(v2_h3)} hexagons, {df_v2_filtered['pct_error'].abs().mean():.2f}% MAE")
    print(f"  V3: {n_communities_v3} communities, {len(v3_h3)} hexagons, {df_v3_filtered['pct_error'].abs().mean():.2f}% MAE")
    print(f"  Improvement: {improvement:+.2f} percentage points")
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
