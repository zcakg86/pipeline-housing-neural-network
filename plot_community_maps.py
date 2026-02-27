"""
Plot spatial maps of V1 vs V2 errors for specific communities
Uses H3 hexagons for spatial aggregation
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
    """Load both model predictions"""
    try:
        df_v1 = pd.read_csv('data/sales_2020_25_with_predictions.csv')
        print(f"✓ Loaded V1: {len(df_v1)} records")
    except FileNotFoundError:
        print("✗ V1 predictions not found")
        return None, None
    
    try:
        df_v2 = pd.read_csv('data/sales_2020_25_with_predictions_v2.csv')
        print(f"✓ Loaded V2: {len(df_v2)} records")
    except FileNotFoundError:
        print("✗ V2 predictions not found")
        return None, None
    
    return df_v1, df_v2


def lat_lng_to_h3(lat, lng, resolution=8):
    """Convert lat/lng to H3 index"""
    try:
        return h3.latlng_to_cell(lat, lng, resolution)
    except:
        return None


def h3_to_polygon(h3_index):
    """Convert H3 index to polygon coordinates"""
    boundary = h3.cell_to_boundary(h3_index)
    # H3 returns (lat, lng), matplotlib needs (lng, lat) for x, y
    return [(lng, lat) for lat, lng in boundary]


def aggregate_by_h3(df, resolution=8):
    """Aggregate data by H3 hexagons"""
    print(f"  Aggregating {len(df)} records to H3 resolution {resolution}...")
    
    # Add H3 index
    df['h3_index'] = df.apply(lambda row: lat_lng_to_h3(row['lat'], row['lng'], resolution), axis=1)
    df = df.dropna(subset=['h3_index'])
    
    # Aggregate by H3
    h3_agg = df.groupby('h3_index').agg({
        'pct_error': ['mean', 'median', 'count'],
        'sale_price': 'mean',
        'lat': 'mean',
        'lng': 'mean'
    }).reset_index()
    
    h3_agg.columns = ['h3_index', 'mean_pct_error', 'median_pct_error', 'count', 'avg_price', 'lat', 'lng']
    
    print(f"  Created {len(h3_agg)} hexagons")
    
    return h3_agg


def plot_community_comparison(community_id, df_v1, df_v2, resolution=8):
    """Plot V1 vs V2 error maps for a specific community"""
    
    print(f"\nProcessing Community {community_id}...")
    
    # Filter to community
    v1_comm = df_v1[df_v1['community'] == community_id].copy()
    v2_comm = df_v2[df_v2['community'] == community_id].copy()
    
    if len(v1_comm) == 0 or len(v2_comm) == 0:
        print(f"  ⚠️  No data found for community {community_id}")
        return None
    
    print(f"  V1: {len(v1_comm)} records, mean error: {v1_comm['pct_error'].abs().mean():.1f}%")
    print(f"  V2: {len(v2_comm)} records, mean error: {v2_comm['pct_error'].abs().mean():.1f}%")
    
    # Aggregate by H3
    v1_h3 = aggregate_by_h3(v1_comm, resolution)
    v2_h3 = aggregate_by_h3(v2_comm, resolution)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Determine color scale (symmetric around 0)
    all_errors = pd.concat([v1_h3['mean_pct_error'], v2_h3['mean_pct_error']])
    vmax = max(abs(all_errors.min()), abs(all_errors.max()))
    vmax = min(vmax, 100)  # Cap at 100% for visualization
    
    # Color map: blue (underestimate) -> white (accurate) -> red (overestimate)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'error_cmap',
        ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0', '#f7f7f7', 
         '#fddbc7', '#f4a582', '#d6604d', '#b2182b']
    )
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    
    # Plot V1
    ax1 = axes[0]
    plot_h3_map(ax1, v1_h3, cmap, norm, f'Community {community_id} - V1 Model')
    
    # Plot V2
    ax2 = axes[1]
    plot_h3_map(ax2, v2_h3, cmap, norm, f'Community {community_id} - V2 Model')
    
    # Add shared colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', 
                       pad=0.05, aspect=40, shrink=0.8)
    cbar.set_label('Mean % Error (negative = underestimate, positive = overestimate)', 
                   fontsize=11, fontweight='bold')
    
    # Add statistics text
    stats_text = (
        f"V1: {len(v1_comm)} sales, {len(v1_h3)} hexagons\n"
        f"Mean error: {v1_comm['pct_error'].mean():.1f}% | "
        f"MAE: {v1_comm['pct_error'].abs().mean():.1f}%\n"
        f"Overest >50%: {(v1_comm['pct_error'] > 50).sum()} ({(v1_comm['pct_error'] > 50).mean()*100:.1f}%)"
    )
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    stats_text = (
        f"V2: {len(v2_comm)} sales, {len(v2_h3)} hexagons\n"
        f"Mean error: {v2_comm['pct_error'].mean():.1f}% | "
        f"MAE: {v2_comm['pct_error'].abs().mean():.1f}%\n"
        f"Overest >50%: {(v2_comm['pct_error'] > 50).sum()} ({(v2_comm['pct_error'] > 50).mean()*100:.1f}%)"
    )
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle(f'Spatial Error Distribution: Community {community_id}', 
                fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    return fig


def plot_h3_map(ax, h3_data, cmap, norm, title):
    """Plot H3 hexagons on map with basemap"""
    
    # Create polygons for each hexagon
    patches = []
    colors = []
    
    for _, row in h3_data.iterrows():
        h3_index = row['h3_index']
        error = row['mean_pct_error']
        
        # Get hexagon boundary
        polygon_coords = h3_to_polygon(h3_index)
        polygon = Polygon(polygon_coords, closed=True)
        patches.append(polygon)
        colors.append(error)
    
    # Create patch collection
    pc = PatchCollection(patches, cmap=cmap, norm=norm, 
                        edgecolors='white', linewidths=0.5, alpha=0.75, zorder=2)
    pc.set_array(np.array(colors))
    
    ax.add_collection(pc)
    
    # Set axis limits
    lngs = [coord[0] for h3_idx in h3_data['h3_index'] 
            for coord in h3_to_polygon(h3_idx)]
    lats = [coord[1] for h3_idx in h3_data['h3_index'] 
            for coord in h3_to_polygon(h3_idx)]
    
    margin = 0.01
    ax.set_xlim(min(lngs) - margin, max(lngs) + margin)
    ax.set_ylim(min(lats) - margin, max(lats) + margin)
    
    # Add basemap
    try:
        ctx.add_basemap(ax, crs='EPSG:4326', source=ctx.providers.CartoDB.Positron, 
                       alpha=0.5, zorder=1)
    except Exception as e:
        print(f"  Warning: Could not add basemap: {e}")
    
    ax.set_xlabel('Longitude', fontsize=10)
    ax.set_ylabel('Latitude', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, zorder=3)
    
    # Add hexagon count annotation
    ax.text(0.98, 0.02, f'{len(h3_data)} hexagons\nH3 res: 8', 
           transform=ax.transAxes, fontsize=8,
           horizontalalignment='right', verticalalignment='bottom',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


def main():
    """Create spatial comparison maps"""
    print("="*80)
    print("SPATIAL ERROR MAPS: V1 vs V2 BY COMMUNITY")
    print("="*80)
    
    df_v1, df_v2 = load_data()
    
    if df_v1 is None or df_v2 is None:
        print("\n❌ Could not load data")
        return
    
    # Communities to analyze (using actual large communities with data)
    # Community 82: 17,211 records, V2 better (20.8% vs V1 30.4%)
    # Community 278: 15,989 records, V2 better (22.1% vs V1 28.6%)
    communities = [82, 278]
    
    # Create combined figure with both communities
    fig = plt.figure(figsize=(18, 18))
    gs = fig.add_gridspec(2, 2, hspace=0.15, wspace=0.12, 
                          left=0.08, right=0.92, top=0.94, bottom=0.08)
    
    all_h3_data = []  # Store all h3 data for global color scale
    
    # First pass: collect all data for consistent color scale
    for community_id in communities:
        v1_comm = df_v1[df_v1['community'] == community_id].copy()
        v2_comm = df_v2[df_v2['community'] == community_id].copy()
        
        if len(v1_comm) > 0 and len(v2_comm) > 0:
            v1_h3 = aggregate_by_h3(v1_comm, resolution=8)
            v2_h3 = aggregate_by_h3(v2_comm, resolution=8)
            all_h3_data.append(v1_h3)
            all_h3_data.append(v2_h3)
    
    # Determine global color scale
    all_errors = pd.concat([df['mean_pct_error'] for df in all_h3_data])
    vmax = max(abs(all_errors.min()), abs(all_errors.max()))
    vmax = min(vmax, 100)
    
    # Color map
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'error_cmap',
        ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0', '#f7f7f7', 
         '#fddbc7', '#f4a582', '#d6604d', '#b2182b']
    )
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    
    # Second pass: create plots
    for idx, community_id in enumerate(communities):
        print(f"\n{'='*80}")
        print(f"Processing Community {community_id}...")
        print(f"{'='*80}")
        
        # Filter to community (same ID in both V1 and V2 now)
        v1_comm = df_v1[df_v1['community'] == community_id].copy()
        v2_comm = df_v2[df_v2['community'] == community_id].copy()
        
        if len(v1_comm) == 0 or len(v2_comm) == 0:
            print(f"  ⚠️  No data found for community {community_id}")
            continue
        
        print(f"  V1: {len(v1_comm)} records, mean error: {v1_comm['pct_error'].mean():.1f}%")
        print(f"  V2: {len(v2_comm)} records, mean error: {v2_comm['pct_error'].mean():.1f}%")
        
        # Aggregate by H3
        v1_h3 = aggregate_by_h3(v1_comm, resolution=8)
        v2_h3 = aggregate_by_h3(v2_comm, resolution=8)
        
        # Plot V1 (left column)
        ax1 = fig.add_subplot(gs[idx, 0])
        plot_h3_map(ax1, v1_h3, cmap, norm, f'Community {community_id} - V1 Model')
        
        # Add statistics
        stats_text = (
            f"V1 Stats:\n"
            f"Sales: {len(v1_comm)}\n"
            f"Hexagons: {len(v1_h3)}\n"
            f"Mean: {v1_comm['pct_error'].mean():.1f}%\n"
            f"MAE: {v1_comm['pct_error'].abs().mean():.1f}%\n"
            f">50%: {(v1_comm['pct_error'] > 50).sum()} "
            f"({(v1_comm['pct_error'] > 50).mean()*100:.1f}%)"
        )
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
                fontsize=8, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        # Plot V2 (right column)
        ax2 = fig.add_subplot(gs[idx, 1])
        plot_h3_map(ax2, v2_h3, cmap, norm, f'Community {community_id} - V2 Model')
        
        # Add statistics
        stats_text = (
            f"V2 Stats:\n"
            f"Sales: {len(v2_comm)}\n"
            f"Hexagons: {len(v2_h3)}\n"
            f"Mean: {v2_comm['pct_error'].mean():.1f}%\n"
            f"MAE: {v2_comm['pct_error'].abs().mean():.1f}%\n"
            f">50%: {(v2_comm['pct_error'] > 50).sum()} "
            f"({(v2_comm['pct_error'] > 50).mean()*100:.1f}%)"
        )
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
                fontsize=8, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Add single colorbar at the bottom for all plots
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    # Position colorbar at bottom, below all plots
    cbar_ax = fig.add_axes([0.15, 0.03, 0.7, 0.015])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Mean % Error (Blue = Underestimate, White = Accurate, Red = Overestimate)', 
                  fontsize=11, fontweight='bold')
    cbar.ax.tick_params(labelsize=9)
    
    plt.suptitle('Spatial Error Distribution: V1 vs V2 Models (H3 Resolution 8)', 
                fontsize=16, fontweight='bold', y=0.97)
    
    plt.savefig('outputs/community_spatial_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n{'='*80}")
    print("✓ Saved visualization to outputs/community_spatial_comparison.png")
    print(f"{'='*80}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
