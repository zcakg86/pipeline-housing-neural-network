"""
Find equivalent communities between V1 and V2 based on geographic overlap
"""
import pandas as pd
import numpy as np

# Load data
df_v1 = pd.read_csv('data/sales_2020_25_with_predictions.csv')
df_v2 = pd.read_csv('data/sales_2020_25_with_predictions_v2.csv')

print("="*80)
print("FINDING EQUIVALENT COMMUNITIES BETWEEN V1 AND V2")
print("="*80)

# Get community centroids for V2 communities 10 and 106
target_communities = [10, 106]

for v2_comm in target_communities:
    print(f"\n{'='*80}")
    print(f"V2 Community {v2_comm}")
    print(f"{'='*80}")
    
    v2_data = df_v2[df_v2['community'] == v2_comm]
    
    print(f"Records: {len(v2_data)}")
    print(f"Lat range: {v2_data['lat'].min():.4f} to {v2_data['lat'].max():.4f}")
    print(f"Lng range: {v2_data['lng'].min():.4f} to {v2_data['lng'].max():.4f}")
    print(f"Centroid: ({v2_data['lat'].mean():.4f}, {v2_data['lng'].mean():.4f})")
    print(f"Avg price: ${v2_data['sale_price'].mean():,.0f}")
    print(f"V2 Mean error: {v2_data['pct_error'].abs().mean():.1f}%")
    
    # Find V1 communities that overlap with this geographic area
    lat_min, lat_max = v2_data['lat'].min(), v2_data['lat'].max()
    lng_min, lng_max = v2_data['lng'].min(), v2_data['lng'].max()
    
    # Find V1 records in the same geographic area
    v1_overlap = df_v1[
        (df_v1['lat'] >= lat_min) & (df_v1['lat'] <= lat_max) &
        (df_v1['lng'] >= lng_min) & (df_v1['lng'] <= lng_max)
    ]
    
    print(f"\nV1 records in same geographic area: {len(v1_overlap)}")
    
    if len(v1_overlap) > 0:
        print(f"\nV1 communities in this area:")
        v1_comm_counts = v1_overlap['community'].value_counts().head(10)
        for comm, count in v1_comm_counts.items():
            v1_comm_data = df_v1[df_v1['community'] == comm]
            overlap_pct = count / len(v1_comm_data) * 100
            print(f"  Community {comm}: {count} records ({overlap_pct:.1f}% of that community)")
            print(f"    Total in V1: {len(v1_comm_data)}")
            print(f"    V1 Mean error: {v1_comm_data['pct_error'].abs().mean():.1f}%")
            print(f"    Centroid: ({v1_comm_data['lat'].mean():.4f}, {v1_comm_data['lng'].mean():.4f})")
            print(f"    Avg price: ${v1_comm_data['sale_price'].mean():,.0f}")

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)
print("\nThe community IDs are different between V1 and V2 because they use")
print("different clustering/assignment methods. To compare equivalent areas:")
print("1. Use the V1 community IDs identified above that overlap geographically")
print("2. Or compare by geographic regions (lat/lng bounds) instead of community ID")
