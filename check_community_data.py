"""
Check community data consistency between V1 and V2
"""
import pandas as pd

# Load data
df_v1 = pd.read_csv('data/sales_2020_25_with_predictions.csv')
df_v2 = pd.read_csv('data/sales_2020_25_with_predictions_v2.csv')

print("="*80)
print("COMMUNITY DATA CONSISTENCY CHECK")
print("="*80)

print(f"\nV1 Total Records: {len(df_v1)}")
print(f"V2 Total Records: {len(df_v2)}")

print(f"\nV1 Unique Communities: {df_v1['community'].nunique()}")
print(f"V2 Unique Communities: {df_v2['community'].nunique()}")

print(f"\nV1 Community Range: {df_v1['community'].min()} to {df_v1['community'].max()}")
print(f"V2 Community Range: {df_v2['community'].min()} to {df_v2['community'].max()}")

# Check if community column exists and has data
print(f"\nV1 Community column type: {df_v1['community'].dtype}")
print(f"V2 Community column type: {df_v2['community'].dtype}")

print(f"\nV1 Community null count: {df_v1['community'].isna().sum()}")
print(f"V2 Community null count: {df_v2['community'].isna().sum()}")

# Top communities by count
print("\nTop 10 Communities in V1:")
print(df_v1['community'].value_counts().head(10))

print("\nTop 10 Communities in V2:")
print(df_v2['community'].value_counts().head(10))

# Check specific communities
print("\n" + "="*80)
print("SPECIFIC COMMUNITY CHECKS")
print("="*80)

for comm in [10, 106]:
    v1_count = len(df_v1[df_v1['community'] == comm])
    v2_count = len(df_v2[df_v2['community'] == comm])
    
    print(f"\nCommunity {comm}:")
    print(f"  V1: {v1_count} records")
    print(f"  V2: {v2_count} records")
    
    if v1_count > 0:
        v1_sample = df_v1[df_v1['community'] == comm][['community', 'lat', 'lng', 'sale_price', 'pct_error']].head(3)
        print(f"  V1 Sample:")
        print(v1_sample.to_string(index=False))
    
    if v2_count > 0:
        v2_sample = df_v2[df_v2['community'] == comm][['community', 'lat', 'lng', 'sale_price', 'pct_error']].head(3)
        print(f"  V2 Sample:")
        print(v2_sample.to_string(index=False))

# Check if there's a mismatch in how communities are assigned
print("\n" + "="*80)
print("CHECKING FOR DATA ALIGNMENT (sampling 10,000 records)")
print("="*80)

# Check if the same sale_nbr has different communities
if 'sale_nbr' in df_v1.columns and 'sale_nbr' in df_v2.columns:
    # Sample to speed up the merge
    sample_size = min(10000, len(df_v1))
    sample_indices = df_v1.sample(n=sample_size, random_state=42).index
    
    df_v1_sample = df_v1.loc[sample_indices, ['sale_nbr', 'community']]
    df_v2_sample = df_v2.loc[sample_indices, ['sale_nbr', 'community']]
    
    merged = df_v1_sample.merge(
        df_v2_sample, 
        on='sale_nbr', 
        suffixes=('_v1', '_v2')
    )
    
    mismatches = merged[merged['community_v1'] != merged['community_v2']]
    print(f"\nSales with different community assignments: {len(mismatches)} ({len(mismatches)/len(merged)*100:.1f}%)")
    print(f"(Based on sample of {len(merged)} records)")
    
    if len(mismatches) > 0:
        print("\nSample of mismatches:")
        print(mismatches.head(10))
