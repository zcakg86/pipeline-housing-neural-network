import pandas as pd

df_v1 = pd.read_csv('data/sales_2020_25_with_predictions.csv')
df_v2 = pd.read_csv('data/sales_2020_25_with_predictions_v2.csv')

print("Top 5 communities by count (same in both):")
top = df_v2['community'].value_counts().head(5)
print(top)

print("\nDetailed stats for top 2 communities:")
for comm in top.index[:2]:
    v1_subset = df_v1[df_v1['community']==comm]
    v2_subset = df_v2[df_v2['community']==comm]
    
    print(f'\n{"="*60}')
    print(f'Community {comm}')
    print(f'{"="*60}')
    print(f'Records: {len(v2_subset)}')
    print(f'V1 Error: {v1_subset["pct_error"].abs().mean():.1f}%')
    print(f'V2 Error: {v2_subset["pct_error"].abs().mean():.1f}%')
    print(f'Lat: {v2_subset["lat"].min():.3f} to {v2_subset["lat"].max():.3f}')
    print(f'Lng: {v2_subset["lng"].min():.3f} to {v2_subset["lng"].max():.3f}')
    print(f'V2 Overest >50%: {(v2_subset["pct_error"] > 50).sum()} ({(v2_subset["pct_error"] > 50).mean()*100:.1f}%)')
