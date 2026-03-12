"""Test loading data for the map"""
import pandas as pd

print("Testing data load...")

try:
    # Zillow
    zillow_df = pd.read_csv('data/zillow_with_predictions.csv')
    print(f"✓ Zillow: {len(zillow_df)} rows")
    print(f"  Columns: {list(zillow_df.columns)}")
    
    # Sales
    sales_df = pd.read_csv('data/sales_2020_25_with_predictions_v3.csv')
    print(f"✓ Sales: {len(sales_df)} rows")
    print(f"  Has 'bath_full': {'bath_full' in sales_df.columns}")
    print(f"  Has 'baths': {'baths' in sales_df.columns}")
    
    # RentCast
    rentcast_df = pd.read_csv('data/rentcast_with_predictions_v3_finetuned.csv')
    print(f"✓ RentCast: {len(rentcast_df)} rows")
    print(f"  Has 'bath_full': {'bath_full' in rentcast_df.columns}")
    print(f"  Has 'baths': {'baths' in rentcast_df.columns}")
    
    # Check if columns match
    sales_cols = set(sales_df.columns)
    rentcast_cols = set(rentcast_df.columns)
    
    missing_in_rentcast = sales_cols - rentcast_cols
    missing_in_sales = rentcast_cols - sales_cols
    
    if missing_in_rentcast:
        print(f"\n⚠ Columns in sales but not rentcast: {missing_in_rentcast}")
    if missing_in_sales:
        print(f"\n⚠ Columns in rentcast but not sales: {missing_in_sales}")
    
    print("\n✓ All data files loaded successfully")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
