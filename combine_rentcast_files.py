"""
Combine RentCast data files and reformat dates
"""
import pandas as pd

# Read both files
print("Reading files...")
df1 = pd.read_csv('data/rentcast_recent_sales_first500.csv')
df2 = pd.read_csv('data/rentcast_recent_sales5001000.csv')

print(f"File 1: {len(df1)} records")
print(f"File 2: {len(df2)} records")

# Check current date format in first file
print(f"\nFile 1 sample dates:")
print(df1['sale_date'].head(3))

print(f"\nFile 2 sample dates:")
print(df2['sale_date'].head(3))

# Reformat sale_date in first file to YYYY-MM-DD
print("\nReformatting dates in first file...")
df1['sale_date'] = pd.to_datetime(df1['sale_date']).dt.strftime('%Y-%m-%d')

# Ensure second file also has correct format
print("Ensuring dates in second file are formatted...")
df2['sale_date'] = pd.to_datetime(df2['sale_date']).dt.strftime('%Y-%m-%d')

print(f"\nFile 1 reformatted dates:")
print(df1['sale_date'].head(3))

print(f"\nFile 2 reformatted dates:")
print(df2['sale_date'].head(3))

# Combine the files
print("\nCombining files...")
df_combined = pd.concat([df1, df2], ignore_index=True)

print(f"Combined: {len(df_combined)} records")

# Remove duplicates based on sale_nbr if any
print("\nChecking for duplicates...")
duplicates = df_combined.duplicated(subset=['sale_nbr'], keep='first').sum()
if duplicates > 0:
    print(f"Found {duplicates} duplicates, removing...")
    df_combined = df_combined.drop_duplicates(subset=['sale_nbr'], keep='first')
    print(f"After removing duplicates: {len(df_combined)} records")
else:
    print("No duplicates found")

# Save combined file
output_file = 'data/rentcast_recent_sales.csv'
df_combined.to_csv(output_file, index=False)

print(f"\n✓ Saved combined file to {output_file}")
print(f"\nSummary:")
print(f"  Total records: {len(df_combined)}")
print(f"  Date range: {df_combined['sale_date'].min()} to {df_combined['sale_date'].max()}")
print(f"  Price range: ${df_combined['sale_price'].min():,.0f} to ${df_combined['sale_price'].max():,.0f}")
print(f"  Median price: ${df_combined['sale_price'].median():,.0f}")

# Show sample
print("\nSample records:")
print(df_combined[['sale_date', 'sale_price', 'sqft', 'beds', 'baths', 'city']].head(10))
