import pandas as pd
from pathlib import Path


def main():
    csv_path = Path('data/sales_2020_25_with_predictions.csv')
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Expected file not found: {csv_path}\n"
            "Please make sure the CSV exists in the data/ directory."
        )

    df = pd.read_csv(csv_path)

    if 'community' not in df.columns:
        raise ValueError("Input CSV must contain a 'community' column.")

    if 'pct_error' not in df.columns:
        raise ValueError("Input CSV must contain a 'pct_error' column.")

    if 'year' not in df.columns:
        if 'sale_date' not in df.columns:
            raise ValueError("Input CSV must contain either a 'year' column or a 'sale_date' column.")
        df['sale_date'] = pd.to_datetime(df['sale_date'], errors='coerce')
        if df['sale_date'].isna().any():
            raise ValueError("Some values in 'sale_date' could not be parsed as dates.")
        df['year'] = df['sale_date'].dt.year

    df['abs_pct_error'] = df['pct_error'].abs()

    counts = df.groupby('community').size().rename('count')
    yearly = (
        df.groupby(['community', 'year'])['abs_pct_error']
          .mean()
          .round(4)
          .rename('mean_abs_pct_error')
          .reset_index()
    )

    table = yearly.pivot(index='community', columns='year', values='mean_abs_pct_error')
    table = table.fillna(0.0)

    summary = pd.concat([counts, table], axis=1)
    summary = summary.sort_values('count', ascending=False)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 120)
    print(summary)

    output_path = Path('outputs/community_error_by_year.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path)
    print(f"\nSaved summary to {output_path}")


if __name__ == '__main__':
    main()
