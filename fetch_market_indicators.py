"""
Fetch historical market indicators from FRED and join to sales data by sale date.

Indicators:
  - MORTGAGE30US : 30-year fixed mortgage rate (weekly, %)
  - UNRATE       : US civilian unemployment rate (monthly, %)

No API key required - uses FRED's public CSV endpoint.

Usage:
  python3 fetch_market_indicators.py                          # join to default sales CSV
  python3 fetch_market_indicators.py --refetch                # force re-download from FRED
  python3 fetch_market_indicators.py --qa                     # run QA checks only, no join
"""
import sys
import pandas as pd
import numpy as np
import requests
import os
from io import StringIO
from datetime import datetime

FRED_BASE  = "https://fred.stlouisfed.org/graph/fredgraph.csv?id="
INDICATORS = {
    "mortgage_rate":     "MORTGAGE30US",
    "unemployment_rate": "UNRATE",
}
CACHE_DIR  = "data/market_indicators"
CACHE_FILE = os.path.join(CACHE_DIR, "fred_indicators.csv")


# ── FRED fetch ────────────────────────────────────────────────────────────────

def fetch_fred_series(series_id: str, start: str = "2019-01-01") -> pd.Series:
    """Download a FRED series as a dated pandas Series with QA checks."""
    url = f"{FRED_BASE}{series_id}"
    print(f"  Fetching {series_id} from FRED ...")
    print(f"  URL: {url}")

    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"HTTP error fetching {series_id}: {e}") from e
    except requests.exceptions.ConnectionError:
        raise RuntimeError(f"Could not connect to FRED. Check your internet connection.")

    # QA: check response looks like CSV not HTML
    content = r.text.strip()
    if content.startswith("<"):
        preview = content[:300]
        raise RuntimeError(
            f"FRED returned HTML instead of CSV for {series_id}.\n"
            f"Preview: {preview}\n"
            f"Try opening {url} in your browser to check the series ID."
        )
    if not content:
        raise RuntimeError(f"FRED returned empty response for {series_id}")

    # QA: inspect columns before parsing
    first_line = content.split("\n")[0].strip()
    print(f"  Response header: {first_line}")
    columns = [c.strip() for c in first_line.split(",")]

    # FRED uses 'DATE' or 'observation_date' depending on endpoint version
    date_col  = columns[0]
    value_col = columns[1] if len(columns) > 1 else None

    if date_col.upper() not in ("DATE", "OBSERVATION_DATE"):
        raise RuntimeError(
            f"Unexpected first column '{date_col}' in FRED response for {series_id}. "
            f"Full header: {first_line}\n"
            f"Try opening {url} in your browser to verify the series."
        )
    if value_col is None:
        raise RuntimeError(f"Only one column found in FRED response for {series_id}.")

    df = pd.read_csv(StringIO(content))
    df.columns = df.columns.str.strip()
    # Normalise date column name to DATE
    df = df.rename(columns={date_col: "DATE"})
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")

    # QA: check for unparseable dates
    bad_dates = df["DATE"].isna().sum()
    if bad_dates > 0:
        print(f"  ⚠ {bad_dates} rows with unparseable dates — dropping")
        df = df.dropna(subset=["DATE"])

    df = df[df["DATE"] >= start].copy()
    df = df.set_index("DATE")[value_col]

    # FRED uses "." for missing values
    df = df.replace(".", np.nan)
    df = df.astype(float)
    df.name = series_id

    # QA: check for all-NaN
    if df.isna().all():
        raise RuntimeError(f"All values are NaN for {series_id} after {start}.")

    n_missing = df.isna().sum()
    if n_missing > 0:
        print(f"  ⚠ {n_missing}/{len(df)} missing values — will forward-fill")

    # QA: sanity check value ranges
    valid = df.dropna()
    print(f"  ✓ {len(valid)} observations | range: {valid.min():.2f} – {valid.max():.2f} "
          f"| dates: {df.index.min().date()} → {df.index.max().date()}")

    if series_id == "MORTGAGE30US" and (valid.max() > 25 or valid.min() < 0.5):
        print(f"  ⚠ Unusual mortgage rate values — check FRED data")
    if series_id == "UNRATE" and (valid.max() > 25 or valid.min() < 0):
        print(f"  ⚠ Unusual unemployment rate values — check FRED data")

    return df


# ── Build daily indicator table ───────────────────────────────────────────────

def build_indicator_table(start: str = "2019-01-01") -> pd.DataFrame:
    os.makedirs(CACHE_DIR, exist_ok=True)
    series = {}
    for col_name, fred_id in INDICATORS.items():
        s = fetch_fred_series(fred_id, start=start)
        s.name = col_name
        series[col_name] = s

    combined = pd.DataFrame(series)
    daily_idx = pd.date_range(start=combined.index.min(), end=datetime.today(), freq="D")
    combined  = combined.reindex(daily_idx).ffill().bfill()
    combined.index.name = "date"

    # QA: no NaNs after fill
    remaining_na = combined.isna().sum().sum()
    if remaining_na > 0:
        print(f"  ⚠ {remaining_na} NaN values remain after forward/back fill")
    else:
        print(f"  ✓ No missing values after fill")

    combined.to_csv(CACHE_FILE)
    print(f"✓ Saved indicator table → {CACHE_FILE} ({len(combined)} daily rows)\n")
    return combined


# ── QA checks ─────────────────────────────────────────────────────────────────

def qa_indicators(indicators: pd.DataFrame, sales: pd.DataFrame) -> None:
    """Run QA checks on the joined data and print a report."""
    print("\n" + "="*60)
    print("QA REPORT")
    print("="*60)

    # Coverage
    sale_dates = pd.to_datetime(sales["sale_date"]).dt.normalize()
    dates_in_range = sale_dates.between(indicators.index.min(), indicators.index.max())
    print(f"Sale dates in indicator range : {dates_in_range.sum()}/{len(sales)} "
          f"({dates_in_range.mean()*100:.1f}%)")
    if not dates_in_range.all():
        out_of_range = sale_dates[~dates_in_range]
        print(f"  ⚠ {len(out_of_range)} sales outside indicator range")
        print(f"    Earliest sale : {sale_dates.min().date()}")
        print(f"    Latest sale   : {sale_dates.max().date()}")
        print(f"    Indicator from: {indicators.index.min().date()}")
        print(f"    Indicator to  : {indicators.index.max().date()}")

    # Missing values after join
    for col in ["mortgage_rate", "unemployment_rate"]:
        if col in sales.columns:
            n_na = sales[col].isna().sum()
            status = "✓" if n_na == 0 else "⚠"
            print(f"{status} {col}: {n_na} missing after join")

    # Value distribution by year
    if "sale_date" in sales.columns and "mortgage_rate" in sales.columns:
        sales["_year"] = pd.to_datetime(sales["sale_date"]).dt.year
        print("\nMortgage rate by year (mean %):")
        print(sales.groupby("_year")["mortgage_rate"].mean().round(2).to_string())
        print("\nUnemployment rate by year (mean %):")
        print(sales.groupby("_year")["unemployment_rate"].mean().round(2).to_string())
        sales.drop(columns=["_year"], inplace=True)

    # Check constants (bad — means join failed)
    for col in ["mortgage_rate", "unemployment_rate"]:
        if col in sales.columns:
            n_unique = sales[col].nunique()
            if n_unique == 1:
                print(f"  ❌ {col} has only 1 unique value — join likely failed!")
            elif n_unique < 5:
                print(f"  ⚠ {col} has only {n_unique} unique values — check join")
            else:
                print(f"✓ {col}: {n_unique} unique values (looks correct)")

    print("="*60)


# ── Main join ─────────────────────────────────────────────────────────────────

def join_indicators_to_sales(
    sales_path:    str  = "data/sales_2020_25.csv",
    output_path:   str  = None,
    force_refetch: bool = False,
    run_qa:        bool = True
) -> pd.DataFrame:

    if output_path is None:
        output_path = sales_path

    # Load or fetch indicator table
    if os.path.exists(CACHE_FILE) and not force_refetch:
        print(f"Loading cached indicators from {CACHE_FILE} ...")
        indicators = pd.read_csv(CACHE_FILE, index_col="date", parse_dates=True)
        print(f"  ✓ {len(indicators)} daily rows "
              f"({indicators.index.min().date()} → {indicators.index.max().date()})")
    else:
        print("Fetching from FRED ...")
        indicators = build_indicator_table()

    # Load sales
    print(f"\nLoading sales data from {sales_path} ...")
    sales = pd.read_csv(sales_path)
    sales["sale_date"] = pd.to_datetime(sales["sale_date"])
    print(f"  ✓ {len(sales)} records "
          f"({sales['sale_date'].min().date()} → {sales['sale_date'].max().date()})")

    # Join by date
    sale_dates = sales["sale_date"].dt.normalize()
    for col in ["mortgage_rate", "unemployment_rate"]:
        if col in indicators.columns:
            sales[col] = sale_dates.map(indicators[col])

    # Fill gaps (holidays, missing weeks)
    for col in ["mortgage_rate", "unemployment_rate"]:
        if col in sales.columns:
            sales[col] = sales[col].ffill().bfill()

    if run_qa:
        qa_indicators(indicators, sales)

    sales.to_csv(output_path, index=False)
    print(f"\n✓ Saved → {output_path}")
    return sales


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sales",   default="data/sales_2020_25.csv")
    parser.add_argument("--output",  default=None)
    parser.add_argument("--refetch", action="store_true", help="Re-download from FRED")
    parser.add_argument("--qa",      action="store_true", help="QA checks only, no save")
    args = parser.parse_args()

    df = join_indicators_to_sales(
        sales_path=args.sales,
        output_path="/dev/null" if args.qa else args.output,
        force_refetch=args.refetch,
        run_qa=True
    )
    print("\nSample (first 5 rows):")
    cols = ["sale_date", "sale_price", "mortgage_rate", "unemployment_rate"]
    print(df[[c for c in cols if c in df.columns]].head().to_string(index=False))
