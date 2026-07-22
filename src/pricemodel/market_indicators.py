"""Causal, backward-as-of joins for dated economic indicators."""
from __future__ import annotations

import numpy as np
import pandas as pd

INDICATOR_COLUMNS = ("mortgage_rate", "unemployment_rate")


def join_indicators_backward_asof(
    sales: pd.DataFrame,
    indicators: pd.DataFrame,
    *,
    sale_date_col: str = "sale_date",
    require_complete: bool = False,
) -> pd.DataFrame:
    """Attach the latest indicator observation available on each sale date.

    Each indicator is joined independently because the source series have
    different frequencies.  ``direction='backward'`` is the important causal
    constraint: an observation dated after a sale can never be selected.
    No backward-fill is performed.
    """
    if sale_date_col not in sales.columns:
        raise KeyError(f"Missing sale date column: {sale_date_col}")

    result = sales.copy()
    sale_dates = pd.to_datetime(result[sale_date_col], errors="coerce").dt.normalize()

    indicator_frame = indicators.copy()
    if "date" in indicator_frame.columns:
        indicator_dates = pd.to_datetime(
            indicator_frame.pop("date"), errors="coerce"
        ).dt.normalize()
    else:
        indicator_dates = pd.Series(
            pd.to_datetime(indicator_frame.index, errors="coerce"),
            index=indicator_frame.index,
        ).dt.normalize()
    indicator_frame = indicator_frame.reset_index(drop=True)
    indicator_frame.insert(0, "_indicator_date", indicator_dates.to_numpy())
    indicator_frame = indicator_frame.dropna(subset=["_indicator_date"])

    valid_sales = pd.DataFrame({
        "_row_number": np.arange(len(result), dtype=np.int64),
        "_sale_date": sale_dates.to_numpy(),
    }).dropna(subset=["_sale_date"]).sort_values("_sale_date")

    joined_columns = []
    for column in INDICATOR_COLUMNS:
        if column not in indicator_frame.columns:
            # Never retain a possibly pre-joined value from the sales CSV when
            # its dated source series is unavailable for verification.
            result[column] = np.nan
            joined_columns.append(column)
            continue

        observations = indicator_frame[["_indicator_date", column]].copy()
        observations[column] = pd.to_numeric(observations[column], errors="coerce")
        observations = (
            observations
            .dropna(subset=[column])
            .sort_values("_indicator_date")
            .drop_duplicates("_indicator_date", keep="last")
        )
        if observations.empty:
            result[column] = np.nan
            joined_columns.append(column)
            continue

        matched = pd.merge_asof(
            valid_sales,
            observations,
            left_on="_sale_date",
            right_on="_indicator_date",
            direction="backward",
            allow_exact_matches=True,
        )
        values = np.full(len(result), np.nan, dtype=np.float64)
        values[matched["_row_number"].to_numpy(dtype=np.int64)] = matched[column]
        result[column] = values
        joined_columns.append(column)

    if require_complete and joined_columns:
        valid_date_rows = sale_dates.notna()
        missing = result.loc[valid_date_rows, joined_columns].isna()
        if missing.any().any():
            counts = ", ".join(
                f"{column}={int(missing[column].sum())}"
                for column in joined_columns
                if missing[column].any()
            )
            earliest_sale = sale_dates.loc[valid_date_rows].min().date()
            raise ValueError(
                "Market indicators do not have historical coverage for every "
                f"sale (earliest sale {earliest_sale}; missing {counts}). "
                "Future observations will not be backfilled. Fetch an earlier "
                "indicator history or remove the uncovered sales."
            )

    return result
