"""Dataset preparation for the neighborhood-aware house-price models.

This module owns deterministic feature engineering and vocabulary construction.
Network architecture and optimization live in separate modules.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import torch

from .h3_neighbor_mapper import ensure_neighbor_cells, ensure_neighbor_communities
from .local_market_features import build_historical_local_features
from .market_indicators import join_indicators_backward_asof
from .water_features import DEFAULT_WATER_PATH, add_water_proximity_features


DEFAULT_MARKET_INDICATOR_CACHE = "data/market_indicators/fred_indicators.csv"

class DatasetBuilder:
    """Prepare leakage-safe spatial, temporal, economic, and water features.

    The builder retains the enriched dataframe and deployment metadata needed by
    both neural and tree trainers. Network tensors and scaler fitting remain the
    responsibility of ``ModelManager`` so train/validation boundaries are known
    before any statistics are fitted.
    """
    def __init__(self):
        self.length = None
        self.n_communities = None   # set by _map_communities(); unknown index = n_communities
        self.scalers = {}
        self.indices = []
        self.timestamp = None
        self.community_df = pd.DataFrame()
        self.community_array = np.empty(0)
        self.community_feature_dim = None
        self.community_indices = torch.empty(0)
        self.property_features = torch.empty(0)
        self.continuous_time_features = torch.empty(0)
        self.market_features = torch.empty(0)
        self.target = torch.empty(0)
        self.reference_date = None
        self.local_market_features = None
        self.local_market_snapshot = None
        self.neighbor_cells_map = None
        self.community_map_path = None
        self.neighbor_cells_path = None
        self.local_feature_cache_dir = None

    def _map_communities(self, dataframe,
                         community_map_path='community_map.json',
                         output_dir='data'):
        """
        Ensure H3  neighbor mappings exist and map each row to a 7-element
        community index vector (center hex + 6 neighbours).

        Must be called before _prepare_data so that 'community_neighbors' and
        'n_communities' are ready when the tensor is built.

        Indices in community_neighbors are already 0-based (0 … n_communities-1),
        with n_communities used as the unknown/padding index.

        Parameters:
            dataframe:            Raw input DataFrame (mutated in-place on the copy
                                  stored as self._raw_df for _prepare_data to pick up)
            community_map_path:   Path to H3 (L8) → community ID JSON
            output_dir:           Where cached mapping files live / are written
        """
        import h3 as _h3

        df = dataframe.copy()

        # --- Ensure h3_09 column ---
        if 'h3_08' not in df.columns:
            if 'lat' in df.columns and 'lng' in df.columns:
                print("  Generating H3 L8 indices from lat/lng...")
                df['h3_08'] = df.apply(
                    lambda row: _h3.latlng_to_cell(row['lat'], row['lng'], 8)
                    if pd.notna(row['lat']) and pd.notna(row['lng'])
                    else None,
                    axis=1
                )
                print(f"  ✓ Generated H3 L8 indices for {df['h3_08'].notna().sum()} properties")
            else:
                print("  Warning: No h3_08 column and no lat/lng columns — "
                      "community mapping unavailable")
                df['h3_08'] = None

        # --- Map h3_08 → community via community_map.json (H3 → community ID) ---
        # Preserve an explicitly supplied mapping. The historical default is
        # resolved relative to output_dir for backward compatibility.
        if community_map_path == 'community_map.json':
            community_map_path = os.path.join(output_dir, community_map_path)
        self.community_map_path = community_map_path
        self.neighbor_cells_path = os.path.join(
            output_dir, 'h3_l8_neighbor_cells.json'
        )
        self.local_feature_cache_dir = os.path.join(
            output_dir, 'cache', 'local_market_features'
        )
        if 'h3_08' in df.columns and df['h3_08'].notna().any() and os.path.exists(community_map_path):
            print(f"  Loading community map from {community_map_path}...")
            with open(community_map_path, 'r') as f:
                community_map = json.load(f)

            df['community'] = df['h3_08'].map(community_map)
            mapped = df['community'].notna().sum()
            print(f"  ✓ Mapped {mapped}/{len(df)} rows to a community "
                  f"({len(df) - mapped} unmapped will receive unknown index)")
        else:
            if not os.path.exists(community_map_path):
                print(f"  Warning: {community_map_path} not found — 'community' column not set")
            df['community'] = None
            community_map = {}

        # --- Load / compute neighbor map (community_neighbors) ---
        h3_neighbor_map = None
        h3_neighbor_cells = None

        if 'h3_08' in df.columns and df['h3_08'].notna().any():
            try:
                h3_neighbor_map, n_communities = ensure_neighbor_communities(
                    df,
                    community_map_path=community_map_path,
                    output_dir=output_dir
                )
                self.n_communities = n_communities
                h3_neighbor_cells = ensure_neighbor_cells(df, output_dir=output_dir)
                self.neighbor_cells_map = h3_neighbor_cells
                print(f"  ✓ Neighbor mapping ready: {self.n_communities} communities "
                      f"(unknown index = {self.n_communities})")
            except Exception as e:
                print(f"  Warning: Could not compute H3 L8 mappings: {e}")
                print("  Falling back — community_neighbors will be None")

        # --- Fill unmapped community values with the unknown index ---
        # n_communities is one past the last real index, matching the convention
        # used in h3_neighbor_mapper and the ONNX model embedding table.
        if self.n_communities is not None:
            unknown_idx = self.n_communities
            unmapped = df['community'].isna().sum()
            if unmapped:
                df['community'] = df['community'].fillna(unknown_idx).astype(int)
                print(f"  Assigned unknown index ({unknown_idx}) to {unmapped} unmapped rows")
            else:
                df['community'] = df['community'].astype(int)

        if h3_neighbor_map is not None:
            df['community_neighbors'] = df['h3_08'].map(h3_neighbor_map)
            df['h3_neighbor_cells'] = df['h3_08'].map(h3_neighbor_cells)
            missing = df['community_neighbors'].isna().sum()
            if missing:
                print(f"  Warning: {missing} rows have no H3 L8 neighbor mapping "
                      f"(will use unknown index {self.n_communities})")
            print(f"  ✓ Mapped {len(df) - missing}/{len(df)} rows to H3 L8 neighbourhoods")
        else:
            df['community_neighbors'] = None
            df['h3_neighbor_cells'] = None
            print("  ✗ No neighbourhood mapping available")

        # Stash the enriched dataframe so _prepare_data can use it directly
        self.dataframe = df
        return self
    
    def _add_market_indicators(self, df, market_indicator_cache_path=None):
        """Join a frozen indicator table without performing network access.

        Refreshing FRED data is intentionally handled by
        ``fetch_market_indicators.py --refetch``. Training therefore consumes an
        explicit, reproducible cache and fails clearly when it is unavailable.
        """
        if market_indicator_cache_path is None:
            market_indicator_cache_path = DEFAULT_MARKET_INDICATOR_CACHE
        if not os.path.isfile(market_indicator_cache_path):
            raise FileNotFoundError(
                "Frozen market-indicator cache not found: "
                f"{market_indicator_cache_path}. Run "
                "`python3 fetch_market_indicators.py --refetch` explicitly."
            )
        print(f"  Loading frozen indicators from {market_indicator_cache_path}...")
        indicator_frame = pd.read_csv(
            market_indicator_cache_path, index_col='date', parse_dates=True
        ).sort_index()
        joined = join_indicators_backward_asof(
            df,
            indicator_frame,
            require_complete=True,
        )
        for col in ['mortgage_rate', 'unemployment_rate']:
            if col in joined.columns:
                df[col] = joined[col].to_numpy()

    def _prepare_data(
        self,
        market_indicator_cache_path=None,
        local_feature_cache_dir=None,
        force_local_feature_recompute=False,
        water_geojson_path=DEFAULT_WATER_PATH,
        water_feature_cache_dir=None,
    ):
        """
        Data preparation: add market indicators, cleaning, feature engineering, and vocab creation.

        Community mapping (community_neighbors, n_communities) must already be
        set — call _map_communities() first, or pass a dataframe that already
        has a 'community_neighbors' column with 0-based indices.

        Parameters:
            market_indicator_cache_path: Optional path to a cached indicator CSV.
            local_feature_cache_dir: Directory for content-addressed local
                feature caches. Defaults to data/cache/local_market_features
                (relative to the community mapping output directory). Pass
                False to disable caching.
            force_local_feature_recompute: Ignore a valid cache and overwrite it.
        """

        df = self.dataframe

        # Parse dates before the causal as-of join. Row order is immaterial to
        # the join and is normalized chronologically immediately afterwards.
        df['sale_date'] = pd.to_datetime(df['sale_date'], errors='coerce')
        self._add_market_indicators(df, market_indicator_cache_path=market_indicator_cache_path)

        # --- Data Cleaning & Filtering ---
        df = df.sort_values('sale_date')

        df['sale_nbr'] = pd.to_numeric(df['sale_nbr'], errors='coerce')

        df = df.dropna(subset=['sale_price', 'lat', 'lng', 'sqft', 'sale_nbr', 'sale_date', 'sqft_lot'])
        df = df[df['sale_price'] > 0]
        df = df[df['sqft'] > 0]
        df = df[df['sale_nbr'] > 0]
        df = df[df['sqft_lot'] > 0]

        if water_feature_cache_dir is None:
            water_feature_cache_dir = os.path.join('data', 'cache', 'water_features')
        df = add_water_proximity_features(
            df,
            water_path=water_geojson_path,
            cache_dir=water_feature_cache_dir,
        )
        print(
            "Water proximity features: "
            f"{int((df['distance_to_water_m'] <= 500).sum()):,}/{len(df):,} "
            "sales within 500 m"
        )

        # Store reference date for continuous time features
        if self.reference_date is None:
            self.reference_date = df['sale_date'].min()

        # Core model features only. Dates are represented continuously so the
        # model sees adjacent dates as adjacent and December joins smoothly to
        # January instead of learning unrelated categorical year/week vectors.
        df['log_price'] = np.log(df['sale_price'])

        # --- Continuous Time Features ---
        df['time_trend'] = (
            (df['sale_date'] - self.reference_date).dt.days / 365.25
        )
        day_index = df['sale_date'].dt.dayofyear.to_numpy(dtype=np.float64) - 1.0
        days_in_year = np.where(
            df['sale_date'].dt.is_leap_year.to_numpy(), 366.0, 365.0
        )
        annual_phase = 2.0 * np.pi * day_index / days_in_year
        df['annual_sin'] = np.sin(annual_phase)
        df['annual_cos'] = np.cos(annual_phase)

        # --- Dataset Dimensions ---
        self.length = df.shape[0]

        self.dataframe = df.reset_index(drop=True)

        # Each row receives a center-plus-six-neighbor market tensor calculated
        # only from strictly earlier transactions.
        if ('h3_neighbor_cells' in self.dataframe.columns and
                self.dataframe['h3_neighbor_cells'].notna().any()):
            if local_feature_cache_dir is None:
                local_feature_cache_dir = (
                    self.local_feature_cache_dir or
                    os.path.join('data', 'cache', 'local_market_features')
                )
            elif local_feature_cache_dir is False:
                local_feature_cache_dir = None
            self.local_market_features, self.local_market_snapshot = (
                build_historical_local_features(
                    self.dataframe,
                    h3_resolution=8,
                    cache_dir=local_feature_cache_dir,
                    cache_dependencies=[
                        self.community_map_path,
                        self.neighbor_cells_path,
                    ],
                    force_recompute=force_local_feature_recompute,
                )
            )
            print(
                "Local market features: "
                f"{self.local_market_features.shape} (strictly historical)"
            )

        return self




# Compatibility alias retained while callers migrate to the descriptive class name.
dataset = DatasetBuilder

def create_vocab(df, column):
    """Creates a dictionary mapping unique values in a DF column to integers."""
    ids = sorted(df[column].unique())
    vocab = {int(id): index for index, id in enumerate(ids)}
    return vocab


def vocab_replace_tensor(tensor, vocab):
    """Replaces values in a tensor with their indices from vocabulary"""
    # Get the unknown index (should be the last index)
    unknown_idx = vocab.get('unknown', len(vocab) - 1)
    
    replaced = []
    for value in tensor:
        val = int(value.item())
        # Numeric keys are canonical. The string fallback keeps older callers
        # and directly loaded JSON dictionaries safe.
        replaced.append(vocab.get(val, vocab.get(str(val), unknown_idx)))
    
    return torch.tensor(replaced, dtype=torch.int)
