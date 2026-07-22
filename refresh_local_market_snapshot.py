"""Refresh Java local-market artifacts from historical and RentCast sales.

The trained ONNX model and scalers are intentionally untouched. The command
maintains a deduplicated RentCast transaction ledger, rebuilds local H3 market
statistics using sales strictly before ``--as-of``, and installs the refreshed
snapshot/topology JSON files into the Java artifact directories.
"""

import argparse
from datetime import date, datetime, timezone
from glob import glob
import hashlib
import json
import os
from pathlib import Path

import h3
import numpy as np
import pandas as pd

from pricemodel.h3_neighbor_mapper import ordered_k_ring
from pricemodel.local_market_features import build_historical_local_features


ROOT = Path(__file__).resolve().parent
DEFAULT_HISTORICAL = ROOT / "data" / "sales_2020_25.csv"
DEFAULT_LEDGER = ROOT / "data" / "local_market_sales_ledger.csv"
DEFAULT_DATA_DIR = ROOT / "data"
DEFAULT_RENTCAST_DIR = ROOT / "java-app" / "house-price-app" / "data" / "rentcast"
DEFAULT_DEPLOY_DIRS = [
    ROOT / "java-app" / "model-artifacts",
    ROOT / "java-app" / "house-price-app" / "src" / "main" / "resources" / "model-artifacts",
]
LEDGER_COLUMNS = [
    "source_id", "sale_date", "sale_price", "lat", "lng", "source_file",
]


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(temporary, "w") as target:
        json.dump(value, target, indent=2)
    os.replace(temporary, path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_dates(values) -> pd.Series:
    return pd.to_datetime(values, errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()


def _normalize_sales(frame: pd.DataFrame, source_file: str, historical=False) -> pd.DataFrame:
    aliases = {
        "sale_date": "sale_date" if historical else "lastSaleDate",
        "sale_price": "sale_price" if historical else "lastSalePrice",
        "lat": "lat" if historical else "latitude",
        "lng": "lng" if historical else "longitude",
    }
    missing = [source for source in aliases.values() if source not in frame.columns]
    if missing:
        raise ValueError(f"{source_file} is missing required columns: {missing}")

    normalized = pd.DataFrame({
        destination: frame[source]
        for destination, source in aliases.items()
    })
    if historical:
        normalized["source_id"] = ""
    else:
        source_ids = frame["id"] if "id" in frame.columns else pd.Series("", index=frame.index)
        normalized["source_id"] = source_ids.fillna("").astype(str)
    normalized["source_file"] = source_file
    normalized["sale_date"] = _normalized_dates(normalized["sale_date"])
    for column in ["sale_price", "lat", "lng"]:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    valid = (
        normalized["sale_date"].notna()
        & np.isfinite(normalized["sale_price"])
        & (normalized["sale_price"] > 0)
        & np.isfinite(normalized["lat"])
        & normalized["lat"].between(-90, 90)
        & np.isfinite(normalized["lng"])
        & normalized["lng"].between(-180, 180)
    )
    return normalized.loc[valid, LEDGER_COLUMNS].reset_index(drop=True)


def load_historical_sales(path: Path) -> pd.DataFrame:
    return _normalize_sales(pd.read_csv(path), str(path), historical=True)


def load_rentcast_sales(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    elif path.suffix.lower() == ".json":
        with open(path, "r") as source:
            payload = json.load(source)
        if isinstance(payload, dict):
            records = payload.get("properties", payload.get("data", payload.get("records", [])))
        elif isinstance(payload, list):
            records = payload
        else:
            records = []
        if not isinstance(records, list):
            raise ValueError(f"{path} does not contain a RentCast record list")
        frame = pd.DataFrame(records)
    else:
        raise ValueError(f"Unsupported RentCast file type: {path}")
    return _normalize_sales(frame, str(path), historical=False)


def expand_rentcast_paths(patterns) -> list[Path]:
    if not patterns:
        patterns = [str(DEFAULT_RENTCAST_DIR / "*.json"), str(DEFAULT_RENTCAST_DIR / "*.csv")]
    paths = set()
    for pattern in patterns:
        candidate = Path(pattern)
        if candidate.is_file():
            paths.add(candidate.resolve())
        else:
            paths.update(Path(match).resolve() for match in glob(pattern))
    return sorted(path for path in paths if path.suffix.lower() in {".json", ".csv"})


def deduplicate_transactions(frame: pd.DataFrame, prefer_first=True) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    result = frame.copy()
    result["sale_date"] = _normalized_dates(result["sale_date"])
    for column in ["sale_price", "lat", "lng"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.dropna(subset=["sale_date", "sale_price", "lat", "lng"])
    result["_source_key"] = np.where(
        result["source_id"].fillna("").ne(""),
        result["source_id"].astype(str) + "|" + result["sale_date"].dt.strftime("%Y-%m-%d"),
        "",
    )
    with_source_id = result["_source_key"].ne("")
    identified = result.loc[with_source_id].drop_duplicates(
        "_source_key", keep="first" if prefer_first else "last"
    )
    unidentified = result.loc[~with_source_id]
    result = pd.concat([identified, unidentified], ignore_index=True)

    # This key also catches the same transaction appearing in the county data
    # and RentCast. Five decimal coordinate precision is about one metre here.
    result["_transaction_key"] = (
        result["sale_date"].dt.strftime("%Y-%m-%d") + "|"
        + result["sale_price"].round(2).map(lambda value: f"{value:.2f}") + "|"
        + result["lat"].round(5).map(lambda value: f"{value:.5f}") + "|"
        + result["lng"].round(5).map(lambda value: f"{value:.5f}")
    )
    result = result.drop_duplicates(
        "_transaction_key", keep="first" if prefer_first else "last"
    )
    return result.drop(columns=["_source_key", "_transaction_key"]).reset_index(drop=True)


def _load_existing_ledger(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    frame = pd.read_csv(path)
    missing = set(LEDGER_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Existing ledger {path} is missing columns: {sorted(missing)}")
    frame = frame[LEDGER_COLUMNS].copy()
    frame["sale_date"] = _normalized_dates(frame["sale_date"])
    for column in ["sale_price", "lat", "lng"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["source_id"] = frame["source_id"].fillna("").astype(str)
    frame["source_file"] = frame["source_file"].fillna("").astype(str)
    return frame.dropna(subset=["sale_date", "sale_price", "lat", "lng"])


def _build_topology(sales: pd.DataFrame, community_map_path: Path, data_dir: Path):
    with open(community_map_path, "r") as source:
        community_map = json.load(source)
    if not community_map:
        raise ValueError(f"Community map is empty: {community_map_path}")
    unknown_community = max(int(value) for value in community_map.values()) + 1

    cells_path = data_dir / "h3_l8_neighbor_cells.json"
    if cells_path.exists():
        with open(cells_path, "r") as source:
            neighbor_cells = json.load(source)
    else:
        neighbor_cells = {}

    topology_centers = set(community_map).union(sales["h3_08"].dropna().unique())
    for center in sorted(topology_centers):
        neighbor_cells[center] = ordered_k_ring(center)
    neighbor_communities = {
        center: [community_map.get(cell, unknown_community) for cell in ring]
        for center, ring in neighbor_cells.items()
    }
    return neighbor_cells, neighbor_communities


def refresh_snapshot(
    historical_path: Path,
    rentcast_paths: list[Path],
    ledger_path: Path,
    data_dir: Path,
    deploy_dirs: list[Path],
    as_of_date,
    dry_run=False,
):
    as_of = pd.Timestamp(as_of_date).normalize()
    historical = load_historical_sales(historical_path)
    old_ledger = _load_existing_ledger(ledger_path)
    new_frames = [load_rentcast_sales(path) for path in rentcast_paths]
    new_rentcast = (
        pd.concat(new_frames, ignore_index=True)
        if new_frames else pd.DataFrame(columns=LEDGER_COLUMNS)
    )
    ledger_inputs = [frame for frame in [old_ledger, new_rentcast] if not frame.empty]
    ledger = deduplicate_transactions(
        pd.concat(ledger_inputs, ignore_index=True)
        if ledger_inputs else pd.DataFrame(columns=LEDGER_COLUMNS),
        prefer_first=False,
    )

    combined_before_dedup = len(historical) + len(ledger)
    combined = deduplicate_transactions(
        pd.concat([historical, ledger], ignore_index=True),
        prefer_first=True,
    )
    future_or_same_day = int((combined["sale_date"] >= as_of).sum())
    combined = combined.loc[combined["sale_date"] < as_of].copy()
    if combined.empty:
        raise ValueError(f"No valid sales exist before snapshot as-of date {as_of.date()}")

    combined["h3_08"] = [
        h3.latlng_to_cell(float(lat), float(lng), 8)
        for lat, lng in zip(combined["lat"], combined["lng"])
    ]
    neighbor_cells, neighbor_communities = _build_topology(
        combined,
        data_dir / "community_map.json",
        data_dir,
    )
    combined["h3_neighbor_cells"] = combined["h3_08"].map(neighbor_cells)
    combined["log_price"] = np.log(combined["sale_price"].to_numpy(dtype=np.float64))
    combined = combined.sort_values("sale_date", kind="stable").reset_index(drop=True)
    # The deployable state begins immediately after its newest included sale.
    # Java can then causally score any later transaction while dynamically
    # aging recency and the rolling trend to that transaction's date.
    state_as_of = combined["sale_date"].iloc[-1] + pd.Timedelta(days=1)

    _, snapshot = build_historical_local_features(
        combined,
        snapshot_as_of_date=state_as_of,
        cache_dir=None,
    )
    snapshot["refresh"] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_cutoff_date": as_of.date().isoformat(),
        "historical_file": str(historical_path),
        "historical_file_sha256": _sha256(historical_path),
        "historical_valid_rows": len(historical),
        "rentcast_ledger_rows": len(ledger),
        "combined_rows_before_cross_source_deduplication": combined_before_dedup,
        "combined_rows_in_snapshot": len(combined),
        "excluded_on_or_after_as_of_date": future_or_same_day,
        "rentcast_input_files": [str(path) for path in rentcast_paths],
    }

    summary = {
        "as_of_date": snapshot["as_of_date"],
        "latest_sale_date": snapshot["latest_sale_date"],
        "input_cutoff_date": as_of.date().isoformat(),
        "historical_rows": len(historical),
        "new_rentcast_rows_read": len(new_rentcast),
        "rentcast_ledger_rows": len(ledger),
        "snapshot_sales": len(combined),
        "snapshot_cells": len(snapshot["cells"]),
        "cross_source_duplicates_removed": combined_before_dedup - len(combined) - future_or_same_day,
        "excluded_on_or_after_as_of_date": future_or_same_day,
    }
    if dry_run:
        return summary

    _atomic_csv(ledger_path, ledger[LEDGER_COLUMNS].sort_values("sale_date"))
    canonical_artifacts = {
        "local_market_snapshot.json": snapshot,
        "h3_l8_neighbor_cells.json": neighbor_cells,
        "h3_l8_neighbor_communities.json": neighbor_communities,
    }
    for name, value in canonical_artifacts.items():
        _atomic_json(data_dir / name, value)
    for deploy_dir in deploy_dirs:
        for name, value in canonical_artifacts.items():
            _atomic_json(deploy_dir / name, value)
    return summary


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical", type=Path, default=DEFAULT_HISTORICAL)
    parser.add_argument(
        "--rentcast",
        action="append",
        help="RentCast JSON/CSV path or glob; repeat as needed. Defaults to all files in the Java RentCast data directory.",
    )
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--deploy-dir",
        action="append",
        type=Path,
        help="Artifact directory to update; repeat as needed. Defaults to both Java artifact directories.",
    )
    parser.add_argument(
        "--as-of",
        default=date.today().isoformat(),
        help="Input cutoff (YYYY-MM-DD). Only sales strictly before this date are included; default: today.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Build and validate without writing artifacts")
    return parser.parse_args()


def main():
    args = _parse_args()
    rentcast_paths = expand_rentcast_paths(args.rentcast)
    deploy_dirs = args.deploy_dir or DEFAULT_DEPLOY_DIRS
    print(f"Refreshing local market snapshot as of {args.as_of}")
    print(f"Historical sales: {args.historical}")
    print(f"RentCast files: {len(rentcast_paths)}")
    summary = refresh_snapshot(
        historical_path=args.historical,
        rentcast_paths=rentcast_paths,
        ledger_path=args.ledger,
        data_dir=args.data_dir,
        deploy_dirs=deploy_dirs,
        as_of_date=args.as_of,
        dry_run=args.dry_run,
    )
    for key, value in summary.items():
        print(f"  {key}: {value}")
    if args.dry_run:
        print("Dry run complete; no files were written.")
    else:
        print("Snapshot refresh complete. Restart the Java application to load it.")


if __name__ == "__main__":
    main()
