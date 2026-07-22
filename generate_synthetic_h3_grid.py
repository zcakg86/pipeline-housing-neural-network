"""Precompute the deployable H3 level-8 grid used by the Java synthetic layer."""

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import h3
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sales", default="data/sales_2020_25_with_predictions.csv"
    )
    parser.add_argument(
        "--output", default="data/synthetic_h3_l8_grid.json"
    )
    parser.add_argument(
        "--deploy-dir",
        action="append",
        default=None,
        help="May be repeated; defaults to the canonical Java resource directory.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    sales_path = Path(args.sales)
    parents = sorted(
        pd.read_csv(sales_path, usecols=["h3_08"])["h3_08"]
        .dropna()
        .astype(str)
        .unique()
    )

    cells = []
    for h3_l8 in parents:
        lat, lng = h3.cell_to_latlng(h3_l8)
        boundary = [
            [point_lng, point_lat]
            for point_lat, point_lng in h3.cell_to_boundary(h3_l8)
        ]
        boundary.append(boundary[0])
        cells.append(
            {
                "h3_l8": h3_l8,
                "lat": lat,
                "lng": lng,
                "boundary": boundary,
            }
        )

    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": str(sales_path),
        "resolution": 8,
        "cell_count": len(cells),
        "property_defaults": {"sqft": 2000, "beds": 3, "sqft_lot": 4000},
        "cells": cells,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, separators=(",", ":")))

    deploy_dirs = args.deploy_dir or [
        "java-app/house-price-app/src/main/resources/model-artifacts",
    ]
    for raw_dir in deploy_dirs:
        deploy_dir = Path(raw_dir)
        deploy_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output_path, deploy_dir / output_path.name)

    print(
        f"Saved {len(cells):,} H3 level-8 cells to {output_path}"
    )


if __name__ == "__main__":
    main()
