"""Build H3 communities from leakage-safe local property and price summaries."""
import json
from pathlib import Path

import h3
import pandas as pd

from spatial.spatial_graph_detection import run_community_analysis


def prepare_sales(path):
    """Load valid arms-length sales and attach their H3 level-8 location."""
    frame = pd.read_csv(path).drop(columns=["Unnamed: 0"], errors="ignore")
    frame["sale_date"] = pd.to_datetime(frame["sale_date"])
    required = ["sale_price", "lat", "lng", "sqft", "sale_nbr", "sale_date", "sqft_lot"]
    frame = frame.dropna(subset=required)
    frame = frame[(frame["sale_price"] > 0) & (frame["sqft"] > 0) & (frame["sale_nbr"] > 0)]
    frame["price_per_sqft"] = frame["sale_price"] / frame["sqft"]
    frame["h3_08"] = [h3.latlng_to_cell(lat, lng, 8) for lat, lng in zip(frame.lat, frame.lng)]
    return frame


def main():
    """Detect communities and write the H3-to-community deployment map."""
    frame = prepare_sales("data/sales_2020_25.csv")
    # Only the chronological training partition may shape price/property
    # similarity; validation-period prices must remain unseen by the graph.
    summary_cutoff = frame["sale_date"].sort_values().iloc[int(0.7 * len(frame))]
    print(f"Community summaries use sales through {summary_cutoff.date()}")
    *_, community_map, _ = run_community_analysis(
        df=frame,
        location_var="h3_08",
        min_neighbors=2,
        max_k=6,
        max_comm_size=50,
        base_res=1,
        feature_similarity_strength=0.75,
        similarity_floor=0.10,
        summary_end_date=summary_cutoff,
        seed=42,
    )
    destination = Path("data/community_map.json")
    destination.write_text(json.dumps(community_map, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(community_map):,} community mappings to {destination}")


if __name__ == "__main__":
    main()
