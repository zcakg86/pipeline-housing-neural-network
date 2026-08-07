"""Print sampled feature-level Shapley effects for one saved GNN prediction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pricemodel.gnn_trainer import GNNBaselineTrainer


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--row", required=True, type=int, help="Zero-based row in sales_with_predictions.csv")
    parser.add_argument("--coalitions", type=int, default=512)
    parser.add_argument("--output", type=Path, help="Optional JSON destination; otherwise prints JSON")
    return parser.parse_args(argv)


def main(argv=None):
    arguments = parse_args(argv)
    trainer = GNNBaselineTrainer.load_for_explanations(arguments.model_dir)
    explanation = trainer.explain_prediction(arguments.row, coalitions=arguments.coalitions)
    payload = json.dumps(explanation.as_dict(), indent=2) + "\n"
    if arguments.output:
        arguments.output.write_text(payload, encoding="utf-8")
        print(f"Saved: {arguments.output}")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
