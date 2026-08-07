#!/usr/bin/env python3
"""Consolidate saved RentCast JSON envelopes into one de-duplicated file.

The identity deliberately represents a sale/property combination rather than
RentCast's response ID: digits from the first ten characters of ``lastSaleDate``,
``zipCode``, and ``assessorID``, concatenated and stored as a JSON integer. Later
filenames win when two source files have the same derived identity, so the output
is deterministic.

Run from the repository root:

    python3 java-app/house-price-app/data/rentcast/consolidate_rentcast_json.py
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = "rentcast_consolidated.json"


def unique_id(property_record: dict[str, Any]) -> int | None:
    """Return the digits-only sale-date/ZIP/assessor identity as an integer."""
    components = (
        str(property_record.get("lastSaleDate") or "")[:10],
        str(property_record.get("zipCode") or ""),
        str(property_record.get("assessorID") or ""),
    )
    digits = "".join(character for component in components for character in component
                     if character.isnumeric())
    return int(digits) if digits else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory", type=Path, default=Path(__file__).resolve().parent,
        help="Directory containing RentCast JSON envelopes (default: this script's directory).",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    directory = args.directory.resolve()
    output = directory / args.output
    records: dict[int, dict[str, Any]] = {}
    files_read = 0
    missing_identity = 0

    for source in sorted(directory.glob("*.json")):
        if source.resolve() == output.resolve():
            continue
        try:
            envelope = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            print(f"Skipping unreadable JSON {source.name}: {error}")
            continue
        properties = envelope.get("properties") if isinstance(envelope, dict) else None
        if not isinstance(properties, list):
            print(f"Skipping {source.name}: no top-level properties array")
            continue
        files_read += 1
        for property_record in properties:
            if not isinstance(property_record, dict):
                continue
            derived_id = unique_id(property_record)
            if derived_id is None:
                missing_identity += 1
                continue
            record = dict(property_record)
            record["uniqueId"] = derived_id
            records[derived_id] = record

    properties = list(records.values())
    envelope = {
        "requestMetadata": {
            "source": "rentcast-consolidation-script",
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "filesRead": files_read,
            "uniqueProperties": len(properties),
            "recordsMissingUniqueId": missing_identity,
            "uniqueIdDefinition": "integer(digits(lastSaleDate[:10]) + digits(zipCode) + digits(assessorID))",
        },
        "properties": properties,
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(envelope, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(f"Read {files_read} JSON file(s); wrote {len(properties)} unique properties to {output}")
    if missing_identity:
        print(f"Skipped {missing_identity} property object(s) with no derived UniqueId components")


if __name__ == "__main__":
    main()
