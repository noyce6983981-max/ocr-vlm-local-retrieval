"""Convert a completed blind assistant review into model-review overrides."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_assistant_full_review_queues import EXPECTED_COUNTS
from scripts.taxonomy import CATEGORY_LABELS


OUTPUT_FIELDS = [
    "item_id",
    "decision",
    "revised_category",
    "review_notes",
    "reviewed_at",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def materialize(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    completed = [row for row in rows if row.get("status") == "completed"]
    split_counts = Counter(row.get("split", "") for row in completed)
    if dict(split_counts) != EXPECTED_COUNTS:
        raise ValueError(
            "Full review is incomplete or has unexpected split counts: "
            f"{dict(split_counts)}"
        )
    item_ids = [row.get("item_id", "") for row in completed]
    if len(set(item_ids)) != len(item_ids) or not all(item_ids):
        raise ValueError("Full review item IDs must be present and unique.")

    output: list[dict[str, Any]] = []
    for row in completed:
        category = row.get("assistant_category", "")
        if category not in CATEGORY_LABELS:
            raise ValueError(
                f"Unsupported category for {row['item_id']}: {category}"
            )
        output.append(
            {
                "item_id": row["item_id"],
                "decision": "reclassified",
                "revised_category": category,
                "review_notes": (
                    f"{row.get('blind_id', '')}: "
                    f"{row.get('assistant_notes', '')}"
                ).strip(),
                "reviewed_at": row.get("reviewed_at", ""),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    labels_path = (
        args.labels
        if args.labels.is_absolute()
        else PROJECT_ROOT / args.labels
    )
    output_path = (
        args.output
        if args.output.is_absolute()
        else PROJECT_ROOT / args.output
    )
    rows = materialize(read_csv(labels_path))
    write_csv(output_path, rows)
    print(f"wrote={len(rows)} path={output_path}")


if __name__ == "__main__":
    main()
