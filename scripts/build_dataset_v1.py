"""Combine reviewed pilot and expansion metadata into dataset_v1 files."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_INPUTS = (
    PROJECT_ROOT / "data/manifest/pilot_manifest.jsonl",
    PROJECT_ROOT / "data/manifest/expansion_manifest.jsonl",
)
QUERY_INPUTS = (
    PROJECT_ROOT / "data/evaluation/retrieval_queries.csv",
    PROJECT_ROOT / "data/evaluation/expansion_queries.csv",
)
MANIFEST_OUTPUT = PROJECT_ROOT / "data/manifest/dataset_v1_manifest.jsonl"
QUERY_OUTPUT = PROJECT_ROOT / "data/evaluation/dataset_v1_queries.csv"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def require_unique(rows: list[dict[str, Any]], key: str) -> None:
    values = [row[key] for row in rows]
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate {key} found.")


def require_reviewed(rows: list[dict[str, Any]], label: str) -> None:
    pending = [
        row for row in rows if row.get("review_status") != "已确认"
    ]
    if pending:
        raise ValueError(f"{label} contains {len(pending)} unreviewed rows.")


def main() -> None:
    manifest_rows = [
        row
        for input_path in MANIFEST_INPUTS
        for row in load_jsonl(input_path)
    ]
    query_rows = [
        row
        for input_path in QUERY_INPUTS
        for row in load_csv(input_path)
    ]

    require_unique(manifest_rows, "item_id")
    require_unique(query_rows, "query_id")
    require_reviewed(manifest_rows, "Manifest")
    require_reviewed(query_rows, "Queries")

    item_ids = {row["item_id"] for row in manifest_rows}
    missing = [
        row["query_id"]
        for row in query_rows
        if row["expected_item_id"] not in item_ids
    ]
    if missing:
        raise ValueError(f"Queries reference missing items: {missing}")

    MANIFEST_OUTPUT.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in manifest_rows
        ),
        encoding="utf-8",
    )

    fieldnames = list(query_rows[0])
    with QUERY_OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(query_rows)

    print(f"Manifest: {len(manifest_rows)} items -> {MANIFEST_OUTPUT}")
    print(f"Queries: {len(query_rows)} rows -> {QUERY_OUTPUT}")


if __name__ == "__main__":
    main()
