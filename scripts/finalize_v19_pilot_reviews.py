from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUERIES = ROOT / "data/evaluation/v19/pilot/queries_draft.csv"
DEFAULT_REVIEWS = (
    ROOT / "records/private/v19/reviewer_01_pilot_reviews.csv"
)
DEFAULT_OUTPUT = ROOT / "data/evaluation/v19/pilot/queries_reviewed.csv"
DEFAULT_RECEIPT = (
    ROOT / "data/evaluation/v19/pilot/review_freeze_receipt.json"
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV must not be empty: {path}")
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_and_merge(
    queries: list[dict[str, str]],
    reviews: list[dict[str, str]],
) -> list[dict[str, str]]:
    query_ids = [row["query_id"].strip() for row in queries]
    review_ids = [row["query_id"].strip() for row in reviews]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("query IDs must be unique")
    if len(review_ids) != len(set(review_ids)):
        raise ValueError("review query IDs must be unique")
    if set(query_ids) != set(review_ids):
        missing = sorted(set(query_ids) - set(review_ids))
        extra = sorted(set(review_ids) - set(query_ids))
        raise ValueError(f"review coverage mismatch: missing={missing}, extra={extra}")

    reviews_by_id = {row["query_id"].strip(): row for row in reviews}
    merged: list[dict[str, str]] = []
    for query in queries:
        query_id = query["query_id"].strip()
        review = reviews_by_id[query_id]
        if review["query_text"].strip() != query["query_text"].strip():
            raise ValueError(f"query text mismatch for {query_id}")
        if review["proposed_route"].strip() != query["gold_route"].strip():
            raise ValueError(f"proposed route mismatch for {query_id}")
        final_route = review["final_route"].strip()
        if not final_route:
            raise ValueError(f"empty final route for {query_id}")
        merged_row = dict(query)
        merged_row["gold_route"] = final_route
        merged_row["status"] = "human_reviewed_pilot_not_final"
        merged.append(merged_row)
    return merged


def write_csv_atomic(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--reviews", type=Path, default=DEFAULT_REVIEWS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    queries = read_csv(args.queries)
    reviews = read_csv(args.reviews)
    merged = validate_and_merge(queries, reviews)
    write_csv_atomic(args.output, merged)
    receipt = {
        "schema_version": 1,
        "study_id": "v19-local-llm-structured-intent-routing",
        "split": "human_reviewed_pilot_not_final",
        "eligible_for_final_claim": False,
        "query_count": len(merged),
        "unique_query_count": len({row["query_id"] for row in merged}),
        "route_counts": dict(Counter(row["gold_route"] for row in merged)),
        "accepted_proposal_count": sum(
            row.get("accepted_proposal", "").lower() == "true"
            for row in reviews
        ),
        "corrected_count": sum(
            row.get("accepted_proposal", "").lower() != "true"
            for row in reviews
        ),
        "source_query_sha256": sha256_file(args.queries),
        "source_review_sha256": sha256_file(args.reviews),
        "reviewed_query_sha256": sha256_file(args.output),
    }
    write_json_atomic(args.receipt, receipt)
    print(
        "V19 pilot reviews frozen: "
        f"n={len(merged)}, accepted={receipt['accepted_proposal_count']}, "
        f"corrected={receipt['corrected_count']}"
    )
    print(f"Wrote {args.output}")
    print(f"Wrote {args.receipt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
