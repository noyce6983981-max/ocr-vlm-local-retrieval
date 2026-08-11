from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.v19_formal_family_review_app import (
    ROUTE_LABELS,
    family_snapshot_sha256,
    load_families,
    load_reviews,
)

DEFAULT_REVIEWS = (
    ROOT / "records/private/v19/formal/reviewer_01_family_reviews.csv"
)
DEFAULT_FROZEN_REVIEWS = (
    ROOT / "records/private/v19/formal/reviewer_01_family_reviews_frozen.csv"
)
DEFAULT_CALIBRATION_DRAFT = (
    ROOT / "data/evaluation/v19/formal/calibration_queries_draft.csv"
)
DEFAULT_HOLDOUT_DRAFT = (
    ROOT / "data/evaluation/v19/formal/holdout_queries_draft.csv"
)
DEFAULT_CALIBRATION_OUTPUT = (
    ROOT / "data/evaluation/v19/formal/calibration_queries_reviewed.csv"
)
DEFAULT_HOLDOUT_OUTPUT = (
    ROOT / "data/evaluation/v19/formal/holdout_queries_reviewed_sealed.csv"
)
DEFAULT_RECEIPT = (
    ROOT / "data/evaluation/v19/formal/primary_review_freeze_receipt.json"
)

FROZEN_REVIEW_FIELDS = (
    "family_id",
    "family_snapshot_sha256",
    "split",
    "proposed_route",
    "final_route",
    "reviewer_id",
    "reviewed_at_unix",
    "accepted_proposal",
    "acceptance_mode",
    "notes",
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


def freeze_family_reviews(
    families: list[dict[str, Any]],
    explicit_reviews: dict[str, dict[str, str]],
    *,
    reviewer_id: str,
    frozen_at_unix: float,
) -> list[dict[str, str]]:
    family_ids = {str(family["family_id"]) for family in families}
    extra = sorted(set(explicit_reviews) - family_ids)
    if extra:
        raise ValueError(f"review file contains unknown families: {extra}")

    frozen: list[dict[str, str]] = []
    for family in families:
        family_id = str(family["family_id"])
        proposed_route = str(family["proposed_route"])
        if proposed_route not in ROUTE_LABELS:
            raise ValueError(f"unsupported proposed route for {family_id}")
        snapshot = family_snapshot_sha256(family)
        explicit = explicit_reviews.get(family_id)
        if explicit is not None:
            if explicit["family_snapshot_sha256"] != snapshot:
                raise ValueError(f"family snapshot mismatch for {family_id}")
            if explicit["split"] != str(family["split"]):
                raise ValueError(f"family split mismatch for {family_id}")
            if explicit["proposed_route"] != proposed_route:
                raise ValueError(f"proposed route mismatch for {family_id}")
            final_route = explicit["final_route"]
            reviewed_at = explicit["reviewed_at_unix"]
            notes = explicit.get("notes", "")
            acceptance_mode = "explicit_saved"
        else:
            final_route = proposed_route
            reviewed_at = f"{frozen_at_unix:.6f}"
            notes = ""
            acceptance_mode = "default_accept_after_full_browse"
        if final_route not in ROUTE_LABELS:
            raise ValueError(f"unsupported final route for {family_id}")
        frozen.append(
            {
                "family_id": family_id,
                "family_snapshot_sha256": snapshot,
                "split": str(family["split"]),
                "proposed_route": proposed_route,
                "final_route": final_route,
                "reviewer_id": reviewer_id,
                "reviewed_at_unix": reviewed_at,
                "accepted_proposal": str(final_route == proposed_route).lower(),
                "acceptance_mode": acceptance_mode,
                "notes": notes,
            }
        )
    if len(frozen) != 60:
        raise ValueError(f"expected 60 frozen family reviews, got {len(frozen)}")
    return frozen


def materialize_reviewed_queries(
    draft_rows: list[dict[str, str]],
    families: list[dict[str, Any]],
    frozen_reviews: list[dict[str, str]],
    *,
    split: str,
) -> list[dict[str, str]]:
    if split not in {"calibration", "holdout"}:
        raise ValueError(f"unsupported split: {split}")
    families_by_id = {str(row["family_id"]): row for row in families}
    reviews_by_id = {row["family_id"]: row for row in frozen_reviews}
    expected_status = f"primary_reviewed_formal_{split}"
    output: list[dict[str, str]] = []
    seen_query_ids: set[str] = set()
    family_counts: Counter[str] = Counter()
    for row in draft_rows:
        family_id = row["family_id"].strip()
        query_id = row["query_id"].strip()
        if query_id in seen_query_ids:
            raise ValueError(f"duplicate query ID: {query_id}")
        seen_query_ids.add(query_id)
        family = families_by_id.get(family_id)
        review = reviews_by_id.get(family_id)
        if family is None or review is None:
            raise ValueError(f"unknown family in query draft: {family_id}")
        if str(family["split"]) != split or row["split"].strip() != split:
            raise ValueError(f"split mismatch for {query_id}")
        query_ids = [str(value) for value in family["query_ids"]]
        queries = [str(value) for value in family["queries"]]
        if query_id not in query_ids:
            raise ValueError(f"query ID is not bound to family: {query_id}")
        family_index = query_ids.index(query_id)
        if row["query_text"].strip() != queries[family_index].strip():
            raise ValueError(f"query text mismatch for {query_id}")
        if row["gold_route"].strip() != review["proposed_route"]:
            raise ValueError(f"draft route mismatch for {query_id}")
        merged = dict(row)
        merged["gold_route"] = review["final_route"]
        merged["status"] = expected_status
        output.append(merged)
        family_counts[family_id] += 1
    split_family_ids = {
        str(row["family_id"]) for row in families if str(row["split"]) == split
    }
    if len(output) != 120 or len(split_family_ids) != 30:
        raise ValueError(
            f"expected 30 families/120 queries for {split}, "
            f"got {len(split_family_ids)}/{len(output)}"
        )
    if set(family_counts) != split_family_ids or set(family_counts.values()) != {4}:
        raise ValueError(f"family coverage mismatch for {split}")
    return output


def write_csv_atomic(
    path: Path, rows: list[dict[str, str]], fieldnames: tuple[str, ...] | None = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(fieldnames or tuple(rows[0]))
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
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
    parser.add_argument("--reviews", type=Path, default=DEFAULT_REVIEWS)
    parser.add_argument("--reviewer-id", default="reviewer_01")
    parser.add_argument("--frozen-reviews", type=Path, default=DEFAULT_FROZEN_REVIEWS)
    parser.add_argument("--calibration-draft", type=Path, default=DEFAULT_CALIBRATION_DRAFT)
    parser.add_argument("--holdout-draft", type=Path, default=DEFAULT_HOLDOUT_DRAFT)
    parser.add_argument("--calibration-output", type=Path, default=DEFAULT_CALIBRATION_OUTPUT)
    parser.add_argument("--holdout-output", type=Path, default=DEFAULT_HOLDOUT_OUTPUT)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    families = load_families()
    explicit_reviews = load_reviews(args.reviews)
    frozen_at = time.time()
    frozen_reviews = freeze_family_reviews(
        families,
        explicit_reviews,
        reviewer_id=args.reviewer_id,
        frozen_at_unix=frozen_at,
    )
    calibration = materialize_reviewed_queries(
        read_csv(args.calibration_draft),
        families,
        frozen_reviews,
        split="calibration",
    )
    holdout = materialize_reviewed_queries(
        read_csv(args.holdout_draft),
        families,
        frozen_reviews,
        split="holdout",
    )
    write_csv_atomic(args.frozen_reviews, frozen_reviews, FROZEN_REVIEW_FIELDS)
    write_csv_atomic(args.calibration_output, calibration)
    write_csv_atomic(args.holdout_output, holdout)
    receipt = {
        "schema_version": 1,
        "study_id": "v19-local-llm-structured-intent-routing",
        "review_stage": "primary_complete_double_review_pending",
        "eligible_for_final_claim": False,
        "holdout_execution_authorized": False,
        "family_count": len(frozen_reviews),
        "query_count": len(calibration) + len(holdout),
        "calibration_query_count": len(calibration),
        "holdout_query_count": len(holdout),
        "explicit_saved_family_count": sum(
            row["acceptance_mode"] == "explicit_saved" for row in frozen_reviews
        ),
        "default_accepted_family_count": sum(
            row["acceptance_mode"] == "default_accept_after_full_browse"
            for row in frozen_reviews
        ),
        "accepted_proposal_family_count": sum(
            row["accepted_proposal"] == "true" for row in frozen_reviews
        ),
        "corrected_family_count": sum(
            row["accepted_proposal"] != "true" for row in frozen_reviews
        ),
        "route_counts_by_split": {
            "calibration": dict(Counter(row["gold_route"] for row in calibration)),
            "holdout": dict(Counter(row["gold_route"] for row in holdout)),
        },
        "source_family_sha256": sha256_file(
            ROOT / "data/evaluation/v19/formal/query_families_draft.jsonl"
        ),
        "source_explicit_review_sha256": sha256_file(args.reviews),
        "frozen_family_review_sha256": sha256_file(args.frozen_reviews),
        "calibration_reviewed_sha256": sha256_file(args.calibration_output),
        "holdout_reviewed_sha256": sha256_file(args.holdout_output),
    }
    write_json_atomic(args.receipt, receipt)
    print(
        "V19 primary formal review frozen: "
        f"families={receipt['family_count']}, queries={receipt['query_count']}, "
        f"explicit={receipt['explicit_saved_family_count']}, "
        f"default={receipt['default_accepted_family_count']}, "
        f"corrected={receipt['corrected_family_count']}"
    )
    print("Holdout remains sealed; 30% independent second review is pending.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
