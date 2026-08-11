from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.finalize_v19_formal_primary_review import (  # noqa: E402
    read_csv,
    sha256_file,
    write_csv_atomic,
    write_json_atomic,
)
from scripts.v19_formal_family_review_app import ROUTE_LABELS  # noqa: E402
from scripts.v19_second_family_review_app import load_reviews  # noqa: E402

MANIFEST = ROOT / "data/evaluation/v19/formal/second_review_manifest.json"
PRIMARY_REVIEWS = (
    ROOT / "records/private/v19/formal/reviewer_01_family_reviews_frozen.csv"
)
SECOND_REVIEWS = (
    ROOT / "records/private/v19/formal/reviewer_02_second_family_reviews.csv"
)
CALIBRATION_PRIMARY = (
    ROOT / "data/evaluation/v19/formal/calibration_queries_reviewed.csv"
)
HOLDOUT_PRIMARY = (
    ROOT / "data/evaluation/v19/formal/holdout_queries_reviewed_sealed.csv"
)
CALIBRATION_FINAL = (
    ROOT / "data/evaluation/v19/formal/calibration_queries_final.csv"
)
HOLDOUT_FINAL = (
    ROOT / "data/evaluation/v19/formal/holdout_queries_final_sealed.csv"
)
DISAGREEMENTS = (
    ROOT / "records/private/v19/formal/second_review_disagreements.csv"
)
RECEIPT = ROOT / "data/evaluation/v19/formal/double_review_receipt.json"


def cohen_kappa(primary: list[str], secondary: list[str]) -> float:
    if len(primary) != len(secondary) or not primary:
        raise ValueError("paired labels must be non-empty and equal length")
    count = len(primary)
    observed = sum(left == right for left, right in zip(primary, secondary)) / count
    primary_counts = Counter(primary)
    secondary_counts = Counter(secondary)
    expected = sum(
        (primary_counts[label] / count) * (secondary_counts[label] / count)
        for label in ROUTE_LABELS
    )
    return 1.0 if expected == 1.0 else (observed - expected) / (1.0 - expected)


def validate_double_review(
    manifest: dict[str, Any],
    primary_rows: list[dict[str, str]],
    secondary_rows: dict[str, dict[str, str]],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    selected = list(manifest.get("families", []))
    selected_ids = {str(row["family_id"]) for row in selected}
    if len(selected) != 18 or len(selected_ids) != 18:
        raise ValueError("double-review manifest must contain 18 unique families")
    if set(secondary_rows) != selected_ids:
        missing = sorted(selected_ids - set(secondary_rows))
        extra = sorted(set(secondary_rows) - selected_ids)
        raise ValueError(f"second-review coverage mismatch: missing={missing}, extra={extra}")
    primary_by_id = {row["family_id"]: row for row in primary_rows}
    if not selected_ids <= set(primary_by_id):
        raise ValueError("primary review is missing selected families")

    disagreements: list[dict[str, str]] = []
    primary_labels: list[str] = []
    secondary_labels: list[str] = []
    reviewer_ids: set[str] = set()
    for family in selected:
        family_id = str(family["family_id"])
        primary = primary_by_id[family_id]
        secondary = secondary_rows[family_id]
        if secondary["family_snapshot_sha256"] != str(
            family["family_snapshot_sha256"]
        ):
            raise ValueError(f"second-review snapshot mismatch for {family_id}")
        if secondary["split"] != str(family["split"]):
            raise ValueError(f"second-review split mismatch for {family_id}")
        primary_route = primary["final_route"]
        secondary_route = secondary["final_route"]
        if primary_route not in ROUTE_LABELS or secondary_route not in ROUTE_LABELS:
            raise ValueError(f"unsupported route in paired review for {family_id}")
        primary_labels.append(primary_route)
        secondary_labels.append(secondary_route)
        reviewer_ids.update((primary["reviewer_id"], secondary["reviewer_id"]))
        if primary["reviewer_id"] == secondary["reviewer_id"]:
            raise ValueError("primary and second reviewer IDs must be distinct")
        if primary_route != secondary_route:
            disagreements.append(
                {
                    "family_id": family_id,
                    "split": str(family["split"]),
                    "primary_route": primary_route,
                    "secondary_route": secondary_route,
                    "adjudicated_route": "",
                    "notes": "",
                }
            )
    agreement_count = sum(
        left == right for left, right in zip(primary_labels, secondary_labels)
    )
    summary = {
        "double_reviewed_family_count": len(selected),
        "double_reviewed_query_count": len(selected) * 4,
        "double_review_fraction": len(selected) / 60,
        "agreement_count": agreement_count,
        "raw_agreement": agreement_count / len(selected),
        "cohen_kappa": cohen_kappa(primary_labels, secondary_labels),
        "disagreement_count": len(disagreements),
        "reviewer_ids_distinct": len(reviewer_ids) >= 2,
        "primary_route_counts": dict(Counter(primary_labels)),
        "secondary_route_counts": dict(Counter(secondary_labels)),
    }
    return disagreements, summary


def finalize_query_status(
    rows: list[dict[str, str]], *, split: str
) -> list[dict[str, str]]:
    expected = f"primary_reviewed_formal_{split}"
    final_status = f"adjudicated_formal_{split}_30pct_double_review"
    output: list[dict[str, str]] = []
    for row in rows:
        if row["split"] != split or row["status"] != expected:
            raise ValueError(f"unexpected primary status for {row['query_id']}")
        updated = dict(row)
        updated["status"] = final_status
        output.append(updated)
    if len(output) != 120:
        raise ValueError(f"expected 120 final {split} queries")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--primary-reviews", type=Path, default=PRIMARY_REVIEWS)
    parser.add_argument("--second-reviews", type=Path, default=SECOND_REVIEWS)
    parser.add_argument("--receipt", type=Path, default=RECEIPT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    primary_rows = read_csv(args.primary_reviews)
    secondary_rows = load_reviews(args.second_reviews)
    disagreements, summary = validate_double_review(
        manifest, primary_rows, secondary_rows
    )
    disagreement_fields = (
        "family_id",
        "split",
        "primary_route",
        "secondary_route",
        "adjudicated_route",
        "notes",
    )
    if disagreements:
        write_csv_atomic(DISAGREEMENTS, disagreements, disagreement_fields)
        raise RuntimeError(
            f"{len(disagreements)} second-review disagreements require adjudication"
        )
    calibration = finalize_query_status(
        read_csv(CALIBRATION_PRIMARY), split="calibration"
    )
    holdout = finalize_query_status(read_csv(HOLDOUT_PRIMARY), split="holdout")
    write_csv_atomic(CALIBRATION_FINAL, calibration)
    write_csv_atomic(HOLDOUT_FINAL, holdout)
    receipt = {
        "schema_version": 1,
        "study_id": "v19-local-llm-structured-intent-routing",
        "status": "v19_double_review_complete_no_adjudication_needed",
        "eligible_for_final_claim": True,
        "holdout_execution_authorized": True,
        **summary,
        "manifest_sha256": sha256_file(args.manifest),
        "primary_review_sha256": sha256_file(args.primary_reviews),
        "second_review_sha256": sha256_file(args.second_reviews),
        "calibration_final_sha256": sha256_file(CALIBRATION_FINAL),
        "holdout_final_sha256": sha256_file(HOLDOUT_FINAL),
        "method_lock_selection_sha256": str(manifest["method_lock_sha256"]),
    }
    write_json_atomic(args.receipt, receipt)
    print(
        "V19 double review complete: "
        f"n={summary['double_reviewed_family_count']}, "
        f"agreement={summary['raw_agreement']:.4f}, "
        f"kappa={summary['cohen_kappa']:.4f}, disagreements=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
