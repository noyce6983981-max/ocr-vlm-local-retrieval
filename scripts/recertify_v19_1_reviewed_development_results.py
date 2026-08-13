"""Recertify cached V19.1 results when human review changed no contract field."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ocr_vlm_retrieval.runtime.cache import write_json_atomic  # noqa: E402
from ocr_vlm_retrieval.runtime.late_interaction import (  # noqa: E402
    positive_retrieval_summary,
)
from scripts.run_v19_downstream_retrieval_pilot import read_json  # noqa: E402
from scripts.score_v19_v18_l1_development import summarize  # noqa: E402

EVALUATION_ROOT = ROOT / "outputs/evaluation/v19_1/condition_completeness"
DEFAULT_MACHINE_ASSIGNMENTS = EVALUATION_ROOT / "development_assignments_machine.json"
DEFAULT_REVIEWED_ASSIGNMENTS = (
    EVALUATION_ROOT / "development_assignments_human_reviewed.json"
)
DEFAULT_MACHINE_BASELINE = EVALUATION_ROOT / "development_v18_l1_top3_machine_v2.json"
DEFAULT_MACHINE_COLQ = EVALUATION_ROOT / "development_colqwen2_machine_v2.json"
DEFAULT_REVIEWED_BASELINE = (
    EVALUATION_ROOT / "development_v18_l1_top3_human_reviewed.json"
)
DEFAULT_REVIEWED_COLQ = EVALUATION_ROOT / "development_colqwen2_human_reviewed.json"

CONTRACT_FIELDS = (
    "query_id",
    "family_id",
    "query",
    "query_role",
    "content_stratum",
    "gold_answerable",
    "gold_relevant_item_ids",
    "source_item_id",
    "neighbor_item_id",
    "changed_condition_kind",
    "legacy_route",
)
BASELINE_RESULT_FIELDS = (
    "candidate_item_ids",
    "scores",
    "selected_item_id",
    "selected_retrieval_rank",
    "accepted",
    "top_score",
)
COLQ_RESULT_FIELDS = ("ranking_item_ids", "scores", "relevant_rank")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--machine-assignments", type=Path, default=DEFAULT_MACHINE_ASSIGNMENTS
    )
    parser.add_argument(
        "--reviewed-assignments", type=Path, default=DEFAULT_REVIEWED_ASSIGNMENTS
    )
    parser.add_argument("--machine-baseline", type=Path, default=DEFAULT_MACHINE_BASELINE)
    parser.add_argument("--machine-colq", type=Path, default=DEFAULT_MACHINE_COLQ)
    parser.add_argument(
        "--reviewed-baseline", type=Path, default=DEFAULT_REVIEWED_BASELINE
    )
    parser.add_argument("--reviewed-colq", type=Path, default=DEFAULT_REVIEWED_COLQ)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows_by_id(payload: Mapping[str, Any], key: str) -> dict[str, dict[str, Any]]:
    rows = [dict(row) for row in payload.get(key, [])]
    indexed = {str(row["query_id"]): row for row in rows}
    if len(rows) != 48 or len(indexed) != 48:
        raise ValueError(f"expected 48 unique rows in {key}")
    return indexed


def assert_contract_equivalent(
    machine_rows: Mapping[str, Mapping[str, Any]],
    reviewed_rows: Mapping[str, Mapping[str, Any]],
) -> None:
    """Refuse reuse unless every model- or metric-affecting field is identical."""

    if machine_rows.keys() != reviewed_rows.keys():
        raise ValueError("machine and reviewed query IDs differ")
    for query_id in sorted(machine_rows):
        machine = machine_rows[query_id]
        reviewed = reviewed_rows[query_id]
        mismatches = [
            field
            for field in CONTRACT_FIELDS
            if machine.get(field) != reviewed.get(field)
        ]
        if mismatches:
            raise ValueError(
                f"review changed execution contract for {query_id}: "
                + ", ".join(mismatches)
            )
        if reviewed.get("review_status") != "human_approved_development_family":
            raise ValueError(f"query is not human approved: {query_id}")
        if not str(reviewed.get("reviewer_id", "")).strip():
            raise ValueError(f"reviewer ID missing: {query_id}")


def _positive_by_stratum(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if bool(row.get("gold_answerable")):
            grouped[str(row.get("content_stratum", "unknown"))].append(row)
    return {
        stratum: positive_retrieval_summary(group, cutoffs=(1, 3, 10, 20))
        for stratum, group in sorted(grouped.items())
    }


def _merge_results(
    reviewed_rows: Mapping[str, Mapping[str, Any]],
    cached_rows: Mapping[str, Mapping[str, Any]],
    result_fields: Sequence[str],
) -> list[dict[str, Any]]:
    if reviewed_rows.keys() != cached_rows.keys():
        raise ValueError("cached result query IDs differ from reviewed assignments")
    merged: list[dict[str, Any]] = []
    for query_id in reviewed_rows:
        cached = cached_rows[query_id]
        missing = [field for field in result_fields if field not in cached]
        if missing:
            raise ValueError(f"cached result fields missing for {query_id}: {missing}")
        merged.append(
            {
                **dict(reviewed_rows[query_id]),
                **{field: cached[field] for field in result_fields},
            }
        )
    return merged


def main() -> int:
    args = parse_args()
    machine_assignments = read_json(args.machine_assignments)
    reviewed_assignments = read_json(args.reviewed_assignments)
    if machine_assignments.get("split") != "v19_1_machine_draft_development_only":
        raise ValueError("machine assignments split is invalid")
    if reviewed_assignments.get("split") != "v19_1_human_reviewed_development_only":
        raise ValueError("reviewed assignments split is invalid")
    machine_rows = _rows_by_id(machine_assignments, "assignments")
    reviewed_rows = _rows_by_id(reviewed_assignments, "assignments")
    assert_contract_equivalent(machine_rows, reviewed_rows)

    machine_sha256 = _sha256(args.machine_assignments)
    reviewed_sha256 = _sha256(args.reviewed_assignments)
    baseline_payload = read_json(args.machine_baseline)
    colq_payload = read_json(args.machine_colq)
    for label, payload in (("baseline", baseline_payload), ("ColQwen2", colq_payload)):
        if payload.get("split") != "v19_1_machine_draft_development_only":
            raise ValueError(f"cached {label} split is invalid")
        if payload.get("source_assignment_sha256") != machine_sha256:
            raise ValueError(f"cached {label} references different machine assignments")

    baseline_cached = _rows_by_id(baseline_payload, "results")
    colq_cached = _rows_by_id(colq_payload, "results")
    baseline_results = _merge_results(
        reviewed_rows, baseline_cached, BASELINE_RESULT_FIELDS
    )
    colq_results = _merge_results(reviewed_rows, colq_cached, COLQ_RESULT_FIELDS)
    proof = {
        "mode": "exact_contract_equivalence_no_model_rerun",
        "machine_assignment_sha256": machine_sha256,
        "reviewed_assignment_sha256": reviewed_sha256,
        "equivalent_fields": list(CONTRACT_FIELDS),
        "equivalent_query_count": len(reviewed_rows),
    }
    baseline_result = {
        **{
            key: value
            for key, value in baseline_payload.items()
            if key not in {"status", "split", "source_assignment_sha256", "results"}
        },
        "status": "human_reviewed_development_diagnostic_only",
        "split": "v19_1_human_reviewed_development_only",
        "eligible_for_promotion": False,
        "holdout_opened": False,
        "source_assignment_sha256": reviewed_sha256,
        "recertification": proof,
        "summary": summarize(baseline_results),
        "results": baseline_results,
    }
    colq_result = {
        **{
            key: value
            for key, value in colq_payload.items()
            if key
            not in {
                "status",
                "split",
                "source_assignment_sha256",
                "summary",
                "by_stratum",
                "results",
            }
        },
        "status": "human_reviewed_development_diagnostic_only",
        "split": "v19_1_human_reviewed_development_only",
        "eligible_for_promotion": False,
        "holdout_opened": False,
        "source_assignment_sha256": reviewed_sha256,
        "recertification": proof,
        "summary": positive_retrieval_summary(colq_results),
        "by_stratum": _positive_by_stratum(colq_results),
        "results": colq_results,
    }
    write_json_atomic(args.reviewed_baseline, baseline_result)
    write_json_atomic(args.reviewed_colq, colq_result)
    print(
        json.dumps(
            {
                "status": "recertified_human_reviewed_development",
                "query_count": len(reviewed_rows),
                "contract_changed": False,
                "model_rerun": False,
                "baseline": baseline_result["summary"],
                "colqwen2": colq_result["summary"],
                "holdout_opened": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
