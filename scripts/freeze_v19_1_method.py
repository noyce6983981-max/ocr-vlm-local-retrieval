"""Freeze and verify the V19.1 method after reviewed development selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ocr_vlm_retrieval.runtime.cache import write_json_atomic  # noqa: E402
from scripts.run_v19_downstream_retrieval_pilot import read_json  # noqa: E402

LOCK_PATH = ROOT / "config/studies/v19_1_condition_completeness_method_lock.json"
PRIVATE_DIR = ROOT / "records/private/v19_1/condition_completeness"
EVALUATION_ROOT = ROOT / "outputs/evaluation/v19_1/condition_completeness"
LOCKED_PATHS = (
    "config/studies/v19_1_condition_completeness_development.json",
    "data/evaluation/v19_1_condition_completeness/protocol_draft.json",
    "scripts/evaluate_v19_colqwen2_development.py",
    "scripts/evaluate_v19_1_colqwen2_development.py",
    "scripts/evaluate_v19_1_machine_development_end_to_end.py",
    "scripts/evaluate_v19_predecessor_on_v19_1_development.py",
    "scripts/score_v19_1_v18_l1_development.py",
    "scripts/recertify_v19_1_reviewed_development_results.py",
    "scripts/freeze_v19_1_method.py",
    "src/ocr_vlm_retrieval/gating/candidate_verification.py",
    "src/ocr_vlm_retrieval/gating/ocr_literals.py",
    "src/ocr_vlm_retrieval/gating/ocr_literals_v19_1.py",
    "src/ocr_vlm_retrieval/gating/literal_evidence_v19_1.py",
    "src/ocr_vlm_retrieval/runtime/late_interaction.py",
    "records/private/v19_1/condition_completeness/source_families_draft.jsonl",
    "records/private/v19_1/condition_completeness/source_family_selection_receipt.json",
    "records/private/v19_1/condition_completeness/reviewer_01_development_query_reviews.jsonl",
    "outputs/evaluation/v19_1/condition_completeness/development_assignments_human_reviewed.json",
    "outputs/evaluation/v19_1/condition_completeness/development_v18_l1_top3_human_reviewed.json",
    "outputs/evaluation/v19_1/condition_completeness/development_colqwen2_human_reviewed.json",
    "outputs/evaluation/v19_1/condition_completeness/development_v19_predecessor_human_reviewed.json",
    "outputs/evaluation/v19_1/condition_completeness/development_v19_1_e2e_human_reviewed.json",
    "outputs/user_library/colqwen2_v1_index/index_receipt.json",
    "outputs/user_library/manifest.jsonl",
    "models/colqwen2-v1.0-hf/config.json",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def locked_hashes(root: Path = ROOT) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in LOCKED_PATHS:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        hashes[relative] = _sha256(path)
    return hashes


def verify_lock(payload: Mapping[str, Any], root: Path = ROOT) -> list[str]:
    expected = payload.get("locked_files")
    if not isinstance(expected, dict):
        return ["locked_files mapping is missing"]
    errors: list[str] = []
    for relative in LOCKED_PATHS:
        path = root / relative
        if not path.is_file():
            errors.append(f"locked file is missing: {relative}")
            continue
        if expected.get(relative) != _sha256(path):
            errors.append(f"SHA-256 mismatch: {relative}")
    extras = sorted(set(expected) - set(LOCKED_PATHS))
    errors.extend(f"unexpected locked path: {relative}" for relative in extras)
    return errors


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _validate_development() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    assignments = read_json(
        EVALUATION_ROOT / "development_assignments_human_reviewed.json"
    )
    predecessor = read_json(
        EVALUATION_ROOT / "development_v19_predecessor_human_reviewed.json"
    )
    candidate = read_json(
        EVALUATION_ROOT / "development_v19_1_e2e_human_reviewed.json"
    )
    expected_split = "v19_1_human_reviewed_development_only"
    for label, payload in (
        ("assignments", assignments),
        ("predecessor", predecessor),
        ("candidate", candidate),
    ):
        if payload.get("split") != expected_split:
            raise ValueError(f"{label} is not human-reviewed V19.1 development")
        if payload.get("holdout_opened") not in {None, False}:
            raise ValueError(f"{label} reports an opened holdout")
    if assignments.get("status") != "human_reviewed_development_only":
        raise ValueError("reviewed assignments are not complete")
    if assignments.get("family_count") != 12 or assignments.get("query_count") != 48:
        raise ValueError("reviewed development must contain 12 families and 48 queries")
    source_sha = _sha256(
        EVALUATION_ROOT / "development_assignments_human_reviewed.json"
    )
    if predecessor.get("source_assignment_sha256") != source_sha:
        raise ValueError("predecessor references different reviewed assignments")
    if candidate.get("source_assignment_sha256") != source_sha:
        raise ValueError("candidate references different reviewed assignments")

    baseline = candidate["baseline_v18"]
    selected = candidate["candidate_v19_1"]
    delta = candidate["delta_vs_v18"]
    previous = predecessor["predecessor_v19"]
    gates = {
        "e2e_gain_vs_v18": float(delta["end_to_end_accuracy"]) >= 0.05,
        "recall_at_3_gain_vs_v18": float(delta["positive_recall_at_3"]) >= 0.05,
        "far_non_inferior_vs_v18": float(delta["false_accept_rate"]) <= 0.0,
        "positive_non_inferior_vs_v19": float(
            selected["positive_selected_relevant"]
        )
        >= float(previous["positive_selected_relevant"]),
        "e2e_improved_vs_v19": float(selected["end_to_end_accuracy"])
        > float(previous["end_to_end_accuracy"]),
    }
    if not all(gates.values()):
        failed = [name for name, passed in gates.items() if not passed]
        raise ValueError(f"development promotion gates failed: {failed}")
    return assignments, {"v18": baseline, "v19": previous, "v19_1": selected}, gates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=LOCK_PATH)
    parser.add_argument("--selection-commit", default=None)
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.verify:
        payload = read_json(args.output)
        errors = verify_lock(payload)
        if errors:
            print(json.dumps({"verified": False, "errors": errors}, indent=2))
            return 1
        print(
            json.dumps(
                {
                    "verified": True,
                    "locked_file_count": len(LOCKED_PATHS),
                    "holdout_scored": payload.get("holdout_scored"),
                },
                indent=2,
            )
        )
        return 0

    assignments, metrics, gates = _validate_development()
    source_receipt = read_json(PRIVATE_DIR / "source_family_selection_receipt.json")
    if source_receipt.get("holdout_opened") is not False:
        raise ValueError("new holdout must remain unopened before method lock")
    selection_commit = str(args.selection_commit or _git_head())
    payload = {
        "schema_version": 1,
        "study_id": "v19-1-condition-completeness-e2e",
        "status": "locked_after_human_reviewed_development_before_holdout",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_commit_sha": selection_commit,
        "holdout_scored": False,
        "holdout_ocr_opened_before_lock": False,
        "reviewed_development_family_count": assignments["family_count"],
        "reviewed_development_query_count": assignments["query_count"],
        "reviewed_development_query_sha256": _sha256(
            EVALUATION_ROOT / "development_assignments_human_reviewed.json"
        ),
        "development_review_sha256": _sha256(
            PRIVATE_DIR / "reviewer_01_development_query_reviews.jsonl"
        ),
        "selected_method": "colqwen2_top3_complete_necessary_condition_evidence",
        "selected_parameters": {
            "top_k": 3,
            "ocr_min_confidence": 0.35,
            "ocr_fuzzy_threshold": 0.88,
            "eligibility_input": "query_text_only",
            "runtime_inputs": ["query_text", "candidate_ocr"],
            "forbidden_runtime_inputs": [
                "content_stratum",
                "gold_answerable",
                "gold_relevant_item_ids",
            ],
            "selection_policy": "first ColQwen2 candidate satisfying every extracted necessary condition; reject when none match",
            "incomplete_extraction_policy": "disable V19.1 override when a high-risk query marker lacks an extracted condition",
        },
        "selected_development_metrics": metrics,
        "promotion_gates": gates,
        "retrieval": {
            "model": "vidore/colqwen2-v1.0-hf",
            "method": "multi_vector_maxsim_late_interaction",
            "indexed_page_count": 1482,
            "storage_dtype": "float16",
            "embedding_dimension": 128,
            "candidate_top_k": 3,
        },
        "governance_disclosure": (
            "The method was selected using 12 human-reviewed development families. "
            "The disjoint V19.1 holdout OCR and queries were not opened before this "
            "lock. No locked rule, parameter, model, or index may change afterward."
        ),
        "locked_files": locked_hashes(),
    }
    write_json_atomic(args.output, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "selection_commit_sha": selection_commit,
                "locked_file_count": len(payload["locked_files"]),
                "promotion_gates": gates,
                "holdout_scored": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
