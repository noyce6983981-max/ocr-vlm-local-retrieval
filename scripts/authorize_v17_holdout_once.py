"""Authorize one V17 holdout evaluation after structural review checks."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from ocr_vlm_retrieval.evaluation.protocol_lock import file_sha256
from scripts.evaluate_v17_locked_holdout import validate_locked_holdout_inputs

REVIEW_FLAGS = (
    "independent_reviews",
    "full_candidate_coverage",
    "blinded_to_method_outputs",
    "model_assisted_labels_used",
    "conflict_adjudication_complete",
    "adjudicator_independent_of_primary_reviewers",
)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def build_authorization(
    *,
    method_lock: Mapping[str, Any],
    method_lock_sha256: str,
    verification: Mapping[str, Any],
    judgments: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    adjudication_report: Mapping[str, Any],
    input_hashes: Mapping[str, str],
) -> dict[str, Any]:
    if method_lock.get("status") != "method_locked_holdout_sealed":
        raise ValueError("Method lock is not sealed")
    if adjudication_report.get("status") != "holdout_human_adjudication_complete":
        raise ValueError("Human adjudication is incomplete")
    expected_review_flags = {
        "independent_reviews": True,
        "full_candidate_coverage": True,
        "blinded_to_method_outputs": True,
        "model_assisted_labels_used": False,
        "conflict_adjudication_complete": True,
        "adjudicator_independent_of_primary_reviewers": True,
    }
    for flag in REVIEW_FLAGS:
        if adjudication_report.get(flag) is not expected_review_flags[flag]:
            raise ValueError(f"Review protocol flag is not satisfied: {flag}")
    if int(adjudication_report.get("primary_reviewer_count", 0)) < 2:
        raise ValueError("Two independent primary reviewers are required")
    if int(adjudication_report.get("independent_adjudicator_count", 0)) < 1:
        raise ValueError("An independent adjudicator is required")

    validation = validate_locked_holdout_inputs(
        method_lock=method_lock,
        verification=verification,
        judgments=judgments,
        baseline_rows=baseline_rows,
    )
    expected_count = int(method_lock["holdout"]["query_count"])
    if validation["query_count"] != expected_count:
        raise ValueError("Validated holdout query count does not match the lock")
    if set(input_hashes) != {"verification", "judgments", "baseline_records"}:
        raise ValueError("Authorization input hashes are incomplete")
    if any(len(str(value)) != 64 for value in input_hashes.values()):
        raise ValueError("Authorization input hashes are invalid")

    return {
        "schema_version": 1,
        "status": "approved_for_one_shot_holdout_evaluation",
        "authorized_at_utc": datetime.now(UTC).isoformat(),
        "method_lock_sha256": method_lock_sha256,
        "review_protocol": {
            "independent_reviewer_count": int(
                adjudication_report["primary_reviewer_count"]
            ),
            **expected_review_flags,
        },
        "structural_preflight": validation,
        "input_hashes": dict(input_hashes),
        "authorization_scope": (
            "Exact immutable inputs for one V17 pooled-relevance holdout run; "
            "no method selection or threshold tuning is authorized."
        ),
    }


def write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime_root = args.runtime_root.resolve()
    lock_path = PROJECT_ROOT / "config/v17_method_lock.json"
    method_lock = read_json(lock_path)
    execution = method_lock["execution"]
    verification_path = runtime_root / execution["verification"]
    judgments_path = runtime_root / execution["judgments"]
    baseline_path = runtime_root / execution["baseline_records"]
    report_path = (
        runtime_root
        / "data/evaluation/v17/human_study/holdout/adjudication_report.json"
    )
    authorization = build_authorization(
        method_lock=method_lock,
        method_lock_sha256=file_sha256(lock_path),
        verification=read_json(verification_path),
        judgments=read_jsonl(judgments_path),
        baseline_rows=read_jsonl(baseline_path),
        adjudication_report=read_json(report_path),
        input_hashes={
            "verification": file_sha256(verification_path),
            "judgments": file_sha256(judgments_path),
            "baseline_records": file_sha256(baseline_path),
        },
    )
    output = runtime_root / execution["authorization"]
    write_json_new(output, authorization)
    print(json.dumps(authorization, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
