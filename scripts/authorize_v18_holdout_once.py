"""Authorize one V18 holdout opening after calibration method lock."""

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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_v17_calibration_retrieval import file_sha256


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def build_authorization(
    *,
    freeze_receipt: Mapping[str, Any],
    calibration_scope: Mapping[str, Any],
    review_report: Mapping[str, Any],
    method_lock: Mapping[str, Any],
    input_hashes: Mapping[str, str],
) -> dict[str, Any]:
    if freeze_receipt.get("retrieval_executed") is not False:
        raise ValueError("Frozen V18 receipt already records retrieval")
    if freeze_receipt.get("holdout_results_opened") is not False:
        raise ValueError("Frozen V18 receipt already records holdout access")
    split_counts = freeze_receipt.get("split_counts", {})
    if split_counts != {"calibration": 80, "holdout": 80}:
        raise ValueError("Frozen V18 split counts are not 80/80")
    if calibration_scope.get("status") != "prepared":
        raise ValueError("Calibration scope is not prepared")
    if calibration_scope.get("exported_split") != "calibration":
        raise ValueError("Calibration scope did not isolate calibration")
    if calibration_scope.get("holdout_results_opened") is not False:
        raise ValueError("Calibration scope indicates holdout access")
    if review_report.get("status") != "v18_calibration_review_complete":
        raise ValueError("Calibration review is not complete")
    if review_report.get("conflict_adjudication_required") is not False:
        raise ValueError("Calibration review conflicts remain")
    if int(review_report.get("double_reviewed_query_count", 0)) < 24:
        raise ValueError("Calibration double-review coverage is below 30 percent")
    if review_report.get("reviewer_ids_distinct") is not True:
        raise ValueError("Calibration reviewer IDs are not distinct")
    if method_lock.get("status") != "v18_method_locked_holdout_not_opened":
        raise ValueError("V18 method lock is not sealed")
    if method_lock.get("holdout_results_opened") is not False:
        raise ValueError("Method lock indicates holdout access")
    if method_lock.get("holdout_retrieval_executed") is not False:
        raise ValueError("Method lock indicates holdout retrieval")
    constraints = method_lock.get("selected_constraints", {})
    if not constraints or not all(value is True for value in constraints.values()):
        raise ValueError("Selected V18 method does not satisfy locked constraints")
    expected_keys = {
        "freeze_receipt",
        "calibration_scope",
        "review_report",
        "final_judgments",
        "verification",
        "methods",
        "method_lock",
    }
    if set(input_hashes) != expected_keys:
        raise ValueError("V18 holdout authorization hashes are incomplete")
    if any(len(str(value)) != 64 for value in input_hashes.values()):
        raise ValueError("V18 holdout authorization contains an invalid hash")
    if method_lock.get("judgments_sha256") != input_hashes["final_judgments"]:
        raise ValueError("Method lock references different final judgments")
    if method_lock.get("verification_sha256") != input_hashes["verification"]:
        raise ValueError("Method lock references different verification scores")
    if method_lock.get("methods_sha256") != input_hashes["methods"]:
        raise ValueError("Method lock references a different method grid")
    return {
        "schema_version": 1,
        "status": "approved_for_one_shot_v18_holdout_opening",
        "authorized_at_utc": datetime.now(UTC).isoformat(),
        "study_id": str(freeze_receipt["study_id"]),
        "frozen_query_set_sha256": str(freeze_receipt["query_set_sha256"]),
        "holdout_query_count": 80,
        "selected_method_id": str(method_lock["selected_method_id"]),
        "selected_parameters": dict(method_lock["selected_parameters"]),
        "input_hashes": dict(input_hashes),
        "authorization_scope": (
            "One immutable V18 holdout opening and evaluation with the locked "
            "method; no threshold or method changes are authorized."
        ),
        "holdout_results_opened": False,
        "v17_artifacts_modified": False,
    }


def write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-receipt", type=Path, required=True)
    parser.add_argument("--calibration-scope", type=Path, required=True)
    parser.add_argument("--review-report", type=Path, required=True)
    parser.add_argument("--final-judgments", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--methods", type=Path, required=True)
    parser.add_argument("--method-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = {key: Path(value).resolve() for key, value in vars(args).items()}
    authorization = build_authorization(
        freeze_receipt=read_json(paths["freeze_receipt"]),
        calibration_scope=read_json(paths["calibration_scope"]),
        review_report=read_json(paths["review_report"]),
        method_lock=read_json(paths["method_lock"]),
        input_hashes={
            key: file_sha256(paths[key])
            for key in (
                "freeze_receipt",
                "calibration_scope",
                "review_report",
                "final_judgments",
                "verification",
                "methods",
                "method_lock",
            )
        },
    )
    write_json_new(paths["output"], authorization)
    print(json.dumps(authorization, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
