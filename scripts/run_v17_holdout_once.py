"""Preflight and, when fully authorized, claim the V17 holdout exactly once."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from ocr_vlm_retrieval.evaluation.protocol_lock import (
    create_one_shot_receipt,
    file_sha256,
    resolve_relative_path,
    validate_method_lock,
)

REQUIRED_REVIEW_FLAGS = (
    "independent_reviews",
    "full_candidate_coverage",
    "blinded_to_method_outputs",
    "model_assisted_labels_used",
    "conflict_adjudication_complete",
    "adjudicator_independent_of_primary_reviewers",
)
INPUT_KEYS = ("verification", "judgments", "baseline_records")


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def execution_paths(
    lock: Mapping[str, Any],
    *,
    project_root: Path,
    runtime_root: Path,
) -> dict[str, Path]:
    plan = lock.get("execution", {})
    if not isinstance(plan, Mapping):
        raise ValueError("Method lock execution plan must be an object")
    project_keys = {"evaluator_script"}
    paths: dict[str, Path] = {}
    for key in (
        "evaluator_script",
        "authorization",
        "verification",
        "judgments",
        "baseline_records",
        "output",
        "receipt",
    ):
        relative = str(plan.get(key, "")).strip()
        if not relative:
            raise ValueError(f"Method lock execution path is missing: {key}")
        root = project_root if key in project_keys else runtime_root
        paths[key] = resolve_relative_path(root, relative)
    return paths


def validate_authorization(
    authorization: Mapping[str, Any],
    *,
    method_lock_sha256: str,
    input_paths: Mapping[str, Path],
) -> list[str]:
    errors: list[str] = []
    if int(authorization.get("schema_version", 0)) != 1:
        errors.append("authorization schema_version must be 1")
    if authorization.get("status") != "approved_for_one_shot_holdout_evaluation":
        errors.append("holdout evaluation is not explicitly approved")
    if authorization.get("method_lock_sha256") != method_lock_sha256:
        errors.append("authorization references a different method lock")

    review = authorization.get("review_protocol", {})
    if not isinstance(review, Mapping):
        errors.append("authorization review_protocol must be an object")
        review = {}
    if int(review.get("independent_reviewer_count", 0)) < 2:
        errors.append("fewer than two independent primary reviewers")
    expected_flags = {
        "independent_reviews": True,
        "full_candidate_coverage": True,
        "blinded_to_method_outputs": True,
        "model_assisted_labels_used": False,
        "conflict_adjudication_complete": True,
        "adjudicator_independent_of_primary_reviewers": True,
    }
    for flag in REQUIRED_REVIEW_FLAGS:
        if review.get(flag) is not expected_flags[flag]:
            errors.append(f"review protocol requirement is not satisfied: {flag}")

    expected_hashes = authorization.get("input_hashes", {})
    if not isinstance(expected_hashes, Mapping):
        errors.append("authorization input_hashes must be an object")
        expected_hashes = {}
    for key in INPUT_KEYS:
        path = input_paths[key]
        if not path.is_file():
            errors.append(f"authorized input is missing: {key}")
            continue
        expected = str(expected_hashes.get(key, ""))
        if not expected or file_sha256(path) != expected:
            errors.append(f"authorized input hash mismatch: {key}")
    return errors


def preflight_holdout(
    *,
    method_lock_path: Path,
    project_root: Path,
    runtime_root: Path,
) -> dict[str, Any]:
    lock = read_json(method_lock_path)
    method_lock_hash = file_sha256(method_lock_path)
    lock_validation = validate_method_lock(
        lock,
        project_root=project_root,
        runtime_root=runtime_root,
        require_clean=bool(lock.get("git_tree_clean_required", True)),
    )
    paths = execution_paths(
        lock,
        project_root=project_root,
        runtime_root=runtime_root,
    )
    authorization_errors: list[str] = []
    if not paths["authorization"].is_file():
        authorization_errors.append("independent review authorization is missing")
    else:
        authorization_errors.extend(
            validate_authorization(
                read_json(paths["authorization"]),
                method_lock_sha256=method_lock_hash,
                input_paths=paths,
            )
        )
    one_shot_errors = []
    if paths["receipt"].exists():
        one_shot_errors.append("one-shot receipt already exists")
    if paths["output"].exists():
        one_shot_errors.append("final holdout output already exists")
    if not paths["evaluator_script"].is_file():
        one_shot_errors.append("locked holdout evaluator is missing")

    errors = [
        *[str(error) for error in lock_validation["errors"]],
        *authorization_errors,
        *one_shot_errors,
    ]
    return {
        "status": "ready_for_one_shot_execution" if not errors else "blocked",
        "ready_for_execution": not errors,
        "method_lock_sha256": method_lock_hash,
        "method_lock_validation": lock_validation,
        "authorization_errors": authorization_errors,
        "one_shot_errors": one_shot_errors,
        "errors": errors,
        "paths": {key: str(value) for key, value in paths.items()},
    }


def execute_holdout_once(
    *,
    method_lock_path: Path,
    project_root: Path,
    runtime_root: Path,
) -> Path:
    report = preflight_holdout(
        method_lock_path=method_lock_path,
        project_root=project_root,
        runtime_root=runtime_root,
    )
    if not report["ready_for_execution"]:
        details = "; ".join(report["errors"])
        raise RuntimeError(f"V17 holdout execution refused: {details}")

    lock = read_json(method_lock_path)
    plan = lock["execution"]
    paths = {key: Path(value) for key, value in report["paths"].items()}
    receipt = {
        "schema_version": 1,
        "status": "one_shot_execution_claimed",
        "claimed_at_utc": datetime.now(timezone.utc).isoformat(),
        "method_lock_sha256": report["method_lock_sha256"],
        "locked_method_git_commit_sha": lock["git_commit_sha"],
        "authorization_sha256": file_sha256(paths["authorization"]),
        "input_hashes": {
            key: file_sha256(paths[key]) for key in INPUT_KEYS
        },
        "output_path": str(paths["output"]),
        "rerun_policy": "forbidden_even_if_the_evaluator_fails",
    }
    create_one_shot_receipt(paths["receipt"], receipt)

    command = [
        sys.executable,
        str(paths["evaluator_script"]),
        "--method-lock",
        str(method_lock_path),
        "--verification",
        str(paths["verification"]),
        "--judgments",
        str(paths["judgments"]),
        "--baseline-records",
        str(paths["baseline_records"]),
        "--output",
        str(paths["output"]),
        "--bootstrap-repetitions",
        str(int(plan.get("bootstrap_repetitions", 10_000))),
        "--seed",
        str(int(plan.get("bootstrap_seed", 17))),
    ]
    subprocess.run(command, cwd=project_root, check=True)
    return paths["output"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method-lock",
        type=Path,
        default=PROJECT_ROOT / "config" / "v17_method_lock.json",
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--runtime-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Claim the immutable receipt and run; default is read-only preflight.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.execute:
        output = execute_holdout_once(
            method_lock_path=args.method_lock.resolve(),
            project_root=args.project_root.resolve(),
            runtime_root=args.runtime_root.resolve(),
        )
        print(output)
        return
    report = preflight_holdout(
        method_lock_path=args.method_lock.resolve(),
        project_root=args.project_root.resolve(),
        runtime_root=args.runtime_root.resolve(),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
