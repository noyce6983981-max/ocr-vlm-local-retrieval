"""Freeze and verify the deployable V19 retrieval candidate before holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "config/studies/v19_colqwen2_literal_evidence_method_lock.json"
CONFIG = ROOT / "config/studies/v19_colqwen2_literal_evidence_development.json"
PRIVATE_DIR = ROOT / "records/private/v19/selective_intervention"
REVIEWED_QUERIES = PRIVATE_DIR / "reviewed_queries.jsonl"
REVIEW_RECEIPT = PRIVATE_DIR / "review_compile_receipt.json"
INDEX_RECEIPT = ROOT / "outputs/user_library/colqwen2_v1_index/index_receipt.json"
MODEL_CONFIG = ROOT / "models/colqwen2-v1.0-hf/config.json"
DEVELOPMENT_RESULT = (
    ROOT
    / "outputs/evaluation/v19/selective_intervention"
    / "development_colqwen2_literal_evidence_override.json"
)
LATENCY_RESULT = (
    ROOT
    / "outputs/evaluation/v19/selective_intervention"
    / "development_colqwen2_warm_latency.json"
)

LOCKED_PATHS = (
    "config/studies/v19_colqwen2_literal_evidence_development.json",
    "config/v17_method_lock.json",
    "scripts/evaluate_v19_colqwen2_development.py",
    "scripts/evaluate_v19_ocr_literal_override.py",
    "scripts/score_v19_v18_l1_development.py",
    "scripts/run_v19_retrieval_holdout_once.py",
    "scripts/v18_1_live_search.py",
    "scripts/live_search.py",
    "src/ocr_vlm_retrieval/gating/literal_evidence.py",
    "src/ocr_vlm_retrieval/gating/ocr_literals.py",
    "src/ocr_vlm_retrieval/runtime/late_interaction.py",
    "records/private/v19/selective_intervention/reviewed_queries.jsonl",
    "records/private/v19/selective_intervention/review_compile_receipt.json",
    "outputs/evaluation/v19/selective_intervention/development_colqwen2_literal_evidence_override.json",
    "outputs/evaluation/v19/selective_intervention/development_colqwen2_warm_latency.json",
    "outputs/user_library/colqwen2_v1_index/index_receipt.json",
    "outputs/user_library/manifest.jsonl",
    "models/colqwen2-v1.0-hf/config.json",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"line {line_number} must contain an object")
            rows.append(payload)
    return rows


def locked_hashes(root: Path = ROOT) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in LOCKED_PATHS:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        hashes[relative] = sha256_file(path)
    return hashes


def verify_lock(payload: Mapping[str, Any], root: Path = ROOT) -> list[str]:
    expected = payload.get("locked_files")
    if not isinstance(expected, Mapping):
        return ["method lock has no locked_files object"]
    try:
        actual = locked_hashes(root)
    except FileNotFoundError as error:
        return [f"locked file is missing: {error.filename}"]
    errors: list[str] = []
    if set(expected) != set(actual):
        errors.append("locked file path set differs")
    for relative, digest in actual.items():
        if expected.get(relative) != digest:
            errors.append(f"SHA-256 mismatch: {relative}")
    return errors


def validate_preconditions(root: Path = ROOT) -> dict[str, Any]:
    config = read_json(root / CONFIG.relative_to(ROOT))
    if config.get("status") != "frozen_before_holdout":
        raise ValueError("V19 retrieval config is not marked frozen")
    promotion = config.get("promotion_state", {})
    if promotion.get("quality_gate_passed_on_development") is not True:
        raise ValueError("development quality gate did not pass")
    if promotion.get("retrieval_latency_gate_passed_on_development") is not True:
        raise ValueError("development retrieval latency gate did not pass")
    if promotion.get("method_frozen") is not True:
        raise ValueError("config does not record method_frozen=true")

    review_receipt = read_json(root / REVIEW_RECEIPT.relative_to(ROOT))
    reviewed_path = root / REVIEWED_QUERIES.relative_to(ROOT)
    if review_receipt.get("reviewed_queries_sha256") != sha256_file(reviewed_path):
        raise ValueError("review receipt references different queries")
    rows = read_jsonl(reviewed_path)
    if len(rows) != 200 or len({str(row["query_id"]) for row in rows}) != 200:
        raise ValueError("reviewed query set must contain 200 unique queries")
    if Counter(str(row.get("split")) for row in rows) != Counter(
        {"development": 100, "holdout": 100}
    ):
        raise ValueError("reviewed query split must be 100 development / 100 holdout")
    for split in ("development", "holdout"):
        split_rows = [row for row in rows if row.get("split") == split]
        families = Counter(str(row.get("family_id")) for row in split_rows)
        if len(families) != 25 or set(families.values()) != {4}:
            raise ValueError(f"{split} must contain 25 four-query families")
        roles = Counter(str(row.get("query_role")) for row in split_rows)
        if set(roles.values()) != {25} or len(roles) != 4:
            raise ValueError(f"{split} query roles are not balanced")

    development = read_json(root / DEVELOPMENT_RESULT.relative_to(ROOT))
    if development.get("status") != "development_diagnostic_only":
        raise ValueError("development result is incomplete")
    if development.get("split") != "development_only":
        raise ValueError("candidate selection did not use development only")
    if development.get("parameters", {}).get("eligibility_input") != (
        "query_text_only"
    ):
        raise ValueError("candidate still depends on non-deployable metadata")
    checks = development.get("promotion_check", {})
    if not checks or not all(
        value is True
        for key, value in checks.items()
        if key.endswith("_passed") or key.endswith("_inferior")
    ):
        raise ValueError("not all development promotion checks passed")

    latency = read_json(root / LATENCY_RESULT.relative_to(ROOT))
    if float(latency.get("p95_ms", float("inf"))) > 100.0:
        raise ValueError("warm ColQwen2 retrieval P95 exceeds 100 ms")

    index_receipt = read_json(root / INDEX_RECEIPT.relative_to(ROOT))
    if index_receipt.get("status") != "complete":
        raise ValueError("ColQwen2 index is incomplete")
    if int(index_receipt.get("failed_item_count", -1)) != 0:
        raise ValueError("ColQwen2 index contains failed pages")
    if int(index_receipt.get("requested_item_count", 0)) != 1482:
        raise ValueError("ColQwen2 index does not contain 1,482 effective pages")
    expected_model_hash = config.get("retrieval", {}).get("model_config_sha256")
    if expected_model_hash != sha256_file(root / MODEL_CONFIG.relative_to(ROOT)):
        raise ValueError("ColQwen2 model config hash changed")
    return {
        "config": config,
        "development": development,
        "latency": latency,
        "query_count": len(rows),
        "query_sha256": sha256_file(reviewed_path),
    }


def current_git_commit(root: Path = ROOT) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return completed.stdout.strip()


def build_lock(root: Path = ROOT) -> dict[str, Any]:
    validated = validate_preconditions(root)
    development = validated["development"]
    config = validated["config"]
    return {
        "schema_version": 1,
        "study_id": "v19-colqwen2-literal-evidence-e2e",
        "status": "locked_after_development_before_holdout",
        "locked_at_utc": datetime.now(UTC).isoformat(),
        "selection_commit_sha": current_git_commit(root),
        "holdout_scored": False,
        "reviewed_query_count": validated["query_count"],
        "reviewed_query_sha256": validated["query_sha256"],
        "development_query_count": 100,
        "holdout_query_count": 100,
        "selected_method": development["method"],
        "selected_parameters": development["parameters"],
        "selected_development_metrics": development["candidate"],
        "selected_development_delta": development["delta"],
        "retrieval": config["retrieval"],
        "governance_disclosure": (
            "Method selection was committed before holdout scoring. During the "
            "pre-freeze protocol audit, holdout text became administratively "
            "visible; no candidate rule, threshold, or model may be changed after "
            "this lock, and final reporting must not call the split fully blind."
        ),
        "locked_files": locked_hashes(root),
    }


def write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.verify:
        payload = read_json(args.output)
        errors = verify_lock(payload)
        if errors:
            raise RuntimeError("; ".join(errors))
        print(f"V19 retrieval method lock verified: {len(LOCKED_PATHS)} files")
        return 0
    payload = build_lock()
    write_json_new(args.output, payload)
    print(f"V19 retrieval method locked: {len(LOCKED_PATHS)} files")
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
