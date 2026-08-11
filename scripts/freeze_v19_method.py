from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data/evaluation/v19/formal/method_lock.json"
LOCKED_PATHS = (
    "config/studies/v19_intent_routing.json",
    "data/evaluation/v19/model_pilot/model_manifest.json",
    "data/evaluation/v19/pilot/queries_reviewed.csv",
    "data/evaluation/v19/pilot/review_freeze_receipt.json",
    "data/evaluation/v19/formal/primary_review_freeze_receipt.json",
    "data/evaluation/v19/formal/calibration_queries_reviewed.csv",
    "data/evaluation/v19/formal/holdout_queries_reviewed_sealed.csv",
    "data/evaluation/v19/formal/second_review_manifest.json",
    "data/evaluation/v19/formal/double_review_receipt.json",
    "data/evaluation/v19/formal/calibration_queries_final.csv",
    "data/evaluation/v19/formal/holdout_queries_final_sealed.csv",
    "scripts/run_intent_routing_study.py",
    "scripts/authorize_v19_holdout_once.py",
    "scripts/claim_v19_holdout_once.py",
    "src/ocr_vlm_retrieval/routing/arbitration.py",
    "src/ocr_vlm_retrieval/routing/hybrid_router.py",
    "src/ocr_vlm_retrieval/routing/llm_router.py",
    "src/ocr_vlm_retrieval/routing/mapping.py",
    "src/ocr_vlm_retrieval/routing/prompt.py",
    "src/ocr_vlm_retrieval/routing/rule_router.py",
    "src/ocr_vlm_retrieval/routing/schema.py",
    "src/ocr_vlm_retrieval/routing/transformers_backend.py",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def locked_hashes(root: Path = ROOT) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in LOCKED_PATHS:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        result[relative] = sha256_file(path)
    return result


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


def verify_lock(payload: dict[str, Any], root: Path = ROOT) -> list[str]:
    expected = payload.get("locked_files")
    if not isinstance(expected, dict):
        return ["method lock has no locked_files object"]
    actual = locked_hashes(root)
    errors: list[str] = []
    if set(expected) != set(actual):
        errors.append("locked file path set differs")
    for relative, digest in actual.items():
        if expected.get(relative) != digest:
            errors.append(f"SHA-256 mismatch: {relative}")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.verify:
        payload = json.loads(args.output.read_text(encoding="utf-8"))
        errors = verify_lock(payload)
        if errors:
            raise RuntimeError("; ".join(errors))
        print(f"V19 method lock verified: {len(payload['locked_files'])} files")
        return 0

    config = json.loads(
        (ROOT / "config/studies/v19_intent_routing.json").read_text(
            encoding="utf-8"
        )
    )
    payload = {
        "schema_version": 1,
        "study_id": "v19-local-llm-structured-intent-routing",
        "status": "locked_after_formal_calibration_before_holdout",
        "created_date": "2026-08-11",
        "holdout_executed": False,
        "selected_method": config["pilot_observation"]["selected_method"],
        "selected_model": config["pilot_observation"]["selected_model"],
        "selected_inference": config["pilot_observation"]["selected_inference"],
        "locked_files": locked_hashes(),
    }
    write_json_atomic(args.output, payload)
    print(f"V19 method locked: {len(payload['locked_files'])} files")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
