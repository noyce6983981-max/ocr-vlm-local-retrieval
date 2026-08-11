from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.freeze_v19_method import verify_lock  # noqa: E402

CONFIG = ROOT / "config/studies/v19_intent_routing.json"
DOUBLE_REVIEW = ROOT / "data/evaluation/v19/formal/double_review_receipt.json"
METHOD_LOCK = ROOT / "data/evaluation/v19/formal/method_lock.json"
HOLDOUT = ROOT / "data/evaluation/v19/formal/holdout_queries_final_sealed.csv"
AUTHORIZATION = ROOT / "data/evaluation/v19/formal/holdout_authorization.json"


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_authorization(
    *,
    config: dict[str, Any],
    double_review: dict[str, Any],
    method_lock: dict[str, Any],
    holdout_path: Path,
) -> dict[str, Any]:
    if config.get("status") != "formal_holdout_authorized_guard3_locked":
        raise ValueError("V19 config has not authorized the formal holdout")
    formal = config.get("formal_calibration", {})
    if formal.get("holdout_executed") is not False:
        raise ValueError("V19 config records an earlier holdout execution")
    if formal.get("holdout_execution_authorized") is not True:
        raise ValueError("V19 config does not authorize holdout execution")
    gates = formal.get("release_gates", {})
    if not gates or not all(value is True for value in gates.values()):
        raise ValueError("not all V19 calibration gates passed")
    if double_review.get("status") != (
        "v19_double_review_complete_no_adjudication_needed"
    ):
        raise ValueError("V19 independent double review is incomplete")
    if int(double_review.get("double_reviewed_family_count", 0)) < 18:
        raise ValueError("V19 double-review coverage is below 30 percent")
    if double_review.get("reviewer_ids_distinct") is not True:
        raise ValueError("V19 reviewer identities are not distinct")
    if int(double_review.get("disagreement_count", -1)) != 0:
        raise ValueError("V19 review disagreements remain")
    if method_lock.get("status") != "locked_after_formal_calibration_before_holdout":
        raise ValueError("V19 final method lock is not sealed")
    if method_lock.get("holdout_executed") is not False:
        raise ValueError("V19 method lock records an earlier holdout execution")
    lock_errors = verify_lock(method_lock)
    if lock_errors:
        raise ValueError("V19 method lock verification failed: " + "; ".join(lock_errors))
    with holdout_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 120 or len({row["query_id"] for row in rows}) != 120:
        raise ValueError("V19 holdout must contain 120 unique queries")
    if {row["status"] for row in rows} != {
        "adjudicated_formal_holdout_30pct_double_review"
    }:
        raise ValueError("V19 holdout query status is not final")
    if Counter(row["gold_route"] for row in rows) != Counter(
        {route: 20 for route in config["scope"]["route_labels"]}
    ):
        raise ValueError("V19 holdout is not route balanced")
    holdout_sha256 = sha256_file(holdout_path)
    if double_review.get("holdout_final_sha256") != holdout_sha256:
        raise ValueError("double-review receipt references different holdout data")
    return {
        "schema_version": 1,
        "status": "approved_for_one_shot_v19_holdout",
        "authorized_at_utc": datetime.now(UTC).isoformat(),
        "study_id": "v19-local-llm-structured-intent-routing",
        "holdout_query_count": 120,
        "holdout_query_sha256": holdout_sha256,
        "method_lock_sha256": sha256_file(METHOD_LOCK),
        "double_review_receipt_sha256": sha256_file(DOUBLE_REVIEW),
        "authorized_modes": ["rule_only", "llm_only", "hybrid", "human_oracle"],
        "selected_method": method_lock["selected_method"],
        "selected_model": method_lock["selected_model"],
        "selected_inference": method_lock["selected_inference"],
        "authorization_scope": (
            "One immutable 120-query V19 holdout evaluation of the locked "
            "B0/B1/B2.1/B3 method grid; no method or threshold changes allowed."
        ),
        "holdout_executed": False,
    }


def write_json_new(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=AUTHORIZATION)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_authorization(
        config=read_json(CONFIG),
        double_review=read_json(DOUBLE_REVIEW),
        method_lock=read_json(METHOD_LOCK),
        holdout_path=HOLDOUT,
    )
    write_json_new(args.output, payload)
    print("V19 one-shot holdout authorized; results remain unopened.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
