"""Open V19.1 holdout authoring only after the frozen method verifies."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ocr_vlm_retrieval.runtime.cache import write_json_atomic  # noqa: E402
from scripts.freeze_v19_1_method import LOCK_PATH, verify_lock  # noqa: E402
from scripts.run_v19_downstream_retrieval_pilot import read_json  # noqa: E402

DEFAULT_OUTPUT = (
    ROOT
    / "records/private/v19_1/condition_completeness"
    / "holdout_authoring_open_receipt.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=LOCK_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"holdout authoring receipt already exists: {args.output}")
    lock = read_json(args.lock)
    errors = verify_lock(lock)
    if errors:
        raise ValueError(f"method lock verification failed: {errors}")
    if lock.get("status") != "locked_after_human_reviewed_development_before_holdout":
        raise ValueError("V19.1 method is not in the pre-holdout locked state")
    if lock.get("holdout_scored") is not False:
        raise ValueError("method lock does not declare an unscored holdout")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    receipt = {
        "schema_version": 1,
        "study_id": "v19-1-condition-completeness-e2e",
        "status": "holdout_authoring_opened_after_method_lock",
        "opened_at_utc": datetime.now(timezone.utc).isoformat(),
        "method_lock_sha256": _sha256(args.lock),
        "method_lock_commit_sha": commit,
        "method_selection_commit_sha": lock["selection_commit_sha"],
        "holdout_family_count": 12,
        "allowed_actions": [
            "read selected holdout target and neighbor pages",
            "read selected holdout OCR",
            "draft and human-review holdout queries",
        ],
        "forbidden_actions_before_query_freeze": [
            "run V18 retrieval",
            "run ColQwen2 retrieval",
            "run V18 L1 verification",
            "run V19 or V19.1 decision scoring",
            "change any method-locked file or parameter",
        ],
        "retrieval_run": False,
        "candidate_scored": False,
    }
    write_json_atomic(args.output, receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
