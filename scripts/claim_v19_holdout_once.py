from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.authorize_v19_holdout_once import (
    AUTHORIZATION,
    HOLDOUT,
    METHOD_LOCK,
    read_json,
    sha256_file,
)

CLAIM = ROOT / "data/evaluation/v19/formal/holdout_claim.json"


def build_claim(
    authorization: dict[str, Any], *, authorization_path: Path
) -> dict[str, Any]:
    if authorization.get("status") != "approved_for_one_shot_v19_holdout":
        raise ValueError("V19 holdout is not authorized")
    if authorization.get("holdout_executed") is not False:
        raise ValueError("V19 authorization records prior execution")
    if authorization.get("method_lock_sha256") != sha256_file(METHOD_LOCK):
        raise ValueError("authorization references a different method lock")
    if authorization.get("holdout_query_sha256") != sha256_file(HOLDOUT):
        raise ValueError("authorization references different holdout queries")
    return {
        "schema_version": 1,
        "status": "v19_one_shot_holdout_claimed",
        "claimed_at_utc": datetime.now(UTC).isoformat(),
        "authorization_sha256": sha256_file(authorization_path),
        "method_lock_sha256": sha256_file(METHOD_LOCK),
        "holdout_query_sha256": sha256_file(HOLDOUT),
        "holdout_query_count": 120,
        "authorized_modes": list(authorization["authorized_modes"]),
        "rerun_policy": "forbidden_after_this_claim",
        "holdout_results_opened": True,
        "holdout_execution_complete": False,
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
    parser.add_argument("--authorization", type=Path, default=AUTHORIZATION)
    parser.add_argument("--output", type=Path, default=CLAIM)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError("V19 holdout has already been claimed")
    claim = build_claim(read_json(args.authorization), authorization_path=args.authorization)
    write_json_new(args.output, claim)
    print("V19 one-shot holdout claimed; reruns are forbidden.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
