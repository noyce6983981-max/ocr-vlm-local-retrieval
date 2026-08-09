"""Claim the V18 one-shot holdout and export it only after the claim exists."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from ocr_vlm_retrieval.studies.query_split import (  # noqa: E402
    query_set_fingerprint,
    validate_frozen_queries,
)
from scripts.authorize_v18_holdout_once import (  # noqa: E402
    read_json,
    write_json_new,
)
from scripts.prepare_v18_calibration_scope import (  # noqa: E402
    read_jsonl,
    write_jsonl_new,
)
from scripts.run_v17_calibration_retrieval import file_sha256  # noqa: E402


def validate_claim_preflight(
    authorization: Mapping[str, Any],
    method_lock: Mapping[str, Any],
    freeze_receipt: Mapping[str, Any],
    *,
    authorization_sha256: str,
    method_lock_sha256: str,
    freeze_receipt_sha256: str,
) -> dict[str, Any]:
    if authorization.get("status") != "approved_for_one_shot_v18_holdout_opening":
        raise ValueError("V18 holdout opening is not authorized")
    hashes = authorization.get("input_hashes", {})
    if not isinstance(hashes, Mapping):
        raise ValueError("Authorization input hashes are missing")
    if hashes.get("method_lock") != method_lock_sha256:
        raise ValueError("Authorization references a different method lock")
    if hashes.get("freeze_receipt") != freeze_receipt_sha256:
        raise ValueError("Authorization references a different freeze receipt")
    if method_lock.get("status") != "v18_method_locked_holdout_not_opened":
        raise ValueError("V18 method lock is not sealed")
    if freeze_receipt.get("holdout_results_opened") is not False:
        raise ValueError("Freeze receipt already records holdout access")
    if authorization.get("frozen_query_set_sha256") != freeze_receipt.get(
        "query_set_sha256"
    ):
        raise ValueError("Authorization references a different frozen query set")
    return {
        "schema_version": 1,
        "status": "v18_one_shot_holdout_claimed",
        "claimed_at_utc": datetime.now(UTC).isoformat(),
        "authorization_sha256": authorization_sha256,
        "method_lock_sha256": method_lock_sha256,
        "freeze_receipt_sha256": freeze_receipt_sha256,
        "frozen_query_set_sha256": str(freeze_receipt["query_set_sha256"]),
        "selected_method_id": str(method_lock["selected_method_id"]),
        "selected_parameters": dict(method_lock["selected_parameters"]),
        "rerun_policy": "forbidden_after_this_claim",
        "holdout_results_opened": True,
        "v17_artifacts_modified": False,
    }


def isolate_holdout_rows(
    rows: Sequence[Mapping[str, Any]],
    freeze_receipt: Mapping[str, Any],
    *,
    expected_total: int = 160,
    expected_holdout: int = 80,
) -> list[dict[str, Any]]:
    validate_frozen_queries(rows, expected_count=expected_total)
    if query_set_fingerprint(rows) != freeze_receipt.get("query_set_sha256"):
        raise ValueError("Frozen V18 query fingerprint does not match its receipt")
    split_counts = Counter(str(row.get("split", "")) for row in rows)
    if split_counts != Counter({"calibration": 80, "holdout": expected_holdout}):
        raise ValueError("Frozen V18 split counts are invalid")
    if any(row.get("review_status") != "human_query_approved" for row in rows):
        raise ValueError("Every V18 query must be human approved")
    holdout = [dict(row) for row in rows if row.get("split") == "holdout"]
    holdout.sort(key=lambda row: str(row["query_id"]))
    return holdout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--method-lock", type=Path, required=True)
    parser.add_argument("--freeze-receipt", type=Path, required=True)
    parser.add_argument("--frozen-queries", type=Path, required=True)
    parser.add_argument("--claim-receipt", type=Path, required=True)
    parser.add_argument("--output-queries", type=Path, required=True)
    parser.add_argument("--output-scope-receipt", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = {key: Path(value).resolve() for key, value in vars(args).items()}
    for key in ("claim_receipt", "output_queries", "output_scope_receipt"):
        if paths[key].exists():
            raise FileExistsError(f"One-shot V18 output already exists: {paths[key]}")
    authorization = read_json(paths["authorization"])
    method_lock = read_json(paths["method_lock"])
    freeze_receipt = read_json(paths["freeze_receipt"])
    claim = validate_claim_preflight(
        authorization,
        method_lock,
        freeze_receipt,
        authorization_sha256=file_sha256(paths["authorization"]),
        method_lock_sha256=file_sha256(paths["method_lock"]),
        freeze_receipt_sha256=file_sha256(paths["freeze_receipt"]),
    )
    write_json_new(paths["claim_receipt"], claim)

    # This is the first authorized read of the sealed holdout query rows.
    holdout = isolate_holdout_rows(read_jsonl(paths["frozen_queries"]), freeze_receipt)
    write_jsonl_new(paths["output_queries"], holdout)
    scope = {
        **claim,
        "status": "v18_holdout_exported_after_one_shot_claim",
        "exported_split": "holdout",
        "holdout_query_count": len(holdout),
        "holdout_query_set_sha256": query_set_fingerprint(holdout),
        "holdout_query_file_sha256": file_sha256(paths["output_queries"]),
        "claim_receipt_sha256": file_sha256(paths["claim_receipt"]),
        "retrieval_executed": False,
    }
    write_json_new(paths["output_scope_receipt"], scope)
    print(json.dumps(scope, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
