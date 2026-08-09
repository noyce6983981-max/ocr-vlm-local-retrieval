"""Build the claimed V18 holdout blind Top-20 relevance pool."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from ocr_vlm_retrieval.studies.query_split import (  # noqa: E402
    query_set_fingerprint,
)
from scripts.build_v17_human_pool import (  # noqa: E402
    index_run,
    write_jsonl_atomic,
)
from scripts.build_v18_calibration_pool import (  # noqa: E402
    BASE_RUN_ID,
    build_pools,
    read_json,
)
from scripts.run_v17_calibration_retrieval import (  # noqa: E402
    file_sha256,
    read_jsonl,
    write_json_atomic,
)
from scripts.run_v18_calibration_retrieval import RUN_METHODS  # noqa: E402


def validate_inputs(
    queries: Sequence[Mapping[str, Any]],
    *,
    scope_receipt: Mapping[str, Any],
    retrieval_receipt: Mapping[str, Any],
    method_lock: Mapping[str, Any],
    scope_receipt_path: Path,
    method_lock_path: Path,
    methods_path: Path,
    expected_count: int,
) -> dict[str, Path]:
    if len(queries) != expected_count:
        raise ValueError(f"Expected {expected_count} V18 holdout queries")
    if any(row.get("split") != "holdout" for row in queries):
        raise ValueError("V18 holdout pool may contain holdout rows only")
    if scope_receipt.get("status") != "v18_holdout_exported_after_one_shot_claim":
        raise ValueError("V18 holdout scope is not claimed")
    if scope_receipt.get("holdout_results_opened") is not True:
        raise ValueError("V18 holdout scope is not open")
    if retrieval_receipt.get("status") != (
        "v18_holdout_retrieval_complete_relevance_not_yet_judged"
    ):
        raise ValueError("V18 holdout retrieval is incomplete")
    if retrieval_receipt.get("executed_split") != "holdout":
        raise ValueError("V18 retrieval receipt is not holdout-only")
    if retrieval_receipt.get("relevance_judgments_read") is not False:
        raise ValueError("V18 holdout retrieval must not read relevance labels")
    fingerprint = query_set_fingerprint(queries)
    if scope_receipt.get("holdout_query_set_sha256") != fingerprint:
        raise ValueError("V18 holdout scope query fingerprint mismatch")
    if retrieval_receipt.get("holdout_query_set_sha256") != fingerprint:
        raise ValueError("V18 holdout retrieval query fingerprint mismatch")
    if retrieval_receipt.get("scope_receipt_sha256") != file_sha256(
        scope_receipt_path
    ):
        raise ValueError("V18 retrieval does not bind this scope receipt")
    if retrieval_receipt.get("method_lock_sha256") != file_sha256(
        method_lock_path
    ):
        raise ValueError("V18 retrieval does not bind this method lock")
    if method_lock.get("status") != "v18_method_locked_holdout_not_opened":
        raise ValueError("V18 method lock is invalid")
    methods_sha256 = file_sha256(methods_path)
    if retrieval_receipt.get("methods_sha256") != methods_sha256:
        raise ValueError("V18 retrieval method hash mismatch")
    run_files = retrieval_receipt.get("run_files")
    if not isinstance(run_files, dict) or set(run_files) != set(RUN_METHODS):
        raise ValueError("V18 holdout retrieval must contain every locked run")
    paths: dict[str, Path] = {}
    for run_id, source in run_files.items():
        if not isinstance(source, dict):
            raise ValueError(f"Invalid V18 holdout run receipt: {run_id}")
        path = Path(str(source.get("path", "")))
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.is_file() or source.get("sha256") != file_sha256(path):
            raise ValueError(f"V18 holdout run hash mismatch: {run_id}")
        paths[str(run_id)] = path
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--scope-receipt", type=Path, required=True)
    parser.add_argument("--retrieval-receipt", type=Path, required=True)
    parser.add_argument("--method-lock", type=Path, required=True)
    parser.add_argument(
        "--methods",
        type=Path,
        default=PROJECT_ROOT / "config/studies/v18_methods.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=80)
    parser.add_argument("--top-per-run", type=int, default=20)
    parser.add_argument("--pool-size", type=int, default=20)
    parser.add_argument("--base-guaranteed-depth", type=int, default=10)
    parser.add_argument("--all-route-guaranteed-depth", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = {
        "queries": args.queries.resolve(),
        "scope_receipt": args.scope_receipt.resolve(),
        "retrieval_receipt": args.retrieval_receipt.resolve(),
        "method_lock": args.method_lock.resolve(),
        "methods": args.methods.resolve(),
        "output_dir": args.output_dir.resolve(),
    }
    queries = read_jsonl(paths["queries"])
    scope_receipt = read_json(paths["scope_receipt"])
    retrieval_receipt = read_json(paths["retrieval_receipt"])
    method_lock = read_json(paths["method_lock"])
    run_paths = validate_inputs(
        queries,
        scope_receipt=scope_receipt,
        retrieval_receipt=retrieval_receipt,
        method_lock=method_lock,
        scope_receipt_path=paths["scope_receipt"],
        method_lock_path=paths["method_lock"],
        methods_path=paths["methods"],
        expected_count=args.expected_count,
    )
    runs = {run_id: index_run(path) for run_id, path in run_paths.items()}
    retrieval_sha256 = file_sha256(paths["retrieval_receipt"])
    audit, reviewer, fingerprint = build_pools(
        queries,
        runs,
        retrieval_receipt_sha256=retrieval_sha256,
        top_per_run=args.top_per_run,
        pool_size=args.pool_size,
        base_guaranteed_depth=args.base_guaranteed_depth,
        all_route_guaranteed_depth=args.all_route_guaranteed_depth,
    )
    audit_path = paths["output_dir"] / "pool_audit.jsonl"
    reviewer_path = paths["output_dir"] / "review_packets.jsonl"
    receipt_path = paths["output_dir"] / "pool_receipt.json"
    if receipt_path.exists():
        raise FileExistsError("V18 holdout pool receipt already exists")
    write_jsonl_atomic(audit_path, audit)
    write_jsonl_atomic(reviewer_path, reviewer)
    receipt = {
        "status": "v18_holdout_pool_ready_for_blind_human_review",
        "executed_split": "holdout",
        "query_count": len(queries),
        "candidate_count_per_query": args.pool_size,
        "study_fingerprint": fingerprint,
        "holdout_query_set_sha256": query_set_fingerprint(queries),
        "methods_sha256": file_sha256(paths["methods"]),
        "method_lock_sha256": file_sha256(paths["method_lock"]),
        "scope_receipt_sha256": file_sha256(paths["scope_receipt"]),
        "retrieval_receipt_sha256": retrieval_sha256,
        "pool_audit_sha256": file_sha256(audit_path),
        "review_packets_sha256": file_sha256(reviewer_path),
        "top_per_run": args.top_per_run,
        "base_run_id": BASE_RUN_ID,
        "base_guaranteed_depth": args.base_guaranteed_depth,
        "all_route_guaranteed_depth": args.all_route_guaranteed_depth,
        "minimum_double_review_fraction": 0.3,
        "relevance_review_complete": False,
        "holdout_results_opened": True,
        "holdout_retrieval_executed": True,
        "v17_artifacts_modified": False,
    }
    write_json_atomic(receipt_path, receipt)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
