"""Run the claimed V18 holdout retrieval once, with checkpoint resume."""

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
from scripts.run_v17_calibration_retrieval import (  # noqa: E402
    file_sha256,
    read_jsonl,
    write_json_atomic,
    write_jsonl_atomic,
)
from scripts.run_v18_calibration_retrieval import (  # noqa: E402
    RUN_METHODS,
    obtain_raw_payload,
    ranking_for_v18,
)
from scripts.live_search import library_revision  # noqa: E402


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def validate_holdout_scope(
    queries: Sequence[Mapping[str, Any]],
    scope_receipt: Mapping[str, Any],
    method_lock: Mapping[str, Any],
    *,
    expected_count: int,
    query_file_sha256: str,
    method_lock_sha256: str,
    methods_sha256: str,
) -> None:
    if len(queries) != expected_count:
        raise ValueError(f"Expected {expected_count} V18 holdout queries")
    if any(row.get("split") != "holdout" for row in queries):
        raise ValueError("V18 holdout retrieval may load holdout rows only")
    if any(row.get("review_status") != "human_query_approved" for row in queries):
        raise ValueError("Every V18 holdout query must be human approved")
    if len({str(row.get("query_id", "")) for row in queries}) != len(queries):
        raise ValueError("V18 holdout query IDs must be unique")
    if scope_receipt.get("status") != "v18_holdout_exported_after_one_shot_claim":
        raise ValueError("V18 holdout scope is not claimed and exported")
    if scope_receipt.get("exported_split") != "holdout":
        raise ValueError("V18 holdout scope exported the wrong split")
    if scope_receipt.get("holdout_results_opened") is not True:
        raise ValueError("V18 holdout opening is not recorded")
    if scope_receipt.get("retrieval_executed") is not False:
        raise ValueError("V18 holdout retrieval is already recorded")
    if scope_receipt.get("v17_artifacts_modified") is not False:
        raise ValueError("V17 read-only invariant is not satisfied")
    if scope_receipt.get("holdout_query_file_sha256") != query_file_sha256:
        raise ValueError("V18 holdout query file hash mismatch")
    if scope_receipt.get("holdout_query_set_sha256") != query_set_fingerprint(
        queries
    ):
        raise ValueError("V18 holdout query fingerprint mismatch")
    if scope_receipt.get("method_lock_sha256") != method_lock_sha256:
        raise ValueError("V18 holdout scope references another method lock")
    if method_lock.get("status") != "v18_method_locked_holdout_not_opened":
        raise ValueError("V18 calibration method is not locked")
    if method_lock.get("methods_sha256") != methods_sha256:
        raise ValueError("V18 method grid hash mismatch")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--scope-receipt", type=Path, required=True)
    parser.add_argument("--method-lock", type=Path, required=True)
    parser.add_argument(
        "--methods",
        type=Path,
        default=PROJECT_ROOT / "config/studies/v18_methods.json",
    )
    parser.add_argument(
        "--attribute-policy",
        type=Path,
        default=PROJECT_ROOT / "config/v17_attribute_coverage.json",
    )
    parser.add_argument("--library-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=80)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = {
        "queries": args.queries.resolve(),
        "scope_receipt": args.scope_receipt.resolve(),
        "method_lock": args.method_lock.resolve(),
        "methods": args.methods.resolve(),
        "attribute_policy": args.attribute_policy.resolve(),
        "library_dir": args.library_dir.resolve(),
        "output_dir": args.output_dir.resolve(),
    }
    receipt_path = paths["output_dir"] / "holdout_retrieval_receipt.json"
    if receipt_path.exists():
        raise FileExistsError("V18 holdout retrieval receipt already exists")
    queries = read_jsonl(paths["queries"])
    scope_receipt = read_json(paths["scope_receipt"])
    method_lock = read_json(paths["method_lock"])
    validate_holdout_scope(
        queries,
        scope_receipt,
        method_lock,
        expected_count=args.expected_count,
        query_file_sha256=file_sha256(paths["queries"]),
        method_lock_sha256=file_sha256(paths["method_lock"]),
        methods_sha256=file_sha256(paths["methods"]),
    )

    raw_dir = paths["output_dir"] / "raw"
    payloads: dict[str, dict[str, dict[str, Any]]] = {"v16": {}, "v17": {}}
    for index, row in enumerate(queries, start=1):
        query_id = str(row["query_id"])
        print(f"[{index:02d}/{len(queries)}] {query_id}", flush=True)
        for version in ("v16", "v17"):
            raw_path = raw_dir / f"{query_id}_{version}.json"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            payloads[version][query_id] = obtain_raw_payload(
                raw_path=raw_path,
                query=str(row["query"]),
                version=version,
                library_dir=paths["library_dir"],
                policy_path=paths["attribute_policy"],
                timeout=args.timeout,
                reuse_raw=False,
                resume=args.resume,
            )

    run_dir = paths["output_dir"] / "runs"
    run_paths: dict[str, Path] = {}
    for run_id, (version, method) in RUN_METHODS.items():
        rows = [
            {
                "query_id": str(query["query_id"]),
                "ranking": ranking_for_v18(
                    payloads[version][str(query["query_id"])], method
                ),
            }
            for query in queries
        ]
        path = run_dir / f"{run_id}.jsonl"
        write_jsonl_atomic(path, rows)
        run_paths[run_id] = path

    receipt = {
        "status": "v18_holdout_retrieval_complete_relevance_not_yet_judged",
        "executed_split": "holdout",
        "holdout_query_count": len(queries),
        "holdout_query_file_sha256": file_sha256(paths["queries"]),
        "holdout_query_set_sha256": query_set_fingerprint(queries),
        "scope_receipt_sha256": file_sha256(paths["scope_receipt"]),
        "method_lock_sha256": file_sha256(paths["method_lock"]),
        "methods_sha256": file_sha256(paths["methods"]),
        "attribute_policy_sha256": file_sha256(paths["attribute_policy"]),
        "library_revision": library_revision(paths["library_dir"]),
        "selected_method_id": method_lock["selected_method_id"],
        "selected_parameters": method_lock["selected_parameters"],
        "run_files": {
            run_id: {"path": str(path), "sha256": file_sha256(path)}
            for run_id, path in sorted(run_paths.items())
        },
        "study_route_fallback_query_ids": [
            str(query["query_id"])
            for query in queries
            if payloads["v16"][str(query["query_id"])].get(
                "v18_study_route_fallback"
            )
            or payloads["v17"][str(query["query_id"])].get(
                "v18_study_route_fallback"
            )
        ],
        "relevance_judgments_read": False,
        "holdout_results_opened": True,
        "holdout_retrieval_executed": True,
        "v17_artifacts_modified": False,
    }
    write_json_atomic(receipt_path, receipt)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
