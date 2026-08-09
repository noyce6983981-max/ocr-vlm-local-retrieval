"""Build the sealed V18 calibration-only blind relevance pool."""

from __future__ import annotations

import argparse
import hashlib
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

from ocr_vlm_retrieval.evaluation.human_evaluation import (  # noqa: E402
    blind_candidate_pool,
    pool_ranked_runs,
)
from ocr_vlm_retrieval.evaluation.judgments import (  # noqa: E402
    POOLED_RELEVANCE_TASK,
)
from ocr_vlm_retrieval.studies.query_split import (  # noqa: E402
    query_set_fingerprint,
)
from scripts.build_v17_human_pool import (  # noqa: E402
    candidate_pool_sha256,
    index_run,
    write_jsonl_atomic,
)
from scripts.run_v17_calibration_retrieval import (  # noqa: E402
    file_sha256,
    read_jsonl,
    write_json_atomic,
)
from scripts.run_v18_calibration_retrieval import RUN_METHODS  # noqa: E402

BASE_RUN_ID = "v17_quality_hybrid"


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def validate_inputs(
    queries: Sequence[Mapping[str, Any]],
    *,
    scope_receipt: Mapping[str, Any],
    retrieval_receipt: Mapping[str, Any],
    scope_receipt_path: Path,
    methods_path: Path,
    expected_count: int,
) -> dict[str, Path]:
    if len(queries) != expected_count:
        raise ValueError(f"Expected {expected_count} calibration queries")
    if any(row.get("split") != "calibration" for row in queries):
        raise ValueError("V18 human pool may contain calibration rows only")
    if any(row.get("review_status") != "human_query_approved" for row in queries):
        raise ValueError("Every V18 calibration query must be human approved")
    if scope_receipt.get("exported_split") != "calibration":
        raise ValueError("Scope receipt does not isolate calibration")
    if scope_receipt.get("holdout_results_opened") is not False:
        raise ValueError("Holdout access invariant is not satisfied")
    if retrieval_receipt.get("status") != (
        "calibration_retrieval_complete_relevance_not_yet_judged"
    ):
        raise ValueError("Calibration retrieval is not complete")
    if retrieval_receipt.get("executed_split") != "calibration":
        raise ValueError("Retrieval receipt is not calibration-only")
    if retrieval_receipt.get("holdout_retrieval_executed") is not False:
        raise ValueError("Holdout retrieval must remain unexecuted")
    if retrieval_receipt.get("holdout_results_opened") is not False:
        raise ValueError("Holdout results must remain sealed")
    query_fingerprint = query_set_fingerprint(queries)
    if scope_receipt.get("calibration_query_set_sha256") != query_fingerprint:
        raise ValueError("Scope receipt query fingerprint mismatch")
    if retrieval_receipt.get("calibration_query_set_sha256") != query_fingerprint:
        raise ValueError("Retrieval receipt query fingerprint mismatch")
    methods_sha256 = file_sha256(methods_path)
    if scope_receipt.get("methods_sha256") != methods_sha256:
        raise ValueError("Scope receipt method hash mismatch")
    if retrieval_receipt.get("methods_sha256") != methods_sha256:
        raise ValueError("Retrieval receipt method hash mismatch")
    if retrieval_receipt.get("scope_receipt_sha256") != file_sha256(
        scope_receipt_path
    ):
        raise ValueError("Retrieval receipt does not bind this scope receipt")

    run_files = retrieval_receipt.get("run_files")
    if not isinstance(run_files, dict) or set(run_files) != set(RUN_METHODS):
        raise ValueError("Retrieval receipt must contain every locked run")
    paths: dict[str, Path] = {}
    for run_id, source in run_files.items():
        if not isinstance(source, dict):
            raise ValueError(f"Invalid run receipt for {run_id}")
        path = Path(str(source.get("path", "")))
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.is_file():
            raise FileNotFoundError(f"Missing locked run: {path}")
        if source.get("sha256") != file_sha256(path):
            raise ValueError(f"Locked run hash mismatch: {run_id}")
        paths[str(run_id)] = path
    return paths


def pool_study_fingerprint(
    queries: Sequence[Mapping[str, Any]],
    runs: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    *,
    retrieval_receipt_sha256: str,
    top_per_run: int,
    pool_size: int,
    base_guaranteed_depth: int,
    all_route_guaranteed_depth: int,
) -> str:
    material = {
        "query_set_sha256": query_set_fingerprint(queries),
        "retrieval_receipt_sha256": retrieval_receipt_sha256,
        "top_per_run": top_per_run,
        "pool_size": pool_size,
        "base_run_id": BASE_RUN_ID,
        "base_guaranteed_depth": base_guaranteed_depth,
        "all_route_guaranteed_depth": all_route_guaranteed_depth,
        "ranked_item_ids": {
            run_id: {
                query_id: [
                    str(item.get("item_id", ""))
                    for item in ranking[:top_per_run]
                ]
                for query_id, ranking in sorted(indexed.items())
            }
            for run_id, indexed in sorted(runs.items())
        },
    }
    return hashlib.sha256(
        json.dumps(
            material,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def select_capped_pool(
    pooled: Sequence[Mapping[str, Any]],
    rankings: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    pool_size: int,
    base_guaranteed_depth: int,
    all_route_guaranteed_depth: int,
) -> tuple[list[dict[str, Any]], set[str]]:
    if pool_size < 1:
        raise ValueError("pool_size must be positive")
    if base_guaranteed_depth < 1 or all_route_guaranteed_depth < 1:
        raise ValueError("guaranteed depths must be positive")
    mandatory_ids = {
        str(item.get("item_id", ""))
        for item in rankings[BASE_RUN_ID][:base_guaranteed_depth]
    }
    for ranking in rankings.values():
        mandatory_ids.update(
            str(item.get("item_id", ""))
            for item in ranking[:all_route_guaranteed_depth]
        )
    mandatory_ids.discard("")
    if len(mandatory_ids) > pool_size:
        raise ValueError(
            f"Mandatory union has {len(mandatory_ids)} items; pool_size={pool_size}"
        )
    mandatory = [dict(row) for row in pooled if str(row["item_id"]) in mandatory_ids]
    fillers = [dict(row) for row in pooled if str(row["item_id"]) not in mandatory_ids]
    selected = (mandatory + fillers)[:pool_size]
    if not mandatory_ids.issubset({str(row["item_id"]) for row in selected}):
        raise AssertionError("Capped pool lost a mandatory candidate")
    return selected, mandatory_ids


def build_pools(
    queries: Sequence[Mapping[str, Any]],
    runs: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    *,
    retrieval_receipt_sha256: str,
    top_per_run: int = 20,
    pool_size: int = 20,
    base_guaranteed_depth: int = 10,
    all_route_guaranteed_depth: int = 1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    query_ids = {str(row["query_id"]) for row in queries}
    for run_id, indexed in runs.items():
        if set(indexed) != query_ids:
            raise ValueError(f"Run/query mismatch: {run_id}")
    fingerprint = pool_study_fingerprint(
        queries,
        runs,
        retrieval_receipt_sha256=retrieval_receipt_sha256,
        top_per_run=top_per_run,
        pool_size=pool_size,
        base_guaranteed_depth=base_guaranteed_depth,
        all_route_guaranteed_depth=all_route_guaranteed_depth,
    )
    audit_rows: list[dict[str, Any]] = []
    reviewer_rows: list[dict[str, Any]] = []
    for query in sorted(queries, key=lambda row: str(row["query_id"])):
        query_id = str(query["query_id"])
        rankings = {
            run_id: indexed[query_id] for run_id, indexed in runs.items()
        }
        pooled = pool_ranked_runs(rankings, top_per_run=top_per_run)
        selected, mandatory_ids = select_capped_pool(
            pooled,
            rankings,
            pool_size=pool_size,
            base_guaranteed_depth=base_guaranteed_depth,
            all_route_guaranteed_depth=all_route_guaranteed_depth,
        )
        if len(selected) != pool_size:
            raise ValueError(
                f"{query_id} produced {len(selected)} unique candidates; "
                f"expected {pool_size}"
            )
        pool_sha256 = candidate_pool_sha256(
            query_id,
            fingerprint,
            [str(row["item_id"]) for row in selected],
        )
        shared = {
            "task_id": POOLED_RELEVANCE_TASK,
            "query_id": query_id,
            "query": str(query["query"]),
            "split": str(query["split"]),
            "group_id": str(query["group_id"]),
            "query_family": str(query.get("query_family", "")),
            "route": str(query.get("route", "")),
            "pool_sha256": pool_sha256,
            "study_fingerprint": fingerprint,
        }
        audit_rows.append(
            {
                **shared,
                "top_per_run": top_per_run,
                "pool_size": pool_size,
                "base_run_id": BASE_RUN_ID,
                "base_guaranteed_depth": base_guaranteed_depth,
                "all_route_guaranteed_depth": all_route_guaranteed_depth,
                "run_ids": sorted(runs),
                "uncapped_candidate_count": len(pooled),
                "mandatory_candidate_count": len(mandatory_ids),
                "candidate_count": len(selected),
                "candidates": selected,
            }
        )
        reviewer_rows.append(
            {
                **shared,
                "candidate_count": len(selected),
                "candidates": blind_candidate_pool(
                    query_id,
                    selected,
                    study_fingerprint=fingerprint,
                ),
            }
        )
    return audit_rows, reviewer_rows, fingerprint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--scope-receipt", type=Path, required=True)
    parser.add_argument("--retrieval-receipt", type=Path, required=True)
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
    queries_path = args.queries.resolve()
    scope_receipt_path = args.scope_receipt.resolve()
    retrieval_receipt_path = args.retrieval_receipt.resolve()
    methods_path = args.methods.resolve()
    output_dir = args.output_dir.resolve()
    queries = read_jsonl(queries_path)
    scope_receipt = read_json(scope_receipt_path)
    retrieval_receipt = read_json(retrieval_receipt_path)
    run_paths = validate_inputs(
        queries,
        scope_receipt=scope_receipt,
        retrieval_receipt=retrieval_receipt,
        scope_receipt_path=scope_receipt_path,
        methods_path=methods_path,
        expected_count=args.expected_count,
    )
    runs = {run_id: index_run(path) for run_id, path in run_paths.items()}
    retrieval_receipt_sha256 = file_sha256(retrieval_receipt_path)
    audit, reviewer, fingerprint = build_pools(
        queries,
        runs,
        retrieval_receipt_sha256=retrieval_receipt_sha256,
        top_per_run=args.top_per_run,
        pool_size=args.pool_size,
        base_guaranteed_depth=args.base_guaranteed_depth,
        all_route_guaranteed_depth=args.all_route_guaranteed_depth,
    )
    audit_path = output_dir / "pool_audit.jsonl"
    reviewer_path = output_dir / "review_packets.jsonl"
    write_jsonl_atomic(audit_path, audit)
    write_jsonl_atomic(reviewer_path, reviewer)
    receipt = {
        "status": "calibration_pool_ready_for_blind_human_review",
        "executed_split": "calibration",
        "query_count": len(queries),
        "candidate_count_per_query": args.pool_size,
        "study_fingerprint": fingerprint,
        "calibration_query_set_sha256": query_set_fingerprint(queries),
        "methods_sha256": file_sha256(methods_path),
        "scope_receipt_sha256": file_sha256(scope_receipt_path),
        "retrieval_receipt_sha256": retrieval_receipt_sha256,
        "pool_audit_sha256": file_sha256(audit_path),
        "review_packets_sha256": file_sha256(reviewer_path),
        "top_per_run": args.top_per_run,
        "base_run_id": BASE_RUN_ID,
        "base_guaranteed_depth": args.base_guaranteed_depth,
        "all_route_guaranteed_depth": args.all_route_guaranteed_depth,
        "minimum_double_review_fraction": 0.3,
        "relevance_review_complete": False,
        "holdout_results_opened": False,
        "holdout_retrieval_executed": False,
        "v17_artifacts_modified": False,
    }
    receipt_path = output_dir / "pool_receipt.json"
    write_json_atomic(receipt_path, receipt)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
