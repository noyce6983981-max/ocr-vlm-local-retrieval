"""Score locked V18 holdout Top-K candidates without reading judgments."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_REPO = PROJECT_ROOT / "third_party/Qwen3-VL-Embedding"
PACKAGE_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from ocr_vlm_retrieval.gating.candidate_verification import (  # noqa: E402
    FULL_QUERY_INSTRUCTION,
)
from ocr_vlm_retrieval.studies.query_split import (  # noqa: E402
    query_set_fingerprint,
)
from scripts.run_v17_calibration_retrieval import (  # noqa: E402
    file_sha256,
    read_jsonl,
    write_json_atomic,
)
from scripts.score_v17_candidate_verification import resume_results  # noqa: E402
from scripts.score_v18_candidate_verification import (  # noqa: E402
    acquire_output_lock,
    build_verification_tasks,
    keyed_rows,
    read_json,
    release_output_lock,
)

BASE_RUN_ID = "v17_quality_hybrid"


def validate_scope(
    queries: Sequence[Mapping[str, Any]],
    pool_rows: Sequence[Mapping[str, Any]],
    ranking_rows: Sequence[Mapping[str, Any]],
    *,
    scope_receipt: Mapping[str, Any],
    pool_receipt: Mapping[str, Any],
    retrieval_receipt: Mapping[str, Any],
    method_lock: Mapping[str, Any],
    scope_receipt_path: Path,
    pool_path: Path,
    pool_receipt_path: Path,
    retrieval_receipt_path: Path,
    method_lock_path: Path,
    ranking_path: Path,
    expected_count: int,
) -> int:
    if len(queries) != expected_count:
        raise ValueError(f"Expected {expected_count} V18 holdout queries")
    if any(row.get("split") != "holdout" for row in queries):
        raise ValueError("V18 holdout verification may load holdout queries only")
    if scope_receipt.get("status") != "v18_holdout_exported_after_one_shot_claim":
        raise ValueError("V18 holdout scope is not claimed")
    if pool_receipt.get("status") != "v18_holdout_pool_ready_for_blind_human_review":
        raise ValueError("V18 holdout pool is not ready")
    if retrieval_receipt.get("status") != (
        "v18_holdout_retrieval_complete_relevance_not_yet_judged"
    ):
        raise ValueError("V18 holdout retrieval is incomplete")
    if method_lock.get("status") != "v18_method_locked_holdout_not_opened":
        raise ValueError("V18 method lock is invalid")
    if method_lock.get("selected_method_id") not in {"L0", "L1", "L2"}:
        raise ValueError("Locked V18 holdout method needs unsupported clause scores")
    top_k = int(method_lock.get("selected_parameters", {}).get("top_k", 0))
    if top_k < 1:
        raise ValueError("Locked V18 holdout top_k is invalid")
    fingerprint = query_set_fingerprint(queries)
    if scope_receipt.get("holdout_query_set_sha256") != fingerprint:
        raise ValueError("Holdout scope query fingerprint mismatch")
    if pool_receipt.get("holdout_query_set_sha256") != fingerprint:
        raise ValueError("Holdout pool query fingerprint mismatch")
    if pool_receipt.get("scope_receipt_sha256") != file_sha256(
        scope_receipt_path
    ):
        raise ValueError("Holdout pool does not bind the scope receipt")
    if pool_receipt.get("retrieval_receipt_sha256") != file_sha256(
        retrieval_receipt_path
    ):
        raise ValueError("Holdout pool does not bind the retrieval receipt")
    if pool_receipt.get("method_lock_sha256") != file_sha256(method_lock_path):
        raise ValueError("Holdout pool does not bind the method lock")
    if pool_receipt.get("pool_audit_sha256") != file_sha256(pool_path):
        raise ValueError("Holdout pool audit hash mismatch")
    run_files = retrieval_receipt.get("run_files", {})
    base_run = run_files.get(BASE_RUN_ID, {}) if isinstance(run_files, dict) else {}
    if base_run.get("sha256") != file_sha256(ranking_path):
        raise ValueError("Holdout base ranking hash mismatch")
    query_by_id = keyed_rows(queries, label="query")
    pool_by_id = keyed_rows(pool_rows, label="pool")
    ranking_by_id = keyed_rows(ranking_rows, label="ranking")
    if set(query_by_id) != set(pool_by_id) or set(query_by_id) != set(ranking_by_id):
        raise ValueError("Holdout query, pool, and ranking IDs do not match")
    for query_id, query in query_by_id.items():
        pool = pool_by_id[query_id]
        if pool.get("split") != "holdout":
            raise ValueError("Holdout pool audit contains another split")
        if str(pool.get("query", "")) != str(query.get("query", "")):
            raise ValueError(f"Holdout pool query text mismatch: {query_id}")
    if pool_receipt.get("holdout_results_opened") is not True:
        raise ValueError("Holdout pool does not record authorized access")
    if pool_receipt.get("relevance_review_complete") is not False:
        raise ValueError("Verifier must run before reading holdout judgments")
    if file_sha256(pool_receipt_path) == "":
        raise AssertionError("Unreachable empty pool receipt hash")
    return top_k


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--scope-receipt", type=Path, required=True)
    parser.add_argument("--pool-audit", type=Path, required=True)
    parser.add_argument("--pool-receipt", type=Path, required=True)
    parser.add_argument("--retrieval-receipt", type=Path, required=True)
    parser.add_argument("--ranking", type=Path, required=True)
    parser.add_argument("--method-lock", type=Path, required=True)
    parser.add_argument(
        "--model", type=Path, default=PROJECT_ROOT / "models/Qwen3-VL-Reranker-2B"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=80)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-pixels", type=int, default=384 * 384)
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    paths = {
        key: Path(value).resolve()
        for key, value in vars(args).items()
        if isinstance(value, Path)
    }
    queries = read_jsonl(paths["queries"])
    pool_rows = read_jsonl(paths["pool_audit"])
    ranking_rows = read_jsonl(paths["ranking"])
    scope_receipt = read_json(paths["scope_receipt"])
    pool_receipt = read_json(paths["pool_receipt"])
    retrieval_receipt = read_json(paths["retrieval_receipt"])
    method_lock = read_json(paths["method_lock"])
    top_k = validate_scope(
        queries,
        pool_rows,
        ranking_rows,
        scope_receipt=scope_receipt,
        pool_receipt=pool_receipt,
        retrieval_receipt=retrieval_receipt,
        method_lock=method_lock,
        scope_receipt_path=paths["scope_receipt"],
        pool_path=paths["pool_audit"],
        pool_receipt_path=paths["pool_receipt"],
        retrieval_receipt_path=paths["retrieval_receipt"],
        method_lock_path=paths["method_lock"],
        ranking_path=paths["ranking"],
        expected_count=args.expected_count,
    )
    tasks = build_verification_tasks(pool_rows, ranking_rows, top_k=top_k)
    run_identity = {
        "scope": "v18_holdout_locked_top_k_candidates_only",
        "judgments_read": False,
        "query_count": len(tasks),
        "top_k": top_k,
        "model": "Qwen3-VL-Reranker-2B",
        "method_lock_sha256": file_sha256(paths["method_lock"]),
        "queries_sha256": file_sha256(paths["queries"]),
        "scope_receipt_sha256": file_sha256(paths["scope_receipt"]),
        "pool_audit_sha256": file_sha256(paths["pool_audit"]),
        "pool_receipt_sha256": file_sha256(paths["pool_receipt"]),
        "retrieval_receipt_sha256": file_sha256(paths["retrieval_receipt"]),
        "ranking_sha256": file_sha256(paths["ranking"]),
        "holdout_results_opened": True,
        "v17_artifacts_modified": False,
    }
    ordered_ids = [str(task["query_id"]) for task in tasks]
    results, previous_elapsed, previous_peak_gib = resume_results(
        paths["output"], run_identity=run_identity, ordered_query_ids=ordered_ids
    )
    pending = tasks[len(results) :]
    if not pending:
        print(paths["output"])
        return

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("V18 holdout verification requires CUDA")
    if str(OFFICIAL_REPO) not in sys.path:
        sys.path.insert(0, str(OFFICIAL_REPO))
    from src.models.qwen3_vl_reranker import Qwen3VLReranker

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = Qwen3VLReranker(
        model_name_or_path=str(paths["model"]),
        max_length=args.max_length,
        min_pixels=32 * 32 * 4,
        max_pixels=args.max_pixels,
        dtype=torch.float16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    load_seconds = time.perf_counter() - started
    for index, task in enumerate(pending, start=len(results) + 1):
        print(f"[{index:02d}/{len(tasks)}] {task['query_id']}", flush=True)
        documents = [
            {"image": str(Path(row["image_path"]).resolve())}
            for row in task["candidates"]
        ]
        query_started = time.perf_counter()
        full_scores = model.process(
            {
                "instruction": FULL_QUERY_INSTRUCTION,
                "query": {"text": task["query"]},
                "documents": documents,
            }
        )
        candidates = [
            {**candidate, "full_query_score": round(float(full_scores[offset]), 8)}
            for offset, candidate in enumerate(task["candidates"])
        ]
        results.append(
            {
                "query_id": task["query_id"],
                "query": task["query"],
                "group_id": task["group_id"],
                "candidates": candidates,
                "elapsed_seconds": round(time.perf_counter() - query_started, 3),
            }
        )
        write_json_atomic(
            paths["output"],
            {
                "schema_version": 1,
                "status": "partial" if index < len(tasks) else "complete",
                **run_identity,
                "completed_query_count": len(results),
                "load_seconds": round(load_seconds, 3),
                "elapsed_seconds": round(
                    previous_elapsed + time.perf_counter() - started, 3
                ),
                "peak_reserved_gib": max(
                    previous_peak_gib,
                    round(torch.cuda.max_memory_reserved() / 1024**3, 3),
                ),
                "results": results,
            },
        )
    print(paths["output"])


def main() -> None:
    args = parse_args()
    lock = acquire_output_lock(args.output.resolve())
    try:
        run(args)
    finally:
        release_output_lock(lock)


if __name__ == "__main__":
    main()
