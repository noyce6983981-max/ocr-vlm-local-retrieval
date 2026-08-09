"""Score V18 calibration candidates with clause-level visual evidence."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, BinaryIO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_REPO = PROJECT_ROOT / "third_party/Qwen3-VL-Embedding"
PACKAGE_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from ocr_vlm_retrieval.gating.attribute_coverage import (  # noqa: E402
    load_attribute_policy,
)
from ocr_vlm_retrieval.gating.candidate_verification import (  # noqa: E402
    ATTRIBUTE_INSTRUCTION,
    FULL_QUERY_INSTRUCTION,
    build_ocr_evidence,
    load_ocr_lines,
    resolve_requirement_scores,
    verification_prompt_payload,
)
from ocr_vlm_retrieval.gating.condition_decomposition import (  # noqa: E402
    PARSER_VERSION,
    decompose_condition_query,
)
from ocr_vlm_retrieval.gating.contrastive_relations import (  # noqa: E402
    relation_margin,
)
from ocr_vlm_retrieval.gating.v18_contrastive_relations import (  # noqa: E402
    build_v18_relation_counterfactual,
    v18_counterfactual_prompt,
)
from ocr_vlm_retrieval.studies.query_split import (  # noqa: E402
    query_set_fingerprint,
)
from scripts.run_v17_calibration_retrieval import (  # noqa: E402
    file_sha256,
    read_jsonl,
    write_json_atomic,
)
from scripts.score_v17_candidate_verification import (  # noqa: E402
    resume_results,
)

BASE_RUN_ID = "v17_quality_hybrid"


def acquire_output_lock(output_path: Path) -> BinaryIO:
    """Acquire a non-blocking process lock for one verification output."""

    lock_path = output_path.with_suffix(f"{output_path.suffix}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    handle.seek(0, os.SEEK_END)
    if handle.tell() == 0:
        handle.write(b"\0")
        handle.flush()
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        handle.close()
        raise RuntimeError(
            f"Another verification process is already writing {output_path}"
        ) from error
    return handle


def release_output_lock(handle: BinaryIO) -> None:
    try:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def keyed_rows(
    rows: Iterable[Mapping[str, Any]], *, label: str
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for source in rows:
        query_id = str(source.get("query_id", "")).strip()
        if not query_id or query_id in result:
            raise ValueError(f"Every {label} row needs a unique query_id")
        result[query_id] = dict(source)
    return result


def validate_scope(
    queries: Sequence[Mapping[str, Any]],
    pool_rows: Sequence[Mapping[str, Any]],
    ranking_rows: Sequence[Mapping[str, Any]],
    *,
    scope_receipt: Mapping[str, Any],
    pool_receipt: Mapping[str, Any],
    retrieval_receipt: Mapping[str, Any],
    scope_receipt_path: Path,
    pool_path: Path,
    retrieval_receipt_path: Path,
    methods_path: Path,
    ranking_path: Path,
    expected_count: int,
) -> None:
    if len(queries) != expected_count:
        raise ValueError(f"Expected {expected_count} V18 calibration queries")
    if any(row.get("split") != "calibration" for row in queries):
        raise ValueError("Verification may load calibration queries only")
    if scope_receipt.get("exported_split") != "calibration":
        raise ValueError("Scope receipt does not isolate calibration")
    if scope_receipt.get("holdout_results_opened") is not False:
        raise ValueError("Holdout access invariant is not satisfied")
    if pool_receipt.get("status") != (
        "calibration_pool_ready_for_blind_human_review"
    ):
        raise ValueError("V18 calibration pool is not ready")
    if pool_receipt.get("executed_split") != "calibration":
        raise ValueError("Pool receipt is not calibration-only")
    if pool_receipt.get("holdout_results_opened") is not False:
        raise ValueError("Pool receipt indicates holdout access")
    if retrieval_receipt.get("status") != (
        "calibration_retrieval_complete_relevance_not_yet_judged"
    ):
        raise ValueError("V18 calibration retrieval is not complete")
    query_fingerprint = query_set_fingerprint(queries)
    if scope_receipt.get("calibration_query_set_sha256") != query_fingerprint:
        raise ValueError("Scope query fingerprint mismatch")
    if pool_receipt.get("calibration_query_set_sha256") != query_fingerprint:
        raise ValueError("Pool query fingerprint mismatch")
    if pool_receipt.get("scope_receipt_sha256") != file_sha256(scope_receipt_path):
        raise ValueError("Pool does not bind this scope receipt")
    if pool_receipt.get("retrieval_receipt_sha256") != file_sha256(
        retrieval_receipt_path
    ):
        raise ValueError("Pool does not bind this retrieval receipt")
    if pool_receipt.get("pool_audit_sha256") != file_sha256(pool_path):
        raise ValueError("Pool audit hash mismatch")
    methods_sha256 = file_sha256(methods_path)
    if scope_receipt.get("methods_sha256") != methods_sha256:
        raise ValueError("Scope method hash mismatch")
    if pool_receipt.get("methods_sha256") != methods_sha256:
        raise ValueError("Pool method hash mismatch")
    run_files = retrieval_receipt.get("run_files", {})
    base_run = run_files.get(BASE_RUN_ID, {}) if isinstance(run_files, dict) else {}
    if base_run.get("sha256") != file_sha256(ranking_path):
        raise ValueError("Base ranking hash mismatch")
    query_by_id = keyed_rows(queries, label="query")
    pool_by_id = keyed_rows(pool_rows, label="pool")
    ranking_by_id = keyed_rows(ranking_rows, label="ranking")
    if set(query_by_id) != set(pool_by_id) or set(query_by_id) != set(ranking_by_id):
        raise ValueError("Query, pool, and ranking IDs do not match")
    for query_id, query in query_by_id.items():
        pool = pool_by_id[query_id]
        if pool.get("split") != "calibration":
            raise ValueError("Pool audit contains a non-calibration row")
        if str(pool.get("query", "")) != str(query.get("query", "")):
            raise ValueError(f"Pool query text mismatch: {query_id}")


def build_verification_tasks(
    pool_rows: Iterable[Mapping[str, Any]],
    ranking_rows: Iterable[Mapping[str, Any]],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    pools = keyed_rows(pool_rows, label="pool")
    rankings = keyed_rows(ranking_rows, label="ranking")
    if set(pools) != set(rankings):
        raise ValueError("Pool and ranking query IDs do not match")
    tasks: list[dict[str, Any]] = []
    for query_id in sorted(pools):
        pool = pools[query_id]
        metadata_by_id = {
            str(row.get("item_id", "")): dict(row.get("review_metadata", {}))
            for row in pool.get("candidates", [])
        }
        ranking = rankings[query_id].get("ranking", [])
        if not isinstance(ranking, list) or len(ranking) < top_k:
            raise ValueError(f"Ranking {query_id} has fewer than {top_k} candidates")
        candidates: list[dict[str, Any]] = []
        for rank, row in enumerate(ranking[:top_k], start=1):
            item_id = str(row.get("item_id", "")).strip()
            if item_id not in metadata_by_id:
                raise ValueError(
                    f"Base Top-{top_k} candidate {item_id!r} is outside the pool"
                )
            image_path = str(metadata_by_id[item_id].get("image_path", "")).strip()
            if not image_path:
                image_path = str(row.get("image_path", "")).strip()
            if not image_path:
                raise ValueError(f"Candidate {item_id} has no image path")
            candidates.append(
                {
                    "item_id": item_id,
                    "image_path": image_path,
                    "retrieval_rank": rank,
                    "ranking_score": float(row.get("score", 0.0)),
                }
            )
        tasks.append(
            {
                "query_id": query_id,
                "query": str(pool.get("query", "")).strip(),
                "group_id": str(pool.get("group_id", "")).strip(),
                "candidates": candidates,
            }
        )
    return tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--scope-receipt", type=Path, required=True)
    parser.add_argument("--pool-audit", type=Path, required=True)
    parser.add_argument("--pool-receipt", type=Path, required=True)
    parser.add_argument("--retrieval-receipt", type=Path, required=True)
    parser.add_argument("--ranking", type=Path, required=True)
    parser.add_argument(
        "--methods",
        type=Path,
        default=PROJECT_ROOT / "config/studies/v18_methods.json",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=PROJECT_ROOT / "config/v17_attribute_coverage.json",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "models/qwen3-vl-reranker-2b",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--ocr-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/user_library/ocr/json",
    )
    parser.add_argument("--expected-count", type=int, default=80)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-pixels", type=int, default=384 * 384)
    parser.add_argument("--ocr-min-confidence", type=float, default=0.35)
    parser.add_argument("--ocr-fuzzy-threshold", type=float, default=0.88)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    paths = {
        "queries": args.queries.resolve(),
        "scope_receipt": args.scope_receipt.resolve(),
        "pool_audit": args.pool_audit.resolve(),
        "pool_receipt": args.pool_receipt.resolve(),
        "retrieval_receipt": args.retrieval_receipt.resolve(),
        "ranking": args.ranking.resolve(),
        "methods": args.methods.resolve(),
        "policy": args.policy.resolve(),
        "model": args.model.resolve(),
        "output": args.output.resolve(),
        "ocr_root": args.ocr_root.resolve(),
    }
    queries = read_jsonl(paths["queries"])
    pool_rows = read_jsonl(paths["pool_audit"])
    ranking_rows = read_jsonl(paths["ranking"])
    scope_receipt = read_json(paths["scope_receipt"])
    pool_receipt = read_json(paths["pool_receipt"])
    retrieval_receipt = read_json(paths["retrieval_receipt"])
    validate_scope(
        queries,
        pool_rows,
        ranking_rows,
        scope_receipt=scope_receipt,
        pool_receipt=pool_receipt,
        retrieval_receipt=retrieval_receipt,
        scope_receipt_path=paths["scope_receipt"],
        pool_path=paths["pool_audit"],
        retrieval_receipt_path=paths["retrieval_receipt"],
        methods_path=paths["methods"],
        ranking_path=paths["ranking"],
        expected_count=args.expected_count,
    )
    tasks = build_verification_tasks(pool_rows, ranking_rows, top_k=args.top_k)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("limit must be positive")
        tasks = tasks[: args.limit]
    policy = load_attribute_policy(paths["policy"])
    parser_source = PROJECT_ROOT / (
        "src/ocr_vlm_retrieval/gating/condition_decomposition.py"
    )
    relation_source = PROJECT_ROOT / (
        "src/ocr_vlm_retrieval/gating/v18_contrastive_relations.py"
    )
    run_identity = {
        "scope": "v18_calibration_base_top10_candidates_only",
        "judgments_read": False,
        "query_count": len(tasks),
        "top_k": args.top_k,
        "model": "Qwen3-VL-Reranker-2B",
        "parser_version": PARSER_VERSION,
        "parser_source_sha256": file_sha256(parser_source),
        "relation_source_sha256": file_sha256(relation_source),
        "verification_prompts": verification_prompt_payload(),
        "methods_sha256": file_sha256(paths["methods"]),
        "policy_sha256": file_sha256(paths["policy"]),
        "queries_sha256": file_sha256(paths["queries"]),
        "scope_receipt_sha256": file_sha256(paths["scope_receipt"]),
        "pool_audit_sha256": file_sha256(paths["pool_audit"]),
        "pool_receipt_sha256": file_sha256(paths["pool_receipt"]),
        "retrieval_receipt_sha256": file_sha256(paths["retrieval_receipt"]),
        "ranking_sha256": file_sha256(paths["ranking"]),
        "ocr_policy": {
            "minimum_confidence": args.ocr_min_confidence,
            "fuzzy_threshold": args.ocr_fuzzy_threshold,
            "precedence": ["exact", "fuzzy", "visual_verifier_fallback"],
        },
        "holdout_results_opened": False,
        "v17_artifacts_modified": False,
    }
    ordered_ids = [str(task["query_id"]) for task in tasks]
    if args.restart:
        results: list[dict[str, Any]] = []
        previous_elapsed = 0.0
        previous_peak_gib = 0.0
    else:
        results, previous_elapsed, previous_peak_gib = resume_results(
            paths["output"],
            run_identity=run_identity,
            ordered_query_ids=ordered_ids,
        )
    pending_tasks = tasks[len(results) :]
    if not pending_tasks:
        print(paths["output"])
        return

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("V18 candidate verification requires CUDA")
    if str(OFFICIAL_REPO) not in sys.path:
        sys.path.insert(0, str(OFFICIAL_REPO))
    from src.models.qwen3_vl_reranker import Qwen3VLReranker

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    run_started = time.perf_counter()
    model = Qwen3VLReranker(
        model_name_or_path=str(paths["model"]),
        max_length=args.max_length,
        min_pixels=32 * 32 * 4,
        max_pixels=args.max_pixels,
        dtype=torch.float16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    load_seconds = time.perf_counter() - run_started
    for index, task in enumerate(pending_tasks, start=len(results) + 1):
        print(f"[{index:02d}/{len(tasks)}] {task['query_id']}", flush=True)
        plan = decompose_condition_query(str(task["query"]), policy)
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
        scores_by_requirement = {
            requirement.requirement_id: model.process(
                {
                    "instruction": ATTRIBUTE_INSTRUCTION,
                    "query": {"text": requirement.prompt},
                    "documents": documents,
                }
            )
            for requirement in plan.requirements
        }
        counterfactual_rows: dict[str, dict[str, Any]] = {}
        for requirement in plan.requirements:
            if requirement.kind != "relation":
                continue
            counterfactual = build_v18_relation_counterfactual(requirement.value)
            if counterfactual is None:
                continue
            negative_prompt = v18_counterfactual_prompt(counterfactual)
            counterfactual_rows[requirement.requirement_id] = {
                **counterfactual.to_dict(),
                "negative_prompt": negative_prompt,
                "negative_scores": model.process(
                    {
                        "instruction": ATTRIBUTE_INSTRUCTION,
                        "query": {"text": negative_prompt},
                        "documents": documents,
                    }
                ),
                "absolute_threshold": requirement.threshold,
            }
        candidate_rows: list[dict[str, Any]] = []
        for candidate_index, candidate in enumerate(task["candidates"]):
            model_requirement_scores = {
                requirement_id: round(float(values[candidate_index]), 8)
                for requirement_id, values in scores_by_requirement.items()
            }
            ocr_lines = load_ocr_lines(
                paths["ocr_root"] / f"{candidate['item_id']}.json",
                minimum_confidence=args.ocr_min_confidence,
            )
            ocr_evidence = build_ocr_evidence(
                plan,
                ocr_lines,
                fuzzy_threshold=args.ocr_fuzzy_threshold,
            )
            resolved_scores, score_sources = resolve_requirement_scores(
                model_requirement_scores,
                ocr_evidence,
            )
            contrastive_evidence = []
            for requirement_id, row in counterfactual_rows.items():
                positive_score = float(model_requirement_scores[requirement_id])
                negative_score = float(row["negative_scores"][candidate_index])
                contrastive_evidence.append(
                    {
                        "requirement_id": requirement_id,
                        "positive_value": row["positive_value"],
                        "negative_value": row["negative_value"],
                        "positive_marker": row["positive_marker"],
                        "negative_marker": row["negative_marker"],
                        "negative_prompt": row["negative_prompt"],
                        "positive_score": round(positive_score, 8),
                        "negative_score": round(negative_score, 8),
                        "margin": round(
                            relation_margin(positive_score, negative_score), 8
                        ),
                        "absolute_threshold": row["absolute_threshold"],
                    }
                )
            candidate_rows.append(
                {
                    **candidate,
                    "full_query_score": round(float(full_scores[candidate_index]), 8),
                    "requirement_scores": model_requirement_scores,
                    "resolved_requirement_scores": resolved_scores,
                    "requirement_score_sources": score_sources,
                    "ocr_evidence": ocr_evidence,
                    "contrastive_relation_evidence": contrastive_evidence,
                }
            )
        results.append(
            {
                "query_id": task["query_id"],
                "query": task["query"],
                "group_id": task["group_id"],
                "attribute_plan": plan.to_dict(),
                "candidates": candidate_rows,
                "elapsed_seconds": round(time.perf_counter() - query_started, 3),
            }
        )
        write_json_atomic(
            paths["output"],
            {
                "schema_version": 2,
                "status": "partial" if index < len(tasks) else "complete",
                **run_identity,
                "completed_query_count": len(results),
                "load_seconds": round(load_seconds, 3),
                "elapsed_seconds": round(
                    previous_elapsed + time.perf_counter() - run_started, 3
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
    output_path = args.output.resolve()
    lock_handle = acquire_output_lock(output_path)
    try:
        run(args)
    finally:
        release_output_lock(lock_handle)


if __name__ == "__main__":
    main()
