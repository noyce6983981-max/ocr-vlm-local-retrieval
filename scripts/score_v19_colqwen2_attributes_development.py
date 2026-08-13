"""Score necessary-condition evidence for ColQwen2 Top-K on development."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ocr_vlm_retrieval.gating.attribute_coverage import (  # noqa: E402
    load_attribute_policy,
)
from ocr_vlm_retrieval.gating.candidate_verification import (  # noqa: E402
    ATTRIBUTE_INSTRUCTION,
    build_ocr_evidence,
    load_ocr_lines,
    resolve_requirement_scores,
)
from ocr_vlm_retrieval.gating.condition_decomposition import (  # noqa: E402
    decompose_condition_query,
)
from ocr_vlm_retrieval.gating.contrastive_relations import (  # noqa: E402
    relation_margin,
)
from ocr_vlm_retrieval.gating.v18_contrastive_relations import (  # noqa: E402
    build_v18_relation_counterfactual,
    v18_counterfactual_prompt,
)
from ocr_vlm_retrieval.runtime.cache import write_json_atomic  # noqa: E402
from ocr_vlm_retrieval.runtime.late_interaction import (  # noqa: E402
    exclusive_process_lock,
)
from scripts.score_v19_colqwen2_verifier_development import (  # noqa: E402
    build_tasks,
)
from scripts.score_v19_v18_l1_development import read_json  # noqa: E402

DEFAULT_RETRIEVAL = (
    ROOT
    / "outputs/evaluation/v19/selective_intervention"
    / "development_colqwen2_v1.json"
)
DEFAULT_FULL_SCORES = (
    ROOT
    / "outputs/evaluation/v19/selective_intervention"
    / "development_colqwen2_v1_qwen_verifier_top3.json"
)
DEFAULT_LIBRARY = ROOT / "outputs/user_library"
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/evaluation/v19/selective_intervention"
    / "development_colqwen2_v1_attribute_evidence_top3.json"
)


def full_score_index(payload: dict[str, Any]) -> dict[tuple[str, str], float]:
    result: dict[tuple[str, str], float] = {}
    for row in payload.get("results", []):
        query_id = str(row["query_id"])
        item_ids = [str(item_id) for item_id in row["candidate_item_ids"]]
        scores = [float(score) for score in row["scores"]]
        if len(item_ids) != len(scores):
            raise ValueError(f"full score length mismatch for {query_id}")
        result.update(
            ((query_id, item_id), score)
            for item_id, score in zip(item_ids, scores, strict=True)
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval", type=Path, default=DEFAULT_RETRIEVAL)
    parser.add_argument("--full-scores", type=Path, default=DEFAULT_FULL_SCORES)
    parser.add_argument("--library-dir", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument(
        "--model", type=Path, default=ROOT / "models/qwen3-vl-reranker-2b"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--policy", type=Path, default=ROOT / "config/v17_attribute_coverage.json"
    )
    parser.add_argument(
        "--ocr-root", type=Path, default=ROOT / "outputs/user_library/ocr/json"
    )
    parser.add_argument("--ocr-min-confidence", type=float, default=0.35)
    parser.add_argument("--ocr-fuzzy-threshold", type=float, default=0.88)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-pixels", type=int, default=384 * 384)
    parser.add_argument("--min-free-gib", type=float, default=5.0)
    parser.add_argument("--throttle-ms", type=int, default=250)
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    if args.top_k <= 0 or args.throttle_ms < 0:
        raise ValueError("top-k must be positive and throttle-ms non-negative")
    retrieval = read_json(args.retrieval)
    tasks = build_tasks(
        retrieval,
        library_dir=args.library_dir,
        top_k=args.top_k,
    )
    retrieval_by_query = {
        str(row["query_id"]): row for row in retrieval.get("results", [])
    }
    full_scores = full_score_index(read_json(args.full_scores))
    policy = load_attribute_policy(args.policy)
    completed: dict[str, dict[str, Any]] = {}
    if args.output.is_file():
        existing = read_json(args.output)
        if int(existing.get("top_k", -1)) != args.top_k:
            raise ValueError("existing attribute payload has a different top-k")
        completed = {
            str(row["query_id"]): row for row in existing.get("results", [])
        }
    pending = [task for task in tasks if str(task["query_id"]) not in completed]
    if not pending:
        print(f"attribute evidence already complete: {args.output.resolve()}")
        return 0

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("attribute verification requires CUDA")
    free_bytes, _ = torch.cuda.mem_get_info()
    free_gib = free_bytes / (1024**3)
    if free_gib < args.min_free_gib:
        raise RuntimeError(
            f"only {free_gib:.2f} GiB GPU memory is free; "
            f"at least {args.min_free_gib:.2f} GiB is required"
        )
    official_repo = ROOT / "third_party/Qwen3-VL-Embedding"
    if str(official_repo) not in sys.path:
        sys.path.insert(0, str(official_repo))
    from src.models.qwen3_vl_reranker import Qwen3VLReranker

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = Qwen3VLReranker(
        model_name_or_path=str(args.model.resolve()),
        max_length=args.max_length,
        min_pixels=32 * 32 * 4,
        max_pixels=args.max_pixels,
        dtype=torch.float16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    load_seconds = time.perf_counter() - started

    for task_index, task in enumerate(tasks, start=1):
        query_id = str(task["query_id"])
        if query_id in completed:
            continue
        plan = decompose_condition_query(str(task["query"]), policy)
        documents = [
            {"image": candidate["image_path"]} for candidate in task["candidates"]
        ]
        print(
            f"[{task_index:03d}/{len(tasks)}] {query_id}: "
            f"{len(plan.requirements)} conditions",
            flush=True,
        )
        scores_by_requirement: dict[str, list[float]] = {}
        for requirement in plan.requirements:
            scores_by_requirement[requirement.requirement_id] = model.process(
                {
                    "instruction": ATTRIBUTE_INSTRUCTION,
                    "query": {"text": requirement.prompt},
                    "documents": documents,
                }
            )
            if args.throttle_ms:
                time.sleep(args.throttle_ms / 1000)
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
            }
            if args.throttle_ms:
                time.sleep(args.throttle_ms / 1000)

        retrieval_row = retrieval_by_query[query_id]
        colqwen_scores = [float(value) for value in retrieval_row["scores"]]
        candidates: list[dict[str, Any]] = []
        for candidate_index, candidate in enumerate(task["candidates"]):
            item_id = str(candidate["item_id"])
            model_requirement_scores = {
                requirement.requirement_id: round(
                    float(scores_by_requirement[requirement.requirement_id][candidate_index]),
                    8,
                )
                for requirement in plan.requirements
            }
            ocr_lines = load_ocr_lines(
                args.ocr_root / f"{item_id}.json",
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
            contrastive_evidence: list[dict[str, Any]] = []
            for requirement_id, row in counterfactual_rows.items():
                positive = float(model_requirement_scores[requirement_id])
                negative = float(row["negative_scores"][candidate_index])
                contrastive_evidence.append(
                    {
                        "requirement_id": requirement_id,
                        "positive_value": row["positive_value"],
                        "negative_value": row["negative_value"],
                        "positive_score": round(positive, 8),
                        "negative_score": round(negative, 8),
                        "margin": round(relation_margin(positive, negative), 8),
                    }
                )
            candidates.append(
                {
                    "item_id": item_id,
                    "retrieval_rank": int(candidate["retrieval_rank"]),
                    "colqwen2_score": round(
                        colqwen_scores[int(candidate["retrieval_rank"]) - 1], 6
                    ),
                    "full_query_score": round(full_scores[(query_id, item_id)], 8),
                    "requirement_scores": model_requirement_scores,
                    "resolved_requirement_scores": resolved_scores,
                    "requirement_score_sources": score_sources,
                    "ocr_evidence": ocr_evidence,
                    "contrastive_relation_evidence": contrastive_evidence,
                }
            )
        completed[query_id] = {
            **{key: value for key, value in task.items() if key != "candidates"},
            "attribute_plan": plan.to_dict(),
            "candidates": candidates,
        }
        ordered_results = [
            completed[str(row["query_id"])]
            for row in tasks
            if str(row["query_id"]) in completed
        ]
        write_json_atomic(
            args.output,
            {
                "status": (
                    "complete"
                    if len(ordered_results) == len(tasks)
                    else "partial_development_diagnostic_only"
                ),
                "method": "colqwen2_topk_attribute_level_qwen_verification",
                "split": "development_only",
                "eligible_for_final_claim": False,
                "top_k": args.top_k,
                "query_count": len(tasks),
                "completed_query_count": len(ordered_results),
                "load_seconds": round(load_seconds, 3),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "peak_reserved_gib": round(
                    torch.cuda.max_memory_reserved() / (1024**3), 3
                ),
                "results": ordered_results,
            },
        )
    print(args.output.resolve())
    return 0


def main() -> int:
    args = parse_args()
    lock_path = args.output.parent / ".v19_gpu_job.lock"
    with exclusive_process_lock(lock_path):
        return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
