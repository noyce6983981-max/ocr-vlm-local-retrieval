"""Verify ColQwen2 candidates with Qwen3-VL on V19 development only."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ocr_vlm_retrieval.gating.candidate_verification import (  # noqa: E402
    FULL_QUERY_INSTRUCTION,
)
from ocr_vlm_retrieval.runtime.cache import write_json_atomic  # noqa: E402
from ocr_vlm_retrieval.runtime.late_interaction import (  # noqa: E402
    exclusive_process_lock,
)
from scripts.score_v19_v18_l1_development import (  # noqa: E402
    l1_result,
    manifest_index,
    read_json,
    summarize,
)

DEFAULT_RETRIEVAL = (
    ROOT
    / "outputs/evaluation/v19/selective_intervention"
    / "development_colqwen2_v1.json"
)
DEFAULT_BASELINE_CACHE = (
    ROOT
    / "outputs/evaluation/v19/selective_intervention"
    / "development_true_v18_l1_top3.json"
)
DEFAULT_LIBRARY = ROOT / "outputs/user_library"
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/evaluation/v19/selective_intervention"
    / "development_colqwen2_v1_qwen_verifier_top3.json"
)


def build_tasks(
    retrieval_payload: Mapping[str, Any],
    *,
    library_dir: Path,
    top_k: int,
) -> list[dict[str, Any]]:
    if retrieval_payload.get("split") != "development_only":
        raise ValueError("candidate verification may read V19 development only")
    if bool(retrieval_payload.get("eligible_for_final_claim")):
        raise ValueError("development retrieval must not be marked as a final claim")
    manifest = manifest_index(library_dir)
    tasks: list[dict[str, Any]] = []
    for row in retrieval_payload.get("results", []):
        candidates: list[dict[str, Any]] = []
        for rank, item_id in enumerate(row.get("ranking_item_ids", [])[:top_k], 1):
            item_id = str(item_id)
            if item_id not in manifest:
                raise ValueError(f"manifest missing {item_id}")
            image_path = ROOT / str(manifest[item_id]["source_path"])
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            candidates.append(
                {
                    "item_id": item_id,
                    "retrieval_rank": rank,
                    "image_path": str(image_path.resolve()),
                }
            )
        tasks.append(
            {
                "query_id": str(row["query_id"]),
                "query": str(row["query"]),
                "query_role": row.get("query_role"),
                "content_stratum": row.get("content_stratum"),
                "gold_answerable": bool(row.get("gold_answerable")),
                "gold_relevant_item_ids": [
                    str(item_id)
                    for item_id in row.get("gold_relevant_item_ids", [])
                ],
                "candidates": candidates,
            }
        )
    if not tasks:
        raise ValueError("no ColQwen2 verification tasks")
    return tasks


def score_cache_from_payload(payload: Mapping[str, Any]) -> dict[tuple[str, str], float]:
    cache: dict[tuple[str, str], float] = {}
    for row in payload.get("results", []):
        query_id = str(row["query_id"])
        item_ids = [str(item_id) for item_id in row.get("candidate_item_ids", [])]
        scores = [float(score) for score in row.get("scores", [])]
        if len(item_ids) != len(scores):
            raise ValueError(f"score cache length mismatch for {query_id}")
        cache.update(
            ((query_id, item_id), score)
            for item_id, score in zip(item_ids, scores, strict=True)
        )
    return cache


def results_at_threshold(
    tasks: Sequence[Mapping[str, Any]],
    score_rows: Sequence[Mapping[str, Any]],
    *,
    threshold: float,
) -> list[dict[str, Any]]:
    scores_by_query = {
        str(row["query_id"]): [float(score) for score in row["scores"]]
        for row in score_rows
    }
    return [
        l1_result(
            task,
            scores_by_query[str(task["query_id"])],
            threshold=threshold,
        )
        for task in tasks
    ]


def choose_threshold(
    tasks: Sequence[Mapping[str, Any]],
    score_rows: Sequence[Mapping[str, Any]],
    *,
    max_false_accept_rate: float,
) -> tuple[float, dict[str, Any]]:
    if not 0.0 <= max_false_accept_rate <= 1.0:
        raise ValueError("max false accept rate must be within [0, 1]")
    top_scores = [max(float(score) for score in row["scores"]) for row in score_rows]
    thresholds = {0.0, 1.0}
    for score in top_scores:
        thresholds.add(score)
        thresholds.add(math.nextafter(score, math.inf))
    best: tuple[tuple[float, float, float, float], float, dict[str, Any]] | None = None
    for threshold in sorted(thresholds):
        summary = summarize(
            results_at_threshold(tasks, score_rows, threshold=threshold)
        )
        if float(summary["false_accept_rate"]) > max_false_accept_rate + 1e-12:
            continue
        objective = (
            float(summary["end_to_end_accuracy"]),
            float(summary["positive_selected_relevant"]),
            -float(summary["false_accept_rate"]),
            threshold,
        )
        if best is None or objective > best[0]:
            best = (objective, threshold, summary)
    if best is None:
        raise ValueError("no threshold satisfies the false-accept constraint")
    return best[1], best[2]


def grouped_cross_validated_summary(
    tasks: Sequence[Mapping[str, Any]],
    score_rows: Sequence[Mapping[str, Any]],
    *,
    max_false_accept_rate: float,
) -> dict[str, Any]:
    score_by_id = {str(row["query_id"]): row for row in score_rows}
    families: dict[str, list[Mapping[str, Any]]] = {}
    for task in tasks:
        query_id = str(task["query_id"])
        family_id = query_id.rsplit("_q", 1)[0]
        families.setdefault(family_id, []).append(task)
    held_out_results: list[dict[str, Any]] = []
    thresholds: list[float] = []
    for family_id, held_out_tasks in sorted(families.items()):
        training_tasks = [
            task
            for task in tasks
            if str(task["query_id"]).rsplit("_q", 1)[0] != family_id
        ]
        training_scores = [score_by_id[str(task["query_id"])] for task in training_tasks]
        threshold, _ = choose_threshold(
            training_tasks,
            training_scores,
            max_false_accept_rate=max_false_accept_rate,
        )
        thresholds.append(threshold)
        held_out_scores = [score_by_id[str(task["query_id"])] for task in held_out_tasks]
        held_out_results.extend(
            results_at_threshold(
                held_out_tasks,
                held_out_scores,
                threshold=threshold,
            )
        )
    return {
        "protocol": "leave_one_query_family_out_threshold_selection",
        "family_count": len(families),
        "threshold_min": min(thresholds),
        "threshold_median": sorted(thresholds)[len(thresholds) // 2],
        "threshold_max": max(thresholds),
        **summarize(held_out_results),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval", type=Path, default=DEFAULT_RETRIEVAL)
    parser.add_argument("--baseline-cache", type=Path, default=DEFAULT_BASELINE_CACHE)
    parser.add_argument(
        "--additional-cache",
        type=Path,
        action="append",
        default=[],
        help="Additional compatible score payload; may be repeated.",
    )
    parser.add_argument("--library-dir", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument(
        "--model", type=Path, default=ROOT / "models/qwen3-vl-reranker-2b"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--fixed-threshold", type=float, default=0.63)
    parser.add_argument("--max-far", type=float, default=0.34)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-pixels", type=int, default=384 * 384)
    parser.add_argument("--min-free-gib", type=float, default=5.0)
    parser.add_argument("--throttle-ms", type=int, default=750)
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    if args.top_k <= 0 or args.throttle_ms < 0:
        raise ValueError("top-k must be positive and throttle-ms non-negative")
    tasks = build_tasks(
        read_json(args.retrieval),
        library_dir=args.library_dir,
        top_k=args.top_k,
    )
    cache = score_cache_from_payload(read_json(args.baseline_cache))
    for cache_path in args.additional_cache:
        cache.update(score_cache_from_payload(read_json(cache_path)))
    if args.output.is_file():
        cache.update(score_cache_from_payload(read_json(args.output)))
    missing_count = sum(
        (str(task["query_id"]), str(candidate["item_id"])) not in cache
        for task in tasks
        for candidate in task["candidates"]
    )

    model: Any = None
    load_seconds = 0.0
    peak_reserved_gib = 0.0
    if missing_count:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Qwen3-VL candidate verification requires CUDA")
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
        load_started = time.perf_counter()
        model = Qwen3VLReranker(
            model_name_or_path=str(args.model.resolve()),
            max_length=args.max_length,
            min_pixels=32 * 32 * 4,
            max_pixels=args.max_pixels,
            dtype=torch.float16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
        load_seconds = time.perf_counter() - load_started

    started = time.perf_counter()
    score_rows: list[dict[str, Any]] = []
    for index, task in enumerate(tasks, start=1):
        query_id = str(task["query_id"])
        missing = [
            candidate
            for candidate in task["candidates"]
            if (query_id, str(candidate["item_id"])) not in cache
        ]
        if missing:
            print(
                f"[{index:03d}/{len(tasks)}] {query_id}: scoring {len(missing)} new",
                flush=True,
            )
            new_scores = model.process(
                {
                    "instruction": FULL_QUERY_INSTRUCTION,
                    "query": {"text": task["query"]},
                    "documents": [
                        {"image": candidate["image_path"]} for candidate in missing
                    ],
                }
            )
            cache.update(
                ((query_id, str(candidate["item_id"])), float(score))
                for candidate, score in zip(missing, new_scores, strict=True)
            )
            if args.throttle_ms:
                time.sleep(args.throttle_ms / 1000)
        score_rows.append(
            {
                "query_id": query_id,
                "candidate_item_ids": [
                    str(candidate["item_id"]) for candidate in task["candidates"]
                ],
                "scores": [
                    cache[(query_id, str(candidate["item_id"]))]
                    for candidate in task["candidates"]
                ],
            }
        )
        partial = {
            "status": "partial_development_diagnostic_only",
            "method": "colqwen2_v1_topk_then_qwen3_vl_pointwise_verifier",
            "split": "development_only",
            "eligible_for_final_claim": False,
            "top_k": args.top_k,
            "results": score_rows,
        }
        write_json_atomic(args.output, partial)

    fixed_results = results_at_threshold(
        tasks, score_rows, threshold=args.fixed_threshold
    )
    tuned_threshold, tuned_summary = choose_threshold(
        tasks,
        score_rows,
        max_false_accept_rate=args.max_far,
    )
    import torch

    if torch.cuda.is_available():
        peak_reserved_gib = torch.cuda.max_memory_reserved() / (1024**3)
    payload = {
        "status": "development_diagnostic_only",
        "method": "colqwen2_v1_topk_then_qwen3_vl_pointwise_verifier",
        "split": "development_only",
        "eligible_for_final_claim": False,
        "metric_warning": (
            "The tuned threshold uses development labels. Only grouped cross-validation "
            "is an out-of-family estimate; neither is a final holdout claim."
        ),
        "parameters": {
            "top_k": args.top_k,
            "fixed_threshold": args.fixed_threshold,
            "max_false_accept_rate": args.max_far,
            "throttle_ms": args.throttle_ms,
        },
        "cache": {
            "candidate_pair_count": sum(len(task["candidates"]) for task in tasks),
            "newly_scored_pair_count": missing_count,
        },
        "fixed_threshold_summary": summarize(fixed_results),
        "tuned_threshold": tuned_threshold,
        "tuned_threshold_summary": tuned_summary,
        "grouped_cross_validated_summary": grouped_cross_validated_summary(
            tasks,
            score_rows,
            max_false_accept_rate=args.max_far,
        ),
        "load_seconds": round(load_seconds, 3),
        "inference_seconds": round(time.perf_counter() - started, 3),
        "peak_reserved_gib": round(peak_reserved_gib, 3),
        "results": score_rows,
    }
    write_json_atomic(args.output, payload)
    print(json.dumps(payload["fixed_threshold_summary"], indent=2))
    print(
        json.dumps(
            {
                "tuned_threshold": tuned_threshold,
                "tuned": tuned_summary,
                "grouped_cv": payload["grouped_cross_validated_summary"],
            },
            indent=2,
        )
    )
    return 0


def main() -> int:
    args = parse_args()
    lock_path = args.output.parent / ".v19_gpu_job.lock"
    with exclusive_process_lock(lock_path):
        return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
