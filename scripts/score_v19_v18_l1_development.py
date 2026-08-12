"""Reproduce the frozen V18 L1 verifier baseline on V19 development only."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

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
from scripts.run_v19_downstream_retrieval_pilot import (  # noqa: E402
    ranking_ids,
)

DEFAULT_ASSIGNMENTS = (
    ROOT
    / "outputs/evaluation/v19/selective_intervention"
    / "development_route_assignments.json"
)
DEFAULT_RETRIEVAL_DIR = (
    ROOT / "outputs/evaluation/v19/selective_intervention/retrieval/baseline"
)
DEFAULT_LIBRARY = ROOT / "outputs/user_library"
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/evaluation/v19/selective_intervention"
    / "development_true_v18_l1_top3.json"
)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def manifest_index(library_dir: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    with (library_dir / "manifest.jsonl").open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            item_id = str(row.get("item_id", ""))
            if item_id:
                result[item_id] = row
    return result


def build_tasks(
    assignments_payload: Mapping[str, Any],
    *,
    retrieval_dir: Path,
    library_dir: Path,
    top_k: int,
) -> list[dict[str, Any]]:
    if assignments_payload.get("split") != "v19_reviewed_development_only":
        raise ValueError("V18 L1 reproduction may read V19 development only")
    manifest = manifest_index(library_dir)
    tasks: list[dict[str, Any]] = []
    for assignment in assignments_payload.get("assignments", []):
        query_id = str(assignment["query_id"])
        retrieval = read_json(retrieval_dir / f"{query_id}_v18_frozen.json")
        if retrieval.get("query") != assignment.get("query"):
            raise ValueError(f"retrieval query mismatch for {query_id}")
        candidates: list[dict[str, Any]] = []
        for rank, item_id in enumerate(ranking_ids(retrieval)[:top_k], start=1):
            if item_id not in manifest:
                raise ValueError(f"manifest missing {item_id}")
            source_path = ROOT / str(manifest[item_id]["source_path"])
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            candidates.append(
                {
                    "item_id": item_id,
                    "retrieval_rank": rank,
                    "image_path": str(source_path.resolve()),
                }
            )
        relevant = [
            str(item_id)
            for item_id in assignment.get("gold_relevant_item_ids", [])
            if str(item_id).strip()
        ]
        if bool(assignment.get("gold_answerable")) and not relevant:
            source = str(assignment.get("source_item_id") or "")
            if source:
                relevant = [source]
        tasks.append(
            {
                "query_id": query_id,
                "query": str(assignment["query"]),
                "query_role": assignment.get("query_role"),
                "content_stratum": assignment.get("content_stratum"),
                "gold_answerable": bool(assignment.get("gold_answerable")),
                "gold_relevant_item_ids": relevant,
                "candidates": candidates,
            }
        )
    if not tasks:
        raise ValueError("no V19 development tasks")
    return tasks


def l1_result(
    task: Mapping[str, Any], scores: Sequence[float], *, threshold: float
) -> dict[str, Any]:
    candidates = list(task.get("candidates", []))
    if len(candidates) != len(scores):
        raise ValueError("candidate and score lengths differ")
    ordered = sorted(
        zip(candidates, scores, strict=True),
        key=lambda pair: (-float(pair[1]), int(pair[0]["retrieval_rank"])),
    )
    selected = ordered[0] if ordered else None
    accepted = selected is not None and float(selected[1]) >= threshold
    return {
        "query_id": task["query_id"],
        "query_role": task.get("query_role"),
        "content_stratum": task.get("content_stratum"),
        "gold_answerable": bool(task.get("gold_answerable")),
        "gold_relevant_item_ids": list(task.get("gold_relevant_item_ids", [])),
        "candidate_item_ids": [str(row["item_id"]) for row in candidates],
        "scores": [round(float(score), 8) for score in scores],
        "selected_item_id": str(selected[0]["item_id"]) if accepted else None,
        "selected_retrieval_rank": (
            int(selected[0]["retrieval_rank"]) if accepted else None
        ),
        "accepted": accepted,
        "top_score": round(float(selected[1]), 8) if selected else None,
    }


def summarize(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    positives = [row for row in results if bool(row.get("gold_answerable"))]
    negatives = [row for row in results if not bool(row.get("gold_answerable"))]

    def selected_relevant(row: Mapping[str, Any]) -> bool:
        return bool(row.get("accepted")) and str(row.get("selected_item_id")) in {
            str(item_id) for item_id in row.get("gold_relevant_item_ids", [])
        }

    positive_correct = sum(selected_relevant(row) for row in positives)
    negative_correct = sum(not bool(row.get("accepted")) for row in negatives)

    def rate(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    return {
        "query_count": len(results),
        "positive_count": len(positives),
        "negative_count": len(negatives),
        "positive_selected_relevant": rate(positive_correct, len(positives)),
        "negative_correct_reject_rate": rate(negative_correct, len(negatives)),
        "end_to_end_accuracy": rate(positive_correct + negative_correct, len(results)),
        "false_accept_rate": (
            1.0 - rate(negative_correct, len(negatives)) if negatives else 0.0
        ),
        "false_reject_rate": rate(
            sum(not bool(row.get("accepted")) for row in positives),
            len(positives),
        ),
        "empty_candidate_count": sum(
            not row.get("candidate_item_ids") for row in results
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--retrieval-dir", type=Path, default=DEFAULT_RETRIEVAL_DIR)
    parser.add_argument("--library-dir", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument(
        "--model", type=Path, default=ROOT / "models/qwen3-vl-reranker-2b"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.63)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-pixels", type=int, default=384 * 384)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.top_k != 3 or args.threshold != 0.63:
        raise ValueError("frozen V18 L1 requires top_k=3 and threshold=0.63")
    tasks = build_tasks(
        read_json(args.assignments),
        retrieval_dir=args.retrieval_dir,
        library_dir=args.library_dir,
        top_k=args.top_k,
    )
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("V18 L1 reproduction requires CUDA")
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
    inference_started = time.perf_counter()
    results: list[dict[str, Any]] = []
    for index, task in enumerate(tasks, start=1):
        print(f"[{index:03d}/{len(tasks)}] {task['query_id']}", flush=True)
        scores = model.process(
            {
                "instruction": FULL_QUERY_INSTRUCTION,
                "query": {"text": task["query"]},
                "documents": [
                    {"image": candidate["image_path"]}
                    for candidate in task["candidates"]
                ],
            }
        )
        results.append(l1_result(task, scores, threshold=args.threshold))
        summary = summarize(results)
        write_json_atomic(
            args.output,
            {
                "status": (
                    "development_diagnostic_only"
                    if index == len(tasks)
                    else "partial_development_diagnostic_only"
                ),
                "method": "frozen_v18_L1_max_verifier_score",
                "split": "development_only",
                "eligible_for_final_claim": False,
                "parameters": {
                    "top_k": args.top_k,
                    "full_query_threshold": args.threshold,
                },
                **summary,
                "load_seconds": round(load_seconds, 3),
                "inference_seconds": round(time.perf_counter() - inference_started, 3),
                "peak_reserved_gib": round(
                    torch.cuda.max_memory_reserved() / 1024**3, 3
                ),
                "results": results,
            },
        )
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
