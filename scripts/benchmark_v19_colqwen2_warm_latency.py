"""Measure warm single-query ColQwen2 latency on V19 development queries."""

from __future__ import annotations

import argparse
import json
import math
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

from ocr_vlm_retrieval.runtime.cache import write_json_atomic  # noqa: E402
from ocr_vlm_retrieval.runtime.late_interaction import (  # noqa: E402
    exclusive_process_lock,
    load_embedding,
)
from scripts.evaluate_v19_colqwen2_development import (  # noqa: E402
    build_query_rows,
    maxsim_score_matrix,
    read_json,
)

DEFAULT_ASSIGNMENTS = (
    ROOT
    / "outputs/evaluation/v19/selective_intervention"
    / "development_route_assignments.json"
)
DEFAULT_RETRIEVAL_DIR = (
    ROOT
    / "outputs/evaluation/v19/selective_intervention/retrieval/baseline"
)
DEFAULT_MODEL = ROOT / "models/colqwen2-v1.0-hf"
DEFAULT_INDEX = ROOT / "outputs/user_library/colqwen2_v1_index"
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/evaluation/v19/selective_intervention"
    / "development_colqwen2_warm_latency.json"
)


def percentile(values: list[float], probability: float) -> float:
    """Return a nearest-rank percentile from a non-empty sample."""

    if not values:
        raise ValueError("latency sample must not be empty")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between zero and one")
    ordered = sorted(values)
    rank = max(1, math.ceil(probability * len(ordered)))
    return ordered[rank - 1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--retrieval-dir", type=Path, default=DEFAULT_RETRIEVAL_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--query-count", type=int, default=20)
    parser.add_argument("--passage-batch-size", type=int, default=16)
    parser.add_argument("--min-free-gib", type=float, default=5.0)
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    if args.query_count <= 0 or args.passage_batch_size <= 0:
        raise ValueError("query count and passage batch size must be positive")
    queries, _ = build_query_rows(
        read_json(args.assignments), retrieval_dir=args.retrieval_dir
    )
    if args.query_count > len(queries):
        raise ValueError("query count exceeds available development queries")
    step = len(queries) / args.query_count
    selected_queries = [queries[int(index * step)] for index in range(args.query_count)]
    receipt = read_json(args.index_dir / "index_receipt.json")
    if receipt.get("status") != "complete" or receipt.get("failed_item_count") != 0:
        raise ValueError("a complete failure-free ColQwen2 index is required")
    indexed_item_ids = [str(value) for value in receipt["indexed_item_ids"]]
    shard_dir = args.index_dir / "shards"
    shard_paths = [shard_dir / f"{item_id}.npy" for item_id in indexed_item_ids]
    missing = [path for path in shard_paths if not path.is_file()]
    if missing:
        raise ValueError(f"index missing {len(missing)} required shards")

    import torch
    from transformers import ColQwen2ForRetrieval, ColQwen2Processor

    if not torch.cuda.is_available():
        raise RuntimeError("ColQwen2 latency benchmark requires CUDA")
    free_bytes, _ = torch.cuda.mem_get_info()
    free_gib = free_bytes / (1024**3)
    if free_gib < args.min_free_gib:
        raise RuntimeError(
            f"only {free_gib:.2f} GiB GPU memory is free; "
            f"at least {args.min_free_gib:.2f} GiB is required"
        )
    model = ColQwen2ForRetrieval.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
    ).eval()
    processor = ColQwen2Processor.from_pretrained(args.model, use_fast=True)
    passage_vectors = [
        torch.from_numpy(load_embedding(path, expected_dim=128)).to(
            device=model.device, dtype=torch.bfloat16
        )
        for path in shard_paths
    ]

    def score_one(query: str) -> float:
        started = time.perf_counter()
        inputs = processor(text=[query]).to(model.device)
        with torch.inference_mode():
            query_embedding = model(**inputs).embeddings[0]
            blocks = [
                maxsim_score_matrix(
                    [query_embedding],
                    passage_vectors[
                        start : start + args.passage_batch_size
                    ],
                ).cpu()
                for start in range(
                    0, len(passage_vectors), args.passage_batch_size
                )
            ]
        scores = torch.cat(blocks, dim=1)
        torch.topk(scores[0], k=3)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000
        del inputs, query_embedding, blocks, scores
        return elapsed_ms

    score_one(str(selected_queries[0]["query"]))
    torch.cuda.reset_peak_memory_stats()
    measurements = [
        {
            "query_id": str(row["query_id"]),
            "content_stratum": str(row["content_stratum"]),
            "elapsed_ms": round(score_one(str(row["query"])), 3),
        }
        for row in selected_queries
    ]
    latencies = [float(row["elapsed_ms"]) for row in measurements]
    summary: dict[str, Any] = {
        "status": "development_diagnostic_only",
        "method": "colqwen2_full_index_warm_single_query",
        "split": "development_only",
        "eligible_for_final_claim": False,
        "query_count": len(measurements),
        "indexed_page_count": len(passage_vectors),
        "passage_batch_size": args.passage_batch_size,
        "model_load_and_index_preload_excluded": True,
        "p50_ms": round(percentile(latencies, 0.50), 3),
        "p95_ms": round(percentile(latencies, 0.95), 3),
        "maximum_ms": round(max(latencies), 3),
        "mean_ms": round(sum(latencies) / len(latencies), 3),
        "peak_gpu_gib_during_queries": round(
            torch.cuda.max_memory_allocated() / (1024**3), 3
        ),
        "measurements": measurements,
    }
    write_json_atomic(args.output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    args = parse_args()
    lock_path = args.index_dir / ".colqwen2_gpu_job.lock"
    with exclusive_process_lock(lock_path):
        return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
