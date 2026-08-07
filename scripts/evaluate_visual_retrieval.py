"""Evaluate text-to-image retrieval with Qwen3-VL-Embedding-2B."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_REPO = PROJECT_ROOT / "third_party/Qwen3-VL-Embedding"
if str(OFFICIAL_REPO) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_REPO))

from src.models.qwen3_vl_embedding import Qwen3VLEmbedder  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate visual retrieval.")
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("data/evaluation/retrieval_queries.csv"),
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("artifacts/visual_index"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/qwen3-vl-embedding-2b"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/evaluation/visual_retrieval_baseline.json"),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--allow-unreviewed",
        action="store_true",
        help="Diagnostic only: allow query rows not marked 已确认.",
    )
    return parser.parse_args()


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_queries(
    path: Path, allow_unreviewed: bool = False
) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if (
        not allow_unreviewed
        and any(row["review_status"] != "已确认" for row in rows)
    ):
        raise ValueError("All evaluation queries must be marked 已确认.")
    return rows


def aggregate_metrics(results: list[dict[str, Any]]) -> dict[str, float]:
    ranks = [result["expected_rank"] for result in results]
    return {
        "query_count": len(results),
        "recall_at_1": round(sum(rank == 1 for rank in ranks) / len(ranks), 6),
        "recall_at_3": round(
            sum(rank is not None and rank <= 3 for rank in ranks) / len(ranks), 6
        ),
        "mrr": round(
            statistics.fmean(1.0 / rank if rank else 0.0 for rank in ranks), 6
        ),
    }


def relevant_item_ids(query: dict[str, str]) -> set[str]:
    values = [
        value.strip()
        for value in query.get("relevant_item_ids", "").split(";")
        if value.strip()
    ]
    return set(values or [query["expected_item_id"]])


def main() -> None:
    args = parse_args()
    query_path = project_path(args.queries)
    index_dir = project_path(args.index_dir)
    model_path = project_path(args.model)
    output_path = project_path(args.output)
    queries = read_queries(query_path, args.allow_unreviewed)
    metadata = read_jsonl(index_dir / "metadata.jsonl")
    image_vectors = np.load(index_dir / "embeddings.npy").astype(np.float32)
    if len(metadata) != len(image_vectors):
        raise ValueError("Visual embedding/metadata count mismatch.")
    if not torch.cuda.is_available():
        raise RuntimeError("Visual evaluation requires a CUDA GPU.")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print(f"Loading Qwen3-VL-Embedding-2B and encoding {len(queries)} queries ...")
    model = Qwen3VLEmbedder(
        model_name_or_path=str(model_path),
        max_length=512,
        min_pixels=32 * 32 * 4,
        max_pixels=512 * 512,
        dtype=torch.float16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )

    query_vectors: list[np.ndarray] = []
    started = time.perf_counter()
    for start in range(0, len(queries), args.batch_size):
        batch = queries[start : start + args.batch_size]
        batch_embeddings = model.process(
            [
                {
                    "text": row["query"],
                    "instruction": (
                        "Retrieve the image that best matches the user's "
                        "Chinese natural-language query."
                    ),
                }
                for row in batch
            ]
        )
        query_vectors.append(batch_embeddings.detach().float().cpu().numpy())
    torch.cuda.synchronize()
    query_encode_seconds = time.perf_counter() - started

    query_matrix = np.ascontiguousarray(
        np.vstack(query_vectors), dtype=np.float32
    )
    score_matrix = query_matrix @ image_vectors.T
    item_ids = [row["item_id"] for row in metadata]

    results: list[dict[str, Any]] = []
    for query, scores in zip(queries, score_matrix):
        order = np.argsort(-scores)
        ranking = [
            {
                "item_id": metadata[index]["item_id"],
                "display_name_zh": metadata[index]["display_name_zh"],
                "score": round(float(scores[index]), 6),
            }
            for index in order
        ]
        relevant = relevant_item_ids(query)
        expected_position = next(
            rank
            for rank, index in enumerate(order, start=1)
            if item_ids[int(index)] in relevant
        )
        results.append(
            {
                **query,
                "expected_rank": expected_position,
                "relevant_item_ids_resolved": sorted(relevant),
                "top_3": ranking[:3],
            }
        )

    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_type[result["query_type"]].append(result)

    report = {
        "evaluation_status": (
            "diagnostic_unreviewed"
            if args.allow_unreviewed
            else "formal_reviewed"
        ),
        "system": {
            "retriever": "Qwen3-VL-Embedding-2B text-to-image retrieval",
            "embedding_dimension": int(image_vectors.shape[1]),
            "query_encode_seconds": round(query_encode_seconds, 3),
            "peak_allocated_gib": round(
                torch.cuda.max_memory_allocated() / 1024**3, 3
            ),
            "peak_reserved_gib": round(
                torch.cuda.max_memory_reserved() / 1024**3, 3
            ),
        },
        "overall": aggregate_metrics(results),
        "by_query_type": {
            query_type: aggregate_metrics(type_results)
            for query_type, type_results in sorted(by_type.items())
        },
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        output_path.with_name("visual_score_matrix.npz"),
        scores=score_matrix,
        query_ids=np.array([row["query_id"] for row in queries]),
        item_ids=np.array(item_ids),
    )

    metrics = report["overall"]
    print(
        f"Recall@1={metrics['recall_at_1']:.4f}, "
        f"Recall@3={metrics['recall_at_3']:.4f}, "
        f"MRR={metrics['mrr']:.4f}"
    )
    for result in results:
        print(
            f"{result['query_id']}: expected={result['expected_item_id']}, "
            f"rank={result['expected_rank']}, "
            f"top1={result['top_3'][0]['item_id']}"
        )
    print(f"Report: {output_path}")


if __name__ == "__main__":
    main()
