"""Evaluate document-level OCR text retrieval with reviewed query ground truth."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import faiss
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate OCR text retrieval.")
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("data/evaluation/retrieval_queries.csv"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifest/pilot_manifest.jsonl"),
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("artifacts/text_index"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/bge-m3"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/evaluation/text_retrieval_baseline.json"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--allow-unreviewed",
        action="store_true",
        help="Run even when some query rows are not marked 已确认.",
    )
    return parser.parse_args()


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_queries(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "query_id",
        "query",
        "expected_item_id",
        "query_type",
        "review_status",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Query CSV needs columns: {sorted(required)}")
    return rows


def document_ranking(
    scores: np.ndarray,
    indices: np.ndarray,
    metadata: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ranking: list[dict[str, Any]] = []
    seen_items: set[str] = set()

    for score, index in zip(scores.tolist(), indices.tolist()):
        if index < 0:
            continue
        row = metadata[index]
        item_id = row["item_id"]
        if item_id in seen_items:
            continue
        seen_items.add(item_id)
        ranking.append(
            {
                "item_id": item_id,
                "display_name_zh": row["display_name_zh"],
                "chunk_id": row["chunk_id"],
                "score": round(float(score), 6),
            }
        )
    return ranking


def relevant_item_ids(query: dict[str, str]) -> set[str]:
    values = [
        value.strip()
        for value in query.get("relevant_item_ids", "").split(";")
        if value.strip()
    ]
    return set(values or [query["expected_item_id"]])


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


def main() -> None:
    import torch
    from FlagEmbedding import BGEM3FlagModel

    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access the GPU.")

    query_path = project_path(args.queries)
    manifest_path = project_path(args.manifest)
    index_dir = project_path(args.index_dir)
    model_path = project_path(args.model)
    output_path = project_path(args.output)
    queries = read_queries(query_path)
    manifest = read_jsonl(manifest_path)

    unreviewed = [
        row["query_id"] for row in queries if row["review_status"] != "已确认"
    ]
    if unreviewed and not args.allow_unreviewed:
        raise ValueError(
            "The following queries still need human review: "
            + ", ".join(unreviewed)
        )

    index_data = np.frombuffer(
        (index_dir / "index.faiss").read_bytes(), dtype=np.uint8
    )
    index = faiss.deserialize_index(index_data)
    metadata = read_jsonl(index_dir / "metadata.jsonl")
    if index.ntotal != len(metadata):
        raise ValueError(
            f"Index/metadata mismatch: {index.ntotal} vs {len(metadata)}."
        )

    print(f"Loading BGE-M3 and encoding {len(queries)} queries ...")
    model = BGEM3FlagModel(
        str(model_path),
        use_fp16=args.device.startswith("cuda"),
        devices=args.device,
        batch_size=args.batch_size,
        query_max_length=128,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    started = time.perf_counter()
    encoded = model.encode(
        [row["query"] for row in queries],
        batch_size=args.batch_size,
        max_length=128,
    )
    query_vectors = np.ascontiguousarray(
        encoded["dense_vecs"], dtype=np.float32
    )
    scores, indices = index.search(query_vectors, index.ntotal)
    query_encode_seconds = round(time.perf_counter() - started, 3)

    indexed_items = {row["item_id"] for row in metadata}
    item_ids = [row["item_id"] for row in manifest]
    item_columns = {item_id: index for index, item_id in enumerate(item_ids)}
    document_scores = np.full(
        (len(queries), len(item_ids)), -1.0, dtype=np.float32
    )
    for query_index, (score_row, index_row) in enumerate(zip(scores, indices)):
        for score, metadata_index in zip(score_row, index_row):
            if metadata_index < 0:
                continue
            item_id = metadata[int(metadata_index)]["item_id"]
            column = item_columns[item_id]
            document_scores[query_index, column] = max(
                document_scores[query_index, column], float(score)
            )

    results: list[dict[str, Any]] = []
    for query, score_row, index_row in zip(queries, scores, indices):
        ranking = document_ranking(score_row, index_row, metadata)
        expected = query["expected_item_id"]
        relevant = relevant_item_ids(query)
        rank = next(
            (
                rank_index
                for rank_index, item in enumerate(ranking, start=1)
                if item["item_id"] in relevant
            ),
            None,
        )
        results.append(
            {
                **query,
                "expected_item_indexed": bool(relevant & indexed_items),
                "relevant_item_ids_resolved": sorted(relevant),
                "expected_rank": rank,
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
            "retriever": "BAAI/bge-m3 dense retrieval",
            "index": "FAISS IndexFlatIP",
            "query_encode_seconds": query_encode_seconds,
        },
        "overall": aggregate_metrics(results),
        "retrievable_only": aggregate_metrics(
            [result for result in results if result["expected_item_indexed"]]
        ),
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
        output_path.with_name("text_score_matrix.npz"),
        scores=document_scores,
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
        top_1 = result["top_3"][0]["item_id"] if result["top_3"] else "NONE"
        print(
            f"{result['query_id']}: expected={result['expected_item_id']}, "
            f"rank={result['expected_rank']}, top1={top_1}"
        )
    print(f"Report: {output_path}")


if __name__ == "__main__":
    main()
