"""Batch-score reviewed library queries with Dense and BM25 retrieval."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import torch
from FlagEmbedding import BGEM3FlagModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.bm25_retrieval import score_bm25
from scripts.live_search import (
    library_revision,
    query_key,
    required_search_branches,
    resolve_search_intent,
    write_json_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path(
            "data/evaluation/public_dataset_1500_retrieval_queries_formal.csv"
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/user_library/manifest.jsonl"),
    )
    parser.add_argument(
        "--text-index",
        type=Path,
        default=Path("outputs/user_library/text_index"),
    )
    parser.add_argument(
        "--bm25-index",
        type=Path,
        default=Path("outputs/user_library/bm25_index"),
    )
    parser.add_argument(
        "--metadata-index",
        type=Path,
        default=Path("outputs/user_library/metadata_index"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/bge-m3"),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=("train", "validation"),
    )
    parser.add_argument(
        "--required-review-status",
        default="已确认",
        help="Exact review_status required for every selected query.",
    )
    parser.add_argument(
        "--live-cache-dir",
        type=Path,
        default=None,
        help="Also materialize product-compatible text and BM25 caches.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/evaluation/library_retrieval/dev_text_bm25_scores.npz"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_queries(
    path: Path,
    splits: set[str],
    required_review_status: str = "已确认",
) -> list[dict[str, str]]:
    if path.suffix.lower() == ".jsonl":
        rows = [dict(row) for row in read_jsonl(path)]
    else:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("split") in splits]
    if not rows:
        raise ValueError(f"No queries found for splits: {sorted(splits)}")
    if any(row.get("review_status") != required_review_status for row in rows):
        raise ValueError(
            f"All scored queries must have review_status={required_review_status!r}."
        )
    return rows


def aggregate_chunks(
    chunk_scores: np.ndarray,
    metadata: list[dict[str, Any]],
    item_columns: dict[str, int],
    *,
    fill_value: float,
) -> np.ndarray:
    document_scores = np.full(
        (chunk_scores.shape[0], len(item_columns)),
        fill_value,
        dtype=np.float32,
    )
    for chunk_index, row in enumerate(metadata):
        column = item_columns[row["item_id"]]
        document_scores[:, column] = np.maximum(
            document_scores[:, column],
            chunk_scores[:, chunk_index],
        )
    return document_scores


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    query_path = project_path(args.queries)
    manifest = [
        row
        for row in read_jsonl(project_path(args.manifest))
        if bool(row.get("search_enabled", True))
    ]
    queries = read_queries(
        query_path,
        set(args.splits),
        args.required_review_status,
    )
    item_ids = [row["item_id"] for row in manifest]
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("Manifest contains duplicate item IDs.")
    item_columns = {item_id: column for column, item_id in enumerate(item_ids)}

    text_index_dir = project_path(args.text_index)
    index = faiss.deserialize_index(
        np.frombuffer(
            (text_index_dir / "index.faiss").read_bytes(),
            dtype=np.uint8,
        )
    )
    text_metadata = read_jsonl(text_index_dir / "metadata.jsonl")
    if index.ntotal != len(text_metadata):
        raise ValueError("Dense index and metadata are inconsistent.")

    started = time.perf_counter()
    print(f"Encoding {len(queries)} confirmed queries with BGE-M3 ...")
    model = BGEM3FlagModel(
        str(project_path(args.model)),
        use_fp16=args.device.startswith("cuda"),
        devices=args.device,
        batch_size=args.batch_size,
        query_max_length=128,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    encoded = model.encode(
        [row["query"] for row in queries],
        batch_size=args.batch_size,
        max_length=128,
    )
    query_vectors = np.ascontiguousarray(encoded["dense_vecs"], dtype=np.float32)
    dense_chunk_scores, dense_chunk_indices = index.search(query_vectors, index.ntotal)
    dense_unsorted = np.full((len(queries), index.ntotal), -1.0, dtype=np.float32)
    row_indices = np.arange(len(queries))[:, None]
    dense_unsorted[row_indices, dense_chunk_indices] = dense_chunk_scores
    dense_scores = aggregate_chunks(
        dense_unsorted,
        text_metadata,
        item_columns,
        fill_value=-1.0,
    )
    dense_seconds = time.perf_counter() - started

    metadata_index_dir = project_path(args.metadata_index)
    metadata_scores: np.ndarray | None = None
    if (metadata_index_dir / "index.faiss").is_file():
        metadata_index = faiss.deserialize_index(
            np.frombuffer(
                (metadata_index_dir / "index.faiss").read_bytes(),
                dtype=np.uint8,
            )
        )
        metadata_rows = read_jsonl(metadata_index_dir / "metadata.jsonl")
        if metadata_index.ntotal != len(metadata_rows):
            raise ValueError("Metadata index and metadata are inconsistent.")
        metadata_chunk_scores, metadata_chunk_indices = metadata_index.search(
            query_vectors, metadata_index.ntotal
        )
        metadata_unsorted = np.full(
            (len(queries), metadata_index.ntotal),
            -1.0,
            dtype=np.float32,
        )
        metadata_unsorted[row_indices, metadata_chunk_indices] = metadata_chunk_scores
        metadata_scores = aggregate_chunks(
            metadata_unsorted,
            metadata_rows,
            item_columns,
            fill_value=-1.0,
        )

    bm25_started = time.perf_counter()
    bm25_index_dir = project_path(args.bm25_index)
    with gzip.open(bm25_index_dir / "index.json.gz", "rt", encoding="utf-8") as handle:
        bm25_payload = json.load(handle)
    bm25_metadata = read_jsonl(bm25_index_dir / "metadata.jsonl")
    if len(bm25_metadata) != int(bm25_payload["document_count"]):
        raise ValueError("BM25 index and metadata are inconsistent.")
    bm25_chunk_scores = np.asarray(
        [score_bm25(row["query"], bm25_payload) for row in queries],
        dtype=np.float32,
    )
    bm25_scores = aggregate_chunks(
        bm25_chunk_scores,
        bm25_metadata,
        item_columns,
        fill_value=0.0,
    )
    bm25_seconds = time.perf_counter() - bm25_started

    output_path = project_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    archive_payload: dict[str, Any] = {
        "dense_scores": dense_scores,
        "bm25_scores": bm25_scores,
        "query_ids": np.asarray([row["query_id"] for row in queries]),
        "item_ids": np.asarray(item_ids),
        "splits": np.asarray([row["split"] for row in queries]),
    }
    if metadata_scores is not None:
        archive_payload["metadata_scores"] = metadata_scores
    np.savez_compressed(output_path, **archive_payload)

    if args.live_cache_dir is not None:
        cache_dir = project_path(args.live_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        revision = library_revision(project_path(args.manifest).parent)
        text_elapsed = round(dense_seconds / len(queries), 3)
        bm25_elapsed = round(bm25_seconds / len(queries), 3)
        for index_value, row in enumerate(queries):
            query = " ".join(row["query"].split())
            route, exploratory, _, _ = resolve_search_intent("quality_hybrid", query)
            branches = required_search_branches("quality_hybrid", route, exploratory)
            key = query_key(query, revision)
            if branches["text"]:
                text_payload: dict[str, Any] = {
                    "query": query,
                    "branch": "text",
                    "library_revision": revision,
                    "item_ids": item_ids,
                    "scores": [
                        round(float(score), 8) for score in dense_scores[index_value]
                    ],
                    "elapsed_seconds": text_elapsed,
                }
                if metadata_scores is not None:
                    text_payload["metadata_scores"] = [
                        round(float(score), 8) for score in metadata_scores[index_value]
                    ]
                write_json_atomic(cache_dir / f"{key}_text.json", text_payload)
            if branches["bm25"]:
                bm25_row = bm25_scores[index_value]
                write_json_atomic(
                    cache_dir / f"{key}_bm25.json",
                    {
                        "query": query,
                        "branch": "bm25",
                        "library_revision": revision,
                        "item_ids": item_ids,
                        "scores": [round(float(score), 8) for score in bm25_row],
                        "matched_documents": int(np.sum(bm25_row > 0)),
                        "elapsed_seconds": bm25_elapsed,
                    },
                )
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "query_file": query_path.relative_to(PROJECT_ROOT).as_posix(),
                "splits": list(args.splits),
                "query_count": len(queries),
                "item_count": len(item_ids),
                "dense_seconds": round(dense_seconds, 3),
                "bm25_seconds": round(bm25_seconds, 3),
                "metadata_branch": metadata_scores is not None,
                "live_cache_dir": (
                    project_path(args.live_cache_dir)
                    .relative_to(PROJECT_ROOT)
                    .as_posix()
                    if args.live_cache_dir is not None
                    else None
                ),
                "output": output_path.relative_to(PROJECT_ROOT).as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved {len(queries)} x {len(item_ids)} Dense/BM25 scores: {output_path}")


if __name__ == "__main__":
    main()
