"""Search the OCR text index with a natural-language query."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import torch
from FlagEmbedding import BGEM3FlagModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search the BGE-M3 OCR index.")
    parser.add_argument("query", help="Natural-language search query.")
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
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--allow-duplicate-docs",
        action="store_true",
        help="Allow several chunks from the same source image.",
    )
    return parser.parse_args()


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select_results(
    scores: np.ndarray,
    indices: np.ndarray,
    metadata: list[dict[str, Any]],
    top_k: int,
    unique_documents: bool,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    seen_items: set[str] = set()

    for score, index in zip(scores.tolist(), indices.tolist()):
        if index < 0:
            continue
        row = metadata[index]
        if unique_documents and row["item_id"] in seen_items:
            continue
        seen_items.add(row["item_id"])
        results.append({"score": float(score), **row})
        if len(results) >= top_k:
            break
    return results


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive.")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access the GPU.")

    index_dir = project_path(args.index_dir)
    model_path = project_path(args.model)
    # Read through Python because FAISS's Windows file API can fail on
    # non-ASCII project paths.
    index_bytes = (index_dir / "index.faiss").read_bytes()
    index = faiss.deserialize_index(
        np.frombuffer(index_bytes, dtype=np.uint8)
    )
    metadata = read_jsonl(index_dir / "metadata.jsonl")
    if index.ntotal != len(metadata):
        raise ValueError(
            f"Index/metadata mismatch: {index.ntotal} vs {len(metadata)}."
        )

    model = BGEM3FlagModel(
        str(model_path),
        use_fp16=args.device.startswith("cuda"),
        devices=args.device,
        batch_size=1,
        query_max_length=128,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    encoded = model.encode(
        [args.query],
        batch_size=1,
        max_length=128,
    )
    query_vector = np.ascontiguousarray(
        encoded["dense_vecs"], dtype=np.float32
    )

    search_k = min(index.ntotal, max(args.top_k, args.top_k * 5))
    scores, indices = index.search(query_vector, search_k)
    results = select_results(
        scores[0],
        indices[0],
        metadata,
        args.top_k,
        unique_documents=not args.allow_duplicate_docs,
    )

    print(f"Query: {args.query}")
    for rank, result in enumerate(results, start=1):
        preview = " ".join(result["text"].split())[:180]
        print(
            f"{rank}. {result['display_name_zh']} "
            f"[{result['item_id']}] score={result['score']:.4f}\n"
            f"   {preview}"
        )


if __name__ == "__main__":
    main()
