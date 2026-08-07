"""Encode the OCR corpus with BGE-M3 and build a FAISS cosine index."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import torch
from FlagEmbedding import BGEM3FlagModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the BGE-M3 text index.")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/processed/pilot_corpus.jsonl"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/bge-m3"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/text_index"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    return parser.parse_args()


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    args = parse_args()
    corpus_path = project_path(args.corpus)
    model_path = project_path(args.model)
    output_dir = project_path(args.output)

    if not corpus_path.is_file():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access the GPU.")

    rows = read_jsonl(corpus_path)
    if not rows:
        raise ValueError(f"Corpus is empty: {corpus_path}")
    texts = [row["text"] for row in rows]

    print(f"Loading BGE-M3 from {model_path} on {args.device} ...")
    model = BGEM3FlagModel(
        str(model_path),
        use_fp16=args.device.startswith("cuda"),
        devices=args.device,
        batch_size=args.batch_size,
        passage_max_length=args.max_length,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )

    started = time.perf_counter()
    encoded = model.encode(
        texts,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    vectors = np.ascontiguousarray(encoded["dense_vecs"], dtype=np.float32)
    encode_seconds = round(time.perf_counter() - started, 3)

    # BGE-M3 normalizes dense vectors by default, so inner product is cosine.
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "index.faiss"
    metadata_path = output_dir / "metadata.jsonl"
    config_path = output_dir / "config.json"
    # FAISS's Windows C++ file writer cannot reliably open Unicode paths.
    # Serialize in memory and let Python handle the Chinese project path.
    index_path.write_bytes(faiss.serialize_index(index).tobytes())

    with metadata_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    config = {
        "model_path": model_path.relative_to(PROJECT_ROOT).as_posix(),
        "corpus_path": corpus_path.relative_to(PROJECT_ROOT).as_posix(),
        "index_type": "faiss.IndexFlatIP",
        "similarity": "cosine_on_normalized_dense_vectors",
        "embedding_dimension": int(vectors.shape[1]),
        "vector_count": int(vectors.shape[0]),
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "device": args.device,
        "fp16": args.device.startswith("cuda"),
        "encode_seconds": encode_seconds,
    }
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        f"Index built: {index.ntotal} vectors x {index.d} dimensions "
        f"in {encode_seconds:.3f}s."
    )
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()
