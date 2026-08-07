"""Score one arbitrary query against the BGE-M3 OCR text index."""

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifest/dataset_v1_manifest.jsonl"),
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("artifacts/text_index_dataset_v1"),
    )
    parser.add_argument(
        "--extra-manifest",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--extra-index-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--metadata-index-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/bge-m3"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument("--library-revision", default="base")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_text_model(model_path: Path, device: str) -> BGEM3FlagModel:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return BGEM3FlagModel(
        str(project_path(model_path)),
        use_fp16=device.startswith("cuda"),
        devices=device,
        batch_size=1,
        query_max_length=128,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )


def load_text_resources(
    manifest_path: Path,
    index_dir: Path,
    *,
    extra_manifest: Path | None = None,
    extra_index_dir: Path | None = None,
    metadata_index_dir: Path | None = None,
) -> dict[str, Any]:
    manifest = read_jsonl(project_path(manifest_path))
    if extra_manifest is not None:
        extra_manifest_path = project_path(extra_manifest)
        if extra_manifest_path.is_file():
            manifest.extend(read_jsonl(extra_manifest_path))
    item_ids = [row["item_id"] for row in manifest]
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("Combined manifest contains duplicate item IDs.")
    item_columns = {item_id: index for index, item_id in enumerate(item_ids)}
    index_pairs: list[tuple[Any, list[dict[str, Any]]]] = []
    index_dirs = [project_path(index_dir)]
    if extra_index_dir is not None:
        index_dirs.append(project_path(extra_index_dir))
    for current_index_dir in index_dirs:
        index_path = current_index_dir / "index.faiss"
        metadata_path = current_index_dir / "metadata.jsonl"
        if not index_path.is_file() and not metadata_path.is_file():
            continue
        if not index_path.is_file() or not metadata_path.is_file():
            raise ValueError(f"Incomplete text index: {current_index_dir}")
        metadata = read_jsonl(metadata_path)
        index_bytes = np.frombuffer(index_path.read_bytes(), dtype=np.uint8)
        index = faiss.deserialize_index(index_bytes)
        if index.ntotal != len(metadata):
            raise ValueError(f"Inconsistent text index: {current_index_dir}")
        index_pairs.append((index, metadata))
    if not index_pairs:
        raise FileNotFoundError("No text index is available.")

    metadata_index = None
    metadata_rows: list[dict[str, Any]] = []
    if metadata_index_dir is not None:
        resolved_metadata_dir = project_path(metadata_index_dir)
        metadata_index_path = resolved_metadata_dir / "index.faiss"
        metadata_rows_path = resolved_metadata_dir / "metadata.jsonl"
        if metadata_index_path.is_file() or metadata_rows_path.is_file():
            if not metadata_index_path.is_file() or not metadata_rows_path.is_file():
                raise ValueError(
                    f"Incomplete metadata index: {resolved_metadata_dir}"
                )
            metadata_rows = read_jsonl(metadata_rows_path)
            metadata_index = faiss.deserialize_index(
                np.frombuffer(metadata_index_path.read_bytes(), dtype=np.uint8)
            )
            if metadata_index.ntotal != len(metadata_rows):
                raise ValueError(
                    f"Inconsistent metadata index: {resolved_metadata_dir}"
                )
    return {
        "item_ids": item_ids,
        "item_columns": item_columns,
        "index_pairs": index_pairs,
        "metadata_index": metadata_index,
        "metadata_rows": metadata_rows,
    }


def score_text_query_payload(
    query: str,
    model: BGEM3FlagModel,
    resources: dict[str, Any],
    *,
    library_revision: str,
    started_at: float | None = None,
) -> dict[str, Any]:
    query = query.strip()
    if not query:
        raise ValueError("Query cannot be empty.")
    started = started_at if started_at is not None else time.perf_counter()
    item_ids = resources["item_ids"]
    item_columns = resources["item_columns"]
    encoded = model.encode([query], batch_size=1, max_length=128)
    query_vector = np.ascontiguousarray(
        encoded["dense_vecs"], dtype=np.float32
    )
    document_scores = np.full(len(item_ids), -1.0, dtype=np.float32)
    for index, metadata in resources["index_pairs"]:
        chunk_scores, chunk_indices = index.search(
            query_vector, index.ntotal
        )
        for score, metadata_index in zip(
            chunk_scores[0], chunk_indices[0]
        ):
            if metadata_index < 0:
                continue
            item_id = metadata[int(metadata_index)]["item_id"]
            if item_id not in item_columns:
                raise ValueError(
                    f"Text index item is absent from manifest: {item_id}"
                )
            column = item_columns[item_id]
            document_scores[column] = max(
                document_scores[column], float(score)
            )

    metadata_document_scores: np.ndarray | None = None
    metadata_index = resources["metadata_index"]
    metadata_rows = resources["metadata_rows"]
    if metadata_index is not None:
        scores, indices = metadata_index.search(query_vector, metadata_index.ntotal)
        metadata_document_scores = np.full(
            len(item_ids), -1.0, dtype=np.float32
        )
        for score, metadata_index_value in zip(scores[0], indices[0]):
            if metadata_index_value < 0:
                continue
            item_id = metadata_rows[int(metadata_index_value)]["item_id"]
            if item_id not in item_columns:
                raise ValueError(
                    "Metadata index item is absent from manifest: " f"{item_id}"
                )
            metadata_document_scores[item_columns[item_id]] = float(score)

    payload = {
        "query": query,
        "branch": "text",
        "library_revision": library_revision,
        "item_ids": item_ids,
        "scores": [round(float(score), 8) for score in document_scores],
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    if metadata_document_scores is not None:
        payload["metadata_scores"] = [
            round(float(score), 8)
            for score in metadata_document_scores
        ]
    return payload


def main() -> None:
    args = parse_args()
    query = args.query.strip()
    started = time.perf_counter()
    resources = load_text_resources(
        args.manifest,
        args.index_dir,
        extra_manifest=args.extra_manifest,
        extra_index_dir=args.extra_index_dir,
        metadata_index_dir=args.metadata_index_dir,
    )
    model = load_text_model(args.model, args.device)
    payload = score_text_query_payload(
        query,
        model,
        resources,
        library_revision=args.library_revision,
        started_at=started,
    )
    output_path = project_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(output_path)


if __name__ == "__main__":
    main()
