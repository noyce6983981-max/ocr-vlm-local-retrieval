"""Build a searchable BGE-M3 index from real file/source metadata."""

from __future__ import annotations

import argparse
import json
import re
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
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/user_library/manifest.jsonl"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/bge-m3"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/user_library/metadata_index"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def clean_metadata_value(value: Any) -> str:
    if isinstance(value, list):
        value = " ".join(str(item) for item in value)
    text = str(value or "").strip()
    text = re.sub(r"^File:", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\.(?:jpe?g|png|webp|tiff?|bmp)$", "", text, flags=re.I)
    text = text.replace("_", " ").replace("-", " ")
    return " ".join(text.split())


def metadata_text(row: dict[str, Any]) -> str:
    fields = (
        ("页面名称", row.get("display_name_zh")),
        ("公开原文件名", row.get("public_source_file")),
        ("导入文件名", row.get("source_file_name")),
        ("来源", row.get("source")),
        ("公开数据源", row.get("public_source_name")),
        ("来源标签", row.get("source_tags")),
    )
    parts = [
        f"{label}：{cleaned}"
        for label, value in fields
        if (cleaned := clean_metadata_value(value))
    ]
    return "；".join(parts)


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    rows = [
        row
        for row in read_jsonl(project_path(args.manifest))
        if bool(row.get("search_enabled", True))
    ]
    texts = [metadata_text(row) for row in rows]
    if not rows or any(not text for text in texts):
        raise ValueError("Every active page needs non-empty source metadata.")

    print(f"Encoding real metadata for {len(rows)} active pages ...")
    started = time.perf_counter()
    model = BGEM3FlagModel(
        str(project_path(args.model)),
        use_fp16=args.device.startswith("cuda"),
        devices=args.device,
        batch_size=args.batch_size,
        query_max_length=256,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    encoded = model.encode(
        texts,
        batch_size=args.batch_size,
        max_length=256,
    )
    vectors = np.ascontiguousarray(
        encoded["dense_vecs"], dtype=np.float32
    )
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "index.faiss").write_bytes(
        faiss.serialize_index(index).tobytes()
    )
    with (output_dir / "metadata.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        for row, text in zip(rows, texts):
            handle.write(
                json.dumps(
                    {
                        "item_id": row["item_id"],
                        "metadata_text": text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "model": "BAAI/bge-m3",
                "source": "real manifest and public-source metadata",
                "synthetic_data_count": 0,
                "item_count": len(rows),
                "embedding_dimension": int(vectors.shape[1]),
                "similarity": "cosine_on_normalized_dense_vectors",
                "encode_seconds": round(
                    time.perf_counter() - started, 3
                ),
                "fields": [
                    "display_name_zh",
                    "public_source_file",
                    "source_file_name",
                    "source",
                    "public_source_name",
                    "source_tags",
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(output_dir)


if __name__ == "__main__":
    main()
