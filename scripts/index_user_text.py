"""Incrementally add one uploaded image's OCR text to the BGE-M3 index."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import faiss
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_text_corpus import make_chunks  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--item-id", required=True)
    parser.add_argument("--display-name", required=True)
    parser.add_argument("--category", default="personal_document")
    parser.add_argument("--source-path", required=True)
    parser.add_argument("--ocr-json", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/bge-m3"),
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("outputs/user_library/text_index"),
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_rows(
    item_id: str,
    display_name: str,
    category: str,
    source_path: str,
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    texts = payload.get("rec_texts", [])
    scores = payload.get("rec_scores", [])
    if len(texts) != len(scores):
        raise ValueError("OCR text and confidence counts are inconsistent.")
    boxes = [
        {
            "box_index": index,
            "text": str(text).strip(),
            "confidence": float(score),
        }
        for index, (text, score) in enumerate(zip(texts, scores))
        if str(text).strip()
    ]
    rows: list[dict[str, Any]] = []
    for chunk_index, chunk in enumerate(make_chunks(boxes, 400, 2)):
        rows.append(
            {
                "chunk_id": f"{item_id}_chunk_{chunk_index:03d}",
                "item_id": item_id,
                "display_name_zh": display_name,
                "category": category,
                "source_path": source_path,
                "box_start": chunk[0]["box_index"],
                "box_end": chunk[-1]["box_index"],
                "mean_ocr_confidence": round(
                    statistics.fmean(
                        float(box["confidence"]) for box in chunk
                    ),
                    6,
                ),
                "text": "\n".join(str(box["text"]) for box in chunk),
            }
        )
    return rows


def main() -> None:
    import torch
    from FlagEmbedding import BGEM3FlagModel

    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    index_dir = project_path(args.index_dir)
    metadata_path = index_dir / "metadata.jsonl"
    index_path = index_dir / "index.faiss"
    metadata = read_jsonl(metadata_path)
    if any(row["item_id"] == args.item_id for row in metadata):
        print(f"Text index already contains {args.item_id}.")
        return

    ocr_payload = json.loads(
        project_path(args.ocr_json).read_text(encoding="utf-8")
    )
    rows = build_rows(
        args.item_id,
        args.display_name,
        args.category,
        args.source_path,
        ocr_payload,
    )
    if not rows:
        print(f"No OCR text to index for {args.item_id}.")
        return

    model = BGEM3FlagModel(
        str(project_path(args.model)),
        use_fp16=args.device.startswith("cuda"),
        devices=args.device,
        batch_size=1,
        passage_max_length=512,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    encoded = model.encode(
        [row["text"] for row in rows],
        batch_size=1,
        max_length=512,
    )
    vectors = np.ascontiguousarray(
        encoded["dense_vecs"], dtype=np.float32
    )

    if index_path.is_file():
        serialized = np.frombuffer(index_path.read_bytes(), dtype=np.uint8)
        index = faiss.deserialize_index(serialized)
        if index.ntotal != len(metadata):
            raise ValueError("Existing user text index is inconsistent.")
        if index.d != vectors.shape[1]:
            raise ValueError("Text embedding dimensions do not match.")
    else:
        index = faiss.IndexFlatIP(vectors.shape[1])

    index.add(vectors)
    index_dir.mkdir(parents=True, exist_ok=True)
    index_path.write_bytes(faiss.serialize_index(index).tobytes())
    with metadata_path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Added {len(rows)} OCR chunks for {args.item_id}.")


if __name__ == "__main__":
    main()
