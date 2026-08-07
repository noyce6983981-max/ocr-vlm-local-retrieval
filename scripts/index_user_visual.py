"""Incrementally add one uploaded image to the Qwen3-VL visual index."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_REPO = PROJECT_ROOT / "third_party/Qwen3-VL-Embedding"
if str(OFFICIAL_REPO) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_REPO))

from src.models.qwen3_vl_embedding import Qwen3VLEmbedder  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--item-id", required=True)
    parser.add_argument("--display-name", required=True)
    parser.add_argument("--category", default="personal_document")
    parser.add_argument("--source-path", required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/qwen3-vl-embedding-2b"),
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("outputs/user_library/visual_index"),
    )
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


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3-VL image encoding requires CUDA.")

    index_dir = project_path(args.index_dir)
    metadata_path = index_dir / "metadata.jsonl"
    embeddings_path = index_dir / "embeddings.npy"
    metadata = read_jsonl(metadata_path)
    if any(row["item_id"] == args.item_id for row in metadata):
        print(f"Visual index already contains {args.item_id}.")
        return

    source_path = project_path(Path(args.source_path))
    with Image.open(source_path) as opened:
        image = opened.convert("RGB")

    torch.cuda.empty_cache()
    model = Qwen3VLEmbedder(
        model_name_or_path=str(project_path(args.model)),
        max_length=512,
        min_pixels=32 * 32 * 4,
        max_pixels=512 * 512,
        dtype=torch.float16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    vector = (
        model.process([{"image": image}])
        .detach()
        .float()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    if embeddings_path.is_file():
        existing = np.load(embeddings_path).astype(np.float32)
        if len(existing) != len(metadata):
            raise ValueError("Existing user visual index is inconsistent.")
        if existing.shape[1] != vector.shape[1]:
            raise ValueError("Visual embedding dimensions do not match.")
        matrix = np.ascontiguousarray(
            np.vstack([existing, vector]), dtype=np.float32
        )
    else:
        matrix = np.ascontiguousarray(vector, dtype=np.float32)

    index_dir.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_path, matrix)
    row = {
        "item_id": args.item_id,
        "display_name_zh": args.display_name,
        "category": args.category,
        "source_path": Path(args.source_path).as_posix(),
    }
    with metadata_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Added visual vector for {args.item_id}.")


if __name__ == "__main__":
    main()
