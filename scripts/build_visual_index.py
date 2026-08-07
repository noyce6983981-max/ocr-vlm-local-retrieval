"""Encode all manifest images with Qwen3-VL-Embedding-2B."""

from __future__ import annotations

import argparse
import json
import sys
import time
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
    parser = argparse.ArgumentParser(description="Build the visual embedding index.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifest/pilot_manifest.jsonl"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/qwen3-vl-embedding-2b"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/visual_index"),
    )
    parser.add_argument("--max-pixels", type=int, default=512 * 512)
    return parser.parse_args()


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def gib(value: int) -> float:
    return round(value / 1024**3, 3)


def main() -> None:
    args = parse_args()
    manifest_path = project_path(args.manifest)
    model_path = project_path(args.model)
    output_dir = project_path(args.output)
    items = read_jsonl(manifest_path)

    if not torch.cuda.is_available():
        raise RuntimeError("Visual indexing requires a CUDA GPU.")
    if not (model_path / "model.safetensors").is_file():
        raise FileNotFoundError(f"Incomplete model directory: {model_path}")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print(f"Loading Qwen3-VL-Embedding-2B from {model_path} ...")
    load_started = time.perf_counter()
    model = Qwen3VLEmbedder(
        model_name_or_path=str(model_path),
        max_length=512,
        min_pixels=32 * 32 * 4,
        max_pixels=args.max_pixels,
        dtype=torch.float16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model_load_seconds = time.perf_counter() - load_started

    vectors: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    encoding_started = time.perf_counter()

    for index, item in enumerate(items, start=1):
        source_path = project_path(Path(item["source_path"]))
        with Image.open(source_path) as opened:
            image = opened.convert("RGB")

        item_started = time.perf_counter()
        embedding = model.process([{"image": image}])
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - item_started
        vector = embedding.detach().float().cpu().numpy()[0]
        vectors.append(vector)
        metadata.append(
            {
                "item_id": item["item_id"],
                "display_name_zh": item["display_name_zh"],
                "category": item["category"],
                "source_path": item["source_path"],
                "encoding_seconds": round(elapsed, 3),
            }
        )
        print(
            f"[{index:02d}/{len(items):02d}] {item['item_id']}: "
            f"{elapsed:.3f}s"
        )

    matrix = np.ascontiguousarray(np.vstack(vectors), dtype=np.float32)
    total_encoding_seconds = time.perf_counter() - encoding_started
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "embeddings.npy", matrix)
    with (output_dir / "metadata.jsonl").open("w", encoding="utf-8") as handle:
        for row in metadata:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    config = {
        "model": "Qwen/Qwen3-VL-Embedding-2B",
        "model_path": model_path.relative_to(PROJECT_ROOT).as_posix(),
        "manifest_path": manifest_path.relative_to(PROJECT_ROOT).as_posix(),
        "image_count": len(items),
        "embedding_dimension": int(matrix.shape[1]),
        "normalized": True,
        "dtype": "float16",
        "attention": "sdpa",
        "max_pixels": args.max_pixels,
        "model_load_seconds": round(model_load_seconds, 3),
        "total_encoding_seconds": round(total_encoding_seconds, 3),
        "peak_allocated_gib": gib(torch.cuda.max_memory_allocated()),
        "peak_reserved_gib": gib(torch.cuda.max_memory_reserved()),
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        f"Visual index built: {matrix.shape[0]} x {matrix.shape[1]}, "
        f"encoding={total_encoding_seconds:.3f}s, "
        f"peak_reserved={config['peak_reserved_gib']:.3f}GiB"
    )


if __name__ == "__main__":
    main()
