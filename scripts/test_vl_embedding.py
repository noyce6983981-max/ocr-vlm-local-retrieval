"""Run one guarded Qwen3-VL-Embedding image inference and record GPU memory."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_REPO = PROJECT_ROOT / "third_party/Qwen3-VL-Embedding"
if str(OFFICIAL_REPO) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_REPO))

from src.models.qwen3_vl_embedding import Qwen3VLEmbedder  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-image VL smoke test.")
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=Path(
            "data/raw/pilot/ba2f680c-510b-4733-88a8-a24cb2c1dcd8.png"
        ),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/qwen3-vl-embedding-2b"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/vl_smoke_test"),
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=256 * 256,
        help="Maximum image pixels passed to the visual encoder.",
    )
    return parser.parse_args()


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def gib(value: int) -> float:
    return round(value / 1024**3, 3)


def main() -> None:
    args = parse_args()
    input_path = project_path(args.input)
    model_path = project_path(args.model)
    output_dir = project_path(args.output)

    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if not (model_path / "model.safetensors").is_file():
        raise FileNotFoundError(f"Incomplete model directory: {model_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3-VL smoke test requires a CUDA GPU.")

    output_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(input_path).convert("RGB")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    try:
        model = Qwen3VLEmbedder(
            model_name_or_path=str(model_path),
            max_length=512,
            min_pixels=32 * 32 * 4,
            max_pixels=args.max_pixels,
            dtype=torch.float16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
        load_seconds = time.perf_counter() - load_started

        inference_started = time.perf_counter()
        embedding = model.process(
            [
                {
                    "image": image,
                    "instruction": (
                        "Represent the image for retrieval by a Chinese "
                        "natural-language query."
                    ),
                }
            ]
        )
        torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - inference_started
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        raise RuntimeError(
            "Qwen3-VL-Embedding exceeded 8GB VRAM in the guarded smoke test."
        ) from exc

    vector = embedding.detach().float().cpu().numpy()
    vector_path = output_dir / f"{input_path.stem}_embedding.npy"
    np.save(vector_path, vector)

    summary = {
        "input_path": input_path.relative_to(PROJECT_ROOT).as_posix(),
        "model_path": model_path.relative_to(PROJECT_ROOT).as_posix(),
        "device": torch.cuda.get_device_name(0),
        "dtype": "float16",
        "attention": "sdpa",
        "max_pixels": args.max_pixels,
        "embedding_shape": list(vector.shape),
        "embedding_l2_norm": round(float(np.linalg.norm(vector[0])), 6),
        "model_load_seconds": round(load_seconds, 3),
        "inference_seconds": round(inference_seconds, 3),
        "peak_allocated_gib": gib(torch.cuda.max_memory_allocated()),
        "peak_reserved_gib": gib(torch.cuda.max_memory_reserved()),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Embedding saved to: {vector_path}")


if __name__ == "__main__":
    main()
