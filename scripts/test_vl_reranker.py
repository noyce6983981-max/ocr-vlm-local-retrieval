"""Smoke-test Qwen3-VL-Reranker-2B on two local images."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_REPO = PROJECT_ROOT / "third_party/Qwen3-VL-Embedding"
if str(OFFICIAL_REPO) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_REPO))

from src.models.qwen3_vl_reranker import Qwen3VLReranker  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/qwen3-vl-reranker-2b"),
    )
    parser.add_argument(
        "--positive",
        type=Path,
        default=Path(
            "data/raw/expansion/12_natural_no_text_seaside_sunset.png"
        ),
    )
    parser.add_argument(
        "--negative",
        type=Path,
        default=Path(
            "data/raw/expansion/10_natural_no_text_mountain_lake.png"
        ),
    )
    parser.add_argument(
        "--query",
        default="哪张图片展示海边日落和水面倒影？",
    )
    return parser.parse_args()


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def gib(value: int) -> float:
    return round(value / 1024**3, 3)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the reranker smoke test.")
    model_path = project_path(args.model)
    positive = project_path(args.positive)
    negative = project_path(args.negative)
    for path in (model_path, positive, negative):
        if not path.exists():
            raise FileNotFoundError(path)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = Qwen3VLReranker(
        model_name_or_path=str(model_path),
        max_length=1024,
        min_pixels=32 * 32 * 4,
        max_pixels=384 * 384,
        dtype=torch.float16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    load_seconds = time.perf_counter() - started
    inference_started = time.perf_counter()
    scores = model.process(
        {
            "instruction": (
                "Judge which image best matches the user's Chinese "
                "natural-language retrieval query."
            ),
            "query": {"text": args.query},
            "documents": [
                {"image": str(positive)},
                {"image": str(negative)},
            ],
        }
    )
    torch.cuda.synchronize()
    payload = {
        "query": args.query,
        "positive": positive.name,
        "negative": negative.name,
        "scores": [round(float(score), 6) for score in scores],
        "positive_ranked_first": bool(scores[0] > scores[1]),
        "load_seconds": round(load_seconds, 3),
        "inference_seconds": round(
            time.perf_counter() - inference_started, 3
        ),
        "peak_allocated_gib": gib(torch.cuda.max_memory_allocated()),
        "peak_reserved_gib": gib(torch.cuda.max_memory_reserved()),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
