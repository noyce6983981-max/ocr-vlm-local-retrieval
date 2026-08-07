"""Download Qwen3-VL-Embedding-2B from the mainland-China ModelScope hub."""

from __future__ import annotations

import argparse
from pathlib import Path

from modelscope import snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download qwen/Qwen3-VL-Embedding-2B."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/qwen3-vl-embedding-2b"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = (
        args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = snapshot_download(
        model_id="qwen/Qwen3-VL-Embedding-2B",
        local_dir=str(output_dir),
        max_workers=4,
    )
    print(f"Qwen3-VL-Embedding-2B downloaded to: {model_path}")


if __name__ == "__main__":
    main()
