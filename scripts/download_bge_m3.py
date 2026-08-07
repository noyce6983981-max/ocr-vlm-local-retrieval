"""Download the PyTorch BGE-M3 files from the mainland-China ModelScope hub."""

from __future__ import annotations

import argparse
from pathlib import Path

from modelscope import snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download BAAI/bge-m3.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/bge-m3"),
        help="Local model directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = (
        args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = snapshot_download(
        model_id="BAAI/bge-m3",
        local_dir=str(output_dir),
        ignore_patterns=[
            "onnx/*",
            "imgs/*",
            "*.jpg",
            "*.webp",
        ],
        max_workers=4,
    )
    print(f"BGE-M3 downloaded to: {model_path}")


if __name__ == "__main__":
    main()
