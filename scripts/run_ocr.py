"""Run the reproducible PaddleOCR baseline on one image or an image directory."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Prefer Baidu Object Storage in mainland China instead of Hugging Face.
os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "BOS")

# In the dedicated OCR environment, ModelScope uses CPU-only PyTorch while
# PaddlePaddle uses the GPU. Import CPU PyTorch first, then Paddle, to keep
# Windows DLL initialization deterministic.
import torch  # noqa: F401
import paddle  # noqa: F401
from paddleocr import PaddleOCR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PaddleOCR on one image or an image directory."
    )
    parser.add_argument("input", type=Path, help="Path to an image or directory.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/ocr"),
        help="Directory for JSON and visualization results.",
    )
    parser.add_argument("--device", default="gpu:0", help="Inference device.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_dir = args.output.resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    ocr = PaddleOCR(
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="PP-OCRv5_mobile_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device=args.device,
        engine="paddle",
    )

    results = ocr.predict(str(input_path))
    result_count = 0
    for result in results:
        result.print()
        result.save_to_img(str(output_dir))
        result.save_to_json(str(output_dir))
        result_count += 1

    print(f"OCR completed: {result_count} result(s). Saved to: {output_dir}")


if __name__ == "__main__":
    main()
