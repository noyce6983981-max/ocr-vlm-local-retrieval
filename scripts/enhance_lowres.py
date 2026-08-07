"""Generate reproducible enhancement variants for the low-resolution sample."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enhance one low-resolution image.")
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=Path("data/raw/pilot/download.jpg"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/lowres_variants"),
    )
    return parser.parse_args()


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def write_png(path: Path, image: np.ndarray) -> None:
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise ValueError(f"Cannot encode image: {path}")
    encoded.tofile(path)


def main() -> None:
    args = parse_args()
    input_path = project_path(args.input)
    output_dir = project_path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    image = read_image(input_path)

    up2 = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
    up4 = cv2.resize(image, None, fx=4, fy=4, interpolation=cv2.INTER_LANCZOS4)

    blurred = cv2.GaussianBlur(up4, (0, 0), sigmaX=1.0)
    unsharp = cv2.addWeighted(up4, 1.8, blurred, -0.8, 0)

    gray = cv2.cvtColor(up4, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    clahe_bgr = cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR)

    threshold = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        9,
    )
    threshold_bgr = cv2.cvtColor(threshold, cv2.COLOR_GRAY2BGR)

    variants = {
        "lowres_01_2x_lanczos.png": up2,
        "lowres_02_4x_lanczos.png": up4,
        "lowres_03_4x_unsharp.png": unsharp,
        "lowres_04_4x_clahe.png": clahe_bgr,
        "lowres_05_4x_adaptive_threshold.png": threshold_bgr,
    }
    for filename, variant in variants.items():
        output_path = output_dir / filename
        write_png(output_path, variant)
        print(f"{filename}: {variant.shape[1]}x{variant.shape[0]}")


if __name__ == "__main__":
    main()
