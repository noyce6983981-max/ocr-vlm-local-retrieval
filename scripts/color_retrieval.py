"""Deterministic color evidence for multimodal image retrieval."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COLOR_NAMES = (
    "red",
    "orange",
    "yellow",
    "green",
    "cyan",
    "blue",
    "purple",
    "pink",
    "brown",
    "black",
    "white",
    "gray",
)
CACHE_VERSION = 1


def _hsv_channels(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    maximum = rgb.max(axis=2)
    minimum = rgb.min(axis=2)
    delta = maximum - minimum
    saturation = delta / np.maximum(maximum, 1e-6)
    hue = np.zeros_like(maximum)
    chromatic = delta > 1e-6
    red_max = chromatic & (maximum == rgb[:, :, 0])
    green_max = chromatic & (maximum == rgb[:, :, 1])
    blue_max = chromatic & (maximum == rgb[:, :, 2])
    hue[red_max] = (
        60.0
        * ((rgb[:, :, 1][red_max] - rgb[:, :, 2][red_max]) / delta[red_max])
    ) % 360.0
    hue[green_max] = 60.0 * (
        (rgb[:, :, 2][green_max] - rgb[:, :, 0][green_max])
        / delta[green_max]
        + 2.0
    )
    hue[blue_max] = 60.0 * (
        (rgb[:, :, 0][blue_max] - rgb[:, :, 1][blue_max])
        / delta[blue_max]
        + 4.0
    )
    return hue, saturation, maximum


def color_coverages(image_path: Path) -> np.ndarray:
    """Measure how much of an image is visibly occupied by each color."""
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail((128, 128))
        rgb = np.asarray(image, dtype=np.float32) / 255.0
    hue, saturation, value = _hsv_channels(rgb)
    visible = value > 0.08
    chromatic = visible & (saturation > 0.14)

    masks = {
        "red": chromatic & ((hue < 15.0) | (hue >= 345.0)),
        "orange": chromatic & (hue >= 15.0) & (hue < 45.0),
        "yellow": chromatic & (hue >= 45.0) & (hue < 75.0),
        "green": chromatic & (hue >= 75.0) & (hue < 165.0),
        "cyan": chromatic & (hue >= 165.0) & (hue < 200.0),
        "blue": chromatic & (hue >= 200.0) & (hue < 260.0),
        "purple": chromatic & (hue >= 260.0) & (hue < 300.0),
        "pink": chromatic & (hue >= 300.0) & (hue < 345.0),
        "brown": (
            visible
            & (saturation > 0.20)
            & (hue >= 15.0)
            & (hue < 55.0)
            & (value < 0.72)
        ),
        "black": value < 0.20,
        "white": (saturation < 0.12) & (value > 0.82),
        "gray": (
            (saturation < 0.12)
            & (value >= 0.20)
            & (value <= 0.82)
        ),
    }
    return np.asarray(
        [float(masks[name].mean()) for name in COLOR_NAMES],
        dtype=np.float32,
    )


def _source_path(row: dict[str, Any]) -> Path:
    path = Path(str(row.get("source_path", "")))
    return path if path.is_absolute() else PROJECT_ROOT / path


def build_color_index(
    library_dir: Path,
    manifest: list[dict[str, Any]],
) -> tuple[list[str], np.ndarray]:
    """Build or reuse a lightweight local color index aligned to item IDs."""
    item_ids = [str(row["item_id"]) for row in manifest]
    index_dir = library_dir / "color_index"
    index_path = index_dir / "features_v1.npz"
    if index_path.is_file():
        try:
            archive = np.load(index_path, allow_pickle=False)
            cached_ids = archive["item_ids"].astype(str).tolist()
            features = archive["features"].astype(np.float32)
            if cached_ids == item_ids and features.shape == (
                len(item_ids),
                len(COLOR_NAMES),
            ):
                return item_ids, features
        except (OSError, KeyError, ValueError):
            pass

    features = np.zeros(
        (len(manifest), len(COLOR_NAMES)), dtype=np.float32
    )
    for index, row in enumerate(manifest):
        image_path = _source_path(row)
        if not image_path.is_file():
            continue
        try:
            features[index] = color_coverages(image_path)
        except (OSError, ValueError):
            continue
    index_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        index_path,
        cache_version=np.asarray([CACHE_VERSION], dtype=np.int16),
        item_ids=np.asarray(item_ids),
        color_names=np.asarray(COLOR_NAMES),
        features=features,
    )
    return item_ids, features


def color_query_scores(
    color_name: str | None,
    library_dir: Path,
    manifest: list[dict[str, Any]],
) -> np.ndarray:
    if color_name not in COLOR_NAMES:
        return np.zeros(len(manifest), dtype=np.float32)
    _, features = build_color_index(library_dir, manifest)
    return features[:, COLOR_NAMES.index(str(color_name))]
