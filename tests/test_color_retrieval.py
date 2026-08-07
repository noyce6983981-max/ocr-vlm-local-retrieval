from __future__ import annotations

from tempfile import TemporaryDirectory
from pathlib import Path

from PIL import Image

from scripts.color_retrieval import COLOR_NAMES, color_query_scores


def test_blue_query_prefers_blue_image_over_red_image() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        blue_path = root / "blue.png"
        red_path = root / "red.png"
        Image.new("RGB", (32, 32), (25, 80, 230)).save(blue_path)
        Image.new("RGB", (32, 32), (230, 35, 25)).save(red_path)
        manifest = [
            {"item_id": "blue", "source_path": str(blue_path)},
            {"item_id": "red", "source_path": str(red_path)},
        ]
        scores = color_query_scores("blue", root, manifest)
        assert scores[0] > 0.95
        assert scores[1] < 0.05


def test_unknown_color_returns_zero_scores_without_index() -> None:
    with TemporaryDirectory() as directory:
        scores = color_query_scores("not-a-color", Path(directory), [])
        assert scores.size == 0
        assert not (Path(directory) / "color_index").exists()


def test_color_index_has_expected_feature_names() -> None:
    assert {"red", "green", "blue", "black", "white"}.issubset(
        COLOR_NAMES
    )
