"""Tests for uploaded-image metadata and OCR processing helpers."""

from __future__ import annotations

import unittest

from scripts.index_user_text import build_rows
from scripts.ingest_image import content_item_id, ocr_metrics


class UserLibraryTests(unittest.TestCase):
    def test_content_item_id_is_stable_and_content_based(self) -> None:
        self.assertEqual(content_item_id(b"same"), content_item_id(b"same"))
        self.assertNotEqual(
            content_item_id(b"first"), content_item_id(b"second")
        )
        self.assertTrue(content_item_id(b"image").startswith("user_"))

    def test_ocr_metrics_handles_text_and_empty_results(self) -> None:
        metrics = ocr_metrics(
            {"rec_texts": ["测试", "文本"], "rec_scores": [0.8, 1.0]}
        )
        self.assertEqual(metrics["text_box_count"], 2)
        self.assertEqual(metrics["character_count"], 4)
        self.assertAlmostEqual(metrics["mean_confidence"], 0.9)
        self.assertEqual(
            ocr_metrics({"rec_texts": [], "rec_scores": []})[
                "mean_confidence"
            ],
            0.0,
        )

    def test_build_rows_keeps_upload_metadata(self) -> None:
        rows = build_rows(
            "user_test",
            "测试资料",
            "personal_document",
            "outputs/user_library/images/test.png",
            {"rec_texts": ["第一行", "第二行"], "rec_scores": [0.9, 0.8]},
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["item_id"], "user_test")
        self.assertEqual(rows[0]["display_name_zh"], "测试资料")
        self.assertIn("第一行", rows[0]["text"])


if __name__ == "__main__":
    unittest.main()
