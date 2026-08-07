"""Tests for OCR quality and category-leak routing."""

from __future__ import annotations

import unittest

from scripts.analyze_library_quality_gate import (
    apply_category_suggestion,
    classify_quality,
    contains_sensitive_text,
    suggest_semantic_category,
)


class QualityGateTests(unittest.TestCase):
    def test_text_in_negative_image_is_routed_to_category_review(self) -> None:
        route, _ = classify_quality(
            has_text_expected=False,
            character_count=12,
            text_box_count=2,
            mean_confidence=0.88,
            privacy_review_required=False,
        )
        self.assertEqual(route, "category_text_leak")

    def test_empty_expected_document_is_routed_to_ocr_retry(self) -> None:
        route, _ = classify_quality(
            has_text_expected=True,
            character_count=0,
            text_box_count=0,
            mean_confidence=0.0,
            privacy_review_required=False,
        )
        self.assertEqual(route, "ocr_retry")

    def test_privacy_review_takes_priority(self) -> None:
        route, _ = classify_quality(
            has_text_expected=True,
            character_count=100,
            text_box_count=10,
            mean_confidence=0.95,
            privacy_review_required=True,
        )
        self.assertEqual(route, "privacy_review")

    def test_privacy_flag_without_sensitive_text_can_continue(self) -> None:
        route, _ = classify_quality(
            has_text_expected=True,
            character_count=100,
            text_box_count=10,
            mean_confidence=0.95,
            privacy_review_required=True,
            sensitive_text_detected=False,
        )
        self.assertEqual(route, "pass")

    def test_sensitive_text_patterns(self) -> None:
        self.assertTrue(contains_sensitive_text("Email: user@example.com"))
        self.assertTrue(contains_sensitive_text("身份证 210123199901011234"))
        self.assertFalse(contains_sensitive_text("普通课程通知"))

    def test_identity_document_category_suggestion(self) -> None:
        self.assertEqual(
            suggest_semantic_category(
                "姓名 朋朋 性别 男 出生 1996年 公民身份号码"
            ),
            "table_form_ticket",
        )
        self.assertEqual(suggest_semantic_category("姓名 李明 性别 男"), "")
        self.assertEqual(suggest_semantic_category("论文实验结果"), "")

    def test_semantic_category_mismatch_is_routed_to_review(self) -> None:
        route, action = apply_category_suggestion(
            route="pass",
            action="无需额外处理",
            current_category="complex_academic",
            suggested_category="table_form_ticket",
        )
        self.assertEqual(route, "category_review")
        self.assertIn("类别", action)


if __name__ == "__main__":
    unittest.main()
