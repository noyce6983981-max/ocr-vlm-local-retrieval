from __future__ import annotations

import unittest
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from scripts.audit_public_dataset_200 import (
    category_review_required,
    difference_hash,
)
from scripts.collect_public_dataset_200 import CATEGORY_SPECS
from scripts.apply_public_dataset_metadata import (
    enrich_manifest,
    parse_bool,
)
from scripts.build_public_dataset_splits import (
    CATEGORY_TARGETS,
    build_split_rows,
    validate_splits,
)
from scripts.build_formal_query_set import build_formal_rows
from scripts.analyze_public_dataset_quality import (
    quality_features,
    select_review_queue,
)
from scripts.evaluate_selective_rotation import (
    rotation_utility,
    should_accept_rotation,
)
from scripts.build_text_corpus import resolve_ocr_path
from scripts.generate_public_dataset_queries import (
    choose_evidence,
    query_from_evidence,
    valid_evidence_line,
)
from scripts.prioritize_query_review import priority_score
from scripts.query_review import (
    read_reviews,
    save_review,
    split_item_ids,
    validate_review,
)
from scripts.train_learned_gate import (
    feature_vector,
    normalize_rows as normalize_gate_rows,
    retrieval_metrics as gate_retrieval_metrics,
)
from scripts.train_preference_gate import modality_preference_labels
from scripts.evaluate_text_retrieval import (
    relevant_item_ids as text_relevant_item_ids,
)


class PublicDatasetToolTests(unittest.TestCase):
    def test_build_formal_rows_requires_every_review(self) -> None:
        queue = [
            {
                "query_id": "q001",
                "expected_item_id": "item_a",
                "query_type": "visual",
                "split": "test",
                "category": "scene",
            }
        ]
        with self.assertRaisesRegex(ValueError, "还有 1 条"):
            build_formal_rows(queue, {})

    def test_build_formal_rows_keeps_multi_positive_truth(self) -> None:
        queue = [
            {
                "query_id": "q001",
                "expected_item_id": "item_a",
                "query_type": "visual",
                "split": "test",
                "category": "scene",
            },
            {
                "query_id": "q002",
                "expected_item_id": "item_c",
                "query_type": "visual",
                "split": "validation",
                "category": "scene",
            },
        ]
        reviews = {
            "q001": {
                "decision": "revised",
                "reviewed_query": "查找蓝色建筑",
                "relevant_item_ids": "item_b;item_a",
                "human_notes": "两张近似图都正确",
            },
            "q002": {
                "decision": "excluded",
                "reviewed_query": "",
                "relevant_item_ids": "",
                "human_notes": "无法公平描述",
            },
        }
        rows = build_formal_rows(queue, reviews)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["expected_item_id"], "item_a")
        self.assertEqual(
            rows[0]["relevant_item_ids"], "item_b;item_a"
        )
        self.assertEqual(rows[0]["review_status"], "已确认")

    def test_build_formal_rows_rejects_privacy_risk(self) -> None:
        queue = [
            {
                "query_id": "q001",
                "expected_item_id": "item_a",
                "query_type": "visual",
                "split": "test",
                "category": "document",
                "diagnostic_review_reasons": "privacy",
            }
        ]
        reviews = {
            "q001": {
                "decision": "accepted",
                "reviewed_query": "查找证件页面",
                "relevant_item_ids": "item_a",
                "human_notes": "",
            }
        }
        with self.assertRaisesRegex(ValueError, "隐私风险"):
            build_formal_rows(queue, reviews)

    def test_build_formal_rows_rejects_duplicate_queries(self) -> None:
        queue = [
            {
                "query_id": query_id,
                "expected_item_id": item_id,
                "query_type": "visual",
                "split": "test",
                "category": "scene",
                "diagnostic_review_reasons": "",
            }
            for query_id, item_id in (
                ("q001", "item_a"),
                ("q002", "item_b"),
            )
        ]
        reviews = {
            query_id: {
                "decision": "accepted",
                "reviewed_query": "查找同一内容",
                "relevant_item_ids": item_id,
                "human_notes": "",
            }
            for query_id, item_id in (
                ("q001", "item_a"),
                ("q002", "item_b"),
            )
        }
        with self.assertRaisesRegex(ValueError, "重复查询"):
            build_formal_rows(queue, reviews)

    def test_query_review_validation_and_upsert(self) -> None:
        known_ids = {"item_a", "item_b"}
        query, relevant = validate_review(
            "revised",
            "  哪张   图片？ ",
            "item_a,item_b;item_a",
            known_ids,
        )
        self.assertEqual(query, "哪张 图片？")
        self.assertEqual(relevant, "item_a;item_b")

        with TemporaryDirectory() as directory:
            path = Path(directory) / "reviews.csv"
            fixed_time = datetime(
                2026, 7, 29, 8, 0, tzinfo=timezone.utc
            )
            save_review(
                path,
                {
                    "query_id": "q001",
                    "decision": "revised",
                    "reviewed_query": query,
                    "relevant_item_ids": relevant,
                    "human_notes": "人工确认",
                },
                now=fixed_time,
            )
            reviews = read_reviews(path)
            self.assertEqual(reviews["q001"]["decision"], "revised")
            self.assertEqual(
                reviews["q001"]["reviewed_at"],
                "2026-07-29T08:00:00+00:00",
            )

    def test_query_review_rejects_unknown_item(self) -> None:
        with self.assertRaisesRegex(ValueError, "未知图片ID"):
            validate_review(
                "accepted",
                "查询",
                "item_unknown",
                {"item_a"},
            )

    def test_excluded_query_can_be_blank(self) -> None:
        query, relevant = validate_review(
            "excluded", "", "", {"item_a"}
        )
        self.assertEqual((query, relevant), ("", ""))

    def test_no_answer_query_requires_query_and_empty_truth(self) -> None:
        query, relevant = validate_review(
            "no_answer", "查找库中不存在的火星表格", "", {"item_a"}
        )
        self.assertEqual(query, "查找库中不存在的火星表格")
        self.assertEqual(relevant, "")
        with self.assertRaisesRegex(ValueError, "不能填写"):
            validate_review(
                "no_answer",
                "查找库中不存在的火星表格",
                "item_a",
                {"item_a"},
            )

    def test_build_formal_rows_keeps_no_answer_truth(self) -> None:
        queue = [
            {
                "query_id": "q_open",
                "expected_item_id": "",
                "query_type": "no_answer",
                "split": "test",
                "category": "open_set_no_answer",
                "diagnostic_review_reasons": "no_answer",
            }
        ]
        reviews = {
            "q_open": {
                "decision": "no_answer",
                "reviewed_query": "查找不存在的紫色热气球",
                "relevant_item_ids": "",
                "human_notes": "",
            }
        }
        rows = build_formal_rows(queue, reviews)
        self.assertEqual(rows[0]["expected_item_id"], "")
        self.assertEqual(rows[0]["relevant_item_ids"], "")
        self.assertEqual(rows[0]["is_no_answer"], "True")

    def test_split_item_ids_preserves_order_and_uniqueness(self) -> None:
        self.assertEqual(
            split_item_ids("item_b; item_a,item_b"),
            ["item_b", "item_a"],
        )

    def test_category_plan_has_exactly_200_pages(self) -> None:
        self.assertEqual(sum(CATEGORY_SPECS.values()), 200)

    def test_difference_hash_is_stable(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "sample.png"
            Image.new("RGB", (32, 32), "white").save(path)
            self.assertEqual(difference_hash(path), difference_hash(path))

    def test_category_review_rules_are_explicit(self) -> None:
        self.assertTrue(
            category_review_required("complex_academic_014.jpg")
        )
        self.assertTrue(
            category_review_required("software_web_code_005.jpg")
        )
        self.assertFalse(
            category_review_required("scene_text_001.jpg")
        )

    def test_parse_bool_accepts_csv_values(self) -> None:
        self.assertTrue(parse_bool("True"))
        self.assertFalse(parse_bool("false"))
        with self.assertRaises(ValueError):
            parse_bool("unknown")

    def test_enrich_manifest_joins_by_original_filename(self) -> None:
        manifest = [
            {
                "item_id": "user_abc",
                "source_file_name": "natural_no_text_001.jpg",
                "display_name_zh": "natural_no_text_001",
            }
        ]
        source_rows = [
            {
                "filename": "natural_no_text_001.jpg",
                "category": "natural_no_text",
                "has_text": "False",
                "review_status": "待人工审核",
                "language": "none",
                "quality_tags": "natural_image,no_text",
                "source_name": "Wikimedia Commons",
                "source_url": "https://example.test/file",
                "source_file": "File:Example.jpg",
                "license": "CC0",
                "license_url": "https://creativecommons.org/publicdomain/",
                "hard_negative_group": "natural_no_text_01",
                "perceptual_group": "visual_unique_001",
                "privacy_review_required": "false",
                "category_review_required": "false",
            }
        ]
        enriched = enrich_manifest(manifest, source_rows)
        self.assertEqual(enriched[0]["category"], "natural_no_text")
        self.assertFalse(enriched[0]["has_text_expected"])
        self.assertEqual(
            enriched[0]["display_name_zh"], "自然图像负样本 001"
        )
        self.assertEqual(
            enriched[0]["dataset_name"], "public_dataset_200_candidate"
        )

    def test_public_split_targets_total_120_40_40(self) -> None:
        totals = [
            sum(values[index] for values in CATEGORY_TARGETS.values())
            for index in range(3)
        ]
        self.assertEqual(totals, [120, 40, 40])

    def test_grouped_split_prevents_visual_leakage(self) -> None:
        rows = []
        serial = 0
        for category, counts in CATEGORY_TARGETS.items():
            for index in range(sum(counts)):
                serial += 1
                group = ""
                if category == "clear_document" and index < 2:
                    group = "near_duplicate_pair"
                rows.append(
                    {
                        "item_id": f"{serial:016x}",
                        "filename": f"{category}_{index:03d}.jpg",
                        "category": category,
                        "perceptual_group": group,
                        "hard_negative_group": f"{category}_{index // 2}",
                        "has_text": str(
                            category != "natural_no_text"
                        ),
                        "review_status": "待人工审核",
                        "privacy_review_required": "false",
                        "category_review_required": "false",
                    }
                )
        split_rows = build_split_rows(rows, seed=7)
        summary = validate_splits(split_rows)
        self.assertEqual(
            summary["split_counts"],
            {"train": 120, "validation": 40, "test": 40},
        )
        pair_splits = {
            row["split"]
            for row in split_rows
            if row["perceptual_group"] == "near_duplicate_pair"
        }
        self.assertEqual(len(pair_splits), 1)

    def test_quality_features_distinguish_text_and_empty_pages(self) -> None:
        text = quality_features(
            [0.9, 0.8], ["测试文档内容", "第二行文本"], True
        )
        empty = quality_features([], [], False)
        self.assertTrue(text["proposed_ocr_useful"])
        self.assertFalse(empty["proposed_ocr_useful"])
        self.assertEqual(empty["mean_confidence"], 0.0)

    def test_review_queue_keeps_privacy_and_category_coverage(self) -> None:
        rows = []
        for category in ("a", "b"):
            for index in range(4):
                rows.append(
                    {
                        "filename": f"{category}_{index}.jpg",
                        "category": category,
                        "risk_score": index / 10,
                        "privacy_review_required": (
                            category == "a" and index == 0
                        ),
                    }
                )
        queue = select_review_queue(rows, 5)
        self.assertIn("a_0.jpg", {row["filename"] for row in queue})
        self.assertEqual({row["category"] for row in queue}, {"a", "b"})

    def test_selective_rotation_requires_material_gain(self) -> None:
        baseline = {
            "angle": 0,
            "character_count": 4,
            "mean_confidence": 0.2,
            "low_confidence_fraction": 0.8,
        }
        improved = {
            "angle": 180,
            "character_count": 40,
            "mean_confidence": 0.9,
            "low_confidence_fraction": 0.0,
        }
        self.assertGreater(
            rotation_utility(improved), rotation_utility(baseline)
        )
        self.assertTrue(should_accept_rotation(baseline, improved))
        self.assertFalse(
            should_accept_rotation(baseline, {**improved, "angle": 0})
        )

    def test_ocr_override_is_preferred_without_deleting_baseline(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            baseline_dir = root / "baseline"
            overrides_dir = root / "overrides"
            baseline_dir.mkdir()
            overrides_dir.mkdir()
            baseline = baseline_dir / "sample.json"
            override = overrides_dir / "sample.json"
            baseline.write_text("{}", encoding="utf-8")
            override.write_text("{}", encoding="utf-8")
            resolved, used_override = resolve_ocr_path(
                baseline_dir, "sample", overrides_dir
            )
            self.assertEqual(resolved, override)
            self.assertTrue(used_override)
            self.assertTrue(baseline.is_file())

    def test_query_evidence_rejects_identifiers_and_prefers_unique(self) -> None:
        self.assertFalse(valid_evidence_line("1234567890", 0.99))
        self.assertFalse(
            valid_evidence_line("tingting@example.com", 0.99)
        )
        self.assertFalse(
            valid_evidence_line("家庭住址：某省某市", 0.99)
        )
        self.assertFalse(
            valid_evidence_line(
                "1201 Pennsylvania Avenue, N.W.", 0.99
            )
        )
        self.assertFalse(
            valid_evidence_line("www.example.com", 0.99)
        )
        self.assertFalse(valid_evidence_line("question", 0.99))
        self.assertTrue(valid_evidence_line("卷积神经网络", 0.90))
        evidence = choose_evidence(
            [("常见标题内容", 0.99), ("独特实验设置", 0.90)],
            Counter({"常见标题内容": 5, "独特实验设置": 1}),
        )
        self.assertEqual(evidence[0], "独特实验设置")
        self.assertIn(
            "独特实验设置",
            query_from_evidence("complex_academic", evidence),
        )

    def test_query_review_prioritizes_blank_and_double_failure(self) -> None:
        score, reasons = priority_score(
            {
                "query": "",
                "ambiguity_flags": "privacy_query_requires_manual",
            },
            None,
            None,
            None,
        )
        self.assertGreaterEqual(score, 180)
        self.assertIn("blank_query", reasons)
        failed_score, failed_reasons = priority_score(
            {"query": "测试查询", "ambiguity_flags": ""},
            9,
            8,
            5,
        )
        self.assertGreater(failed_score, 30)
        self.assertIn("both_branches_fail", failed_reasons)

    def test_learned_gate_helpers_are_numerically_stable(self) -> None:
        row = {
            "mean_confidence": "0.9",
            "p10_confidence": "0.7",
            "low_confidence_fraction": "0.1",
            "zero_confidence_fraction": "0.0",
            "character_count": "100",
            "text_box_count": "10",
            "chars_per_box": "10",
        }
        self.assertEqual(len(feature_vector(row)), 7)
        normalized = normalize_gate_rows(
            np.array([[2.0, 2.0], [1.0, 3.0]], dtype=np.float32)
        )
        self.assertTrue(np.isfinite(normalized).all())
        metrics = gate_retrieval_metrics(
            np.array([[0.1, 0.9], [0.8, 0.2]], dtype=np.float32),
            np.array([1, 0], dtype=np.int64),
        )
        self.assertEqual(metrics["recall_at_1"], 1.0)

    def test_modality_preference_labels_text_visual_and_tie(self) -> None:
        text = np.array(
            [[0.9, 0.1], [0.1, 0.9], [0.8, 0.2]],
            dtype=np.float32,
        )
        visual = np.array(
            [[0.1, 0.9], [0.9, 0.1], [0.7, 0.3]],
            dtype=np.float32,
        )
        targets = np.array([0, 0, 0], dtype=np.int64)
        labels = modality_preference_labels(text, visual, targets)
        self.assertEqual(labels.tolist(), [1, 0, -1])

    def test_multi_positive_query_falls_back_to_single_target(self) -> None:
        self.assertEqual(
            text_relevant_item_ids(
                {
                    "expected_item_id": "a",
                    "relevant_item_ids": "a;b",
                }
            ),
            {"a", "b"},
        )
        self.assertEqual(
            text_relevant_item_ids(
                {"expected_item_id": "a", "relevant_item_ids": ""}
            ),
            {"a"},
        )


if __name__ == "__main__":
    unittest.main()
