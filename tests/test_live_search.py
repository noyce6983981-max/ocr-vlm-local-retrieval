from __future__ import annotations

import unittest
import json
import os

import numpy as np

from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.live_search import (
    SEARCH_POLICY_VERSION,
    SELECTED_RETRIEVAL_CONFIG,
    add_reranker_ranking,
    apply_composite_visual_guard,
    apply_contextual_evidence_guard,
    apply_route_acceptance_guard,
    apply_route_ranking_policy,
    promote_complete_quoted_evidence,
    apply_strict_entity_policy,
    align_scores,
    cached_query_matches,
    cached_visual_query_matches,
    discovery_acceptance_decision,
    exact_evidence_item_ids,
    filter_discovery_rankings,
    is_composite_visual_query,
    library_revision,
    query_key,
    prune_json_cache,
    read_json_object,
    reranker_cache_signature,
    retrieval_config_revision,
    required_search_branches,
    resolve_search_intent,
    snapshot_low_confidence_candidates,
    truncate_rankings_for_output,
    write_json_atomic,
)


class LiveSearchTests(unittest.TestCase):
    def test_selected_policy_metadata_matches_runtime(self) -> None:
        payload = json.loads(SELECTED_RETRIEVAL_CONFIG.read_text(encoding="utf-8"))
        policy = payload["product_policy"]
        self.assertEqual(policy["version"], SEARCH_POLICY_VERSION)
        self.assertEqual(policy["precision_timeout_seconds"], 180)

    def test_atomic_json_write_and_corrupt_cache_miss(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "cache.json"
            write_json_atomic(path, {"query": "山水", "count": 3})
            self.assertEqual(read_json_object(path)["count"], 3)
            self.assertEqual(list(path.parent.glob("*.tmp")), [])
            path.write_text('{"query":', encoding="utf-8")
            self.assertIsNone(read_json_object(path))

    def test_cache_pruning_keeps_newest_and_protected(self) -> None:
        with TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            oldest = cache_dir / "oldest.json"
            newest = cache_dir / "newest.json"
            protected = cache_dir / "protected.json"
            for path in (oldest, newest, protected):
                path.write_text("{}", encoding="utf-8")
            oldest.touch()
            newest.touch()
            protected.touch()
            os_times = {
                oldest: (1, 1),
                newest: (3, 3),
                protected: (2, 2),
            }
            for path, times in os_times.items():
                os.utime(path, times)
            summary = prune_json_cache(
                cache_dir,
                protected={protected},
                max_files=2,
                max_bytes=1024,
            )
            self.assertEqual(summary["deleted_files"], 1)
            self.assertFalse(oldest.exists())
            self.assertTrue(newest.exists())
            self.assertTrue(protected.exists())

    def test_reranker_cache_signature_separates_query_modes(self) -> None:
        item_ids = ["a", "b"]
        self.assertNotEqual(
            reranker_cache_signature(item_ids, True),
            reranker_cache_signature(item_ids, False),
        )

    def test_filtered_candidates_remain_available_for_manual_reveal(self) -> None:
        rankings = {
            "quality_hybrid": [
                {"item_id": "a", "score": 0.9},
                {"item_id": "b", "score": 0.2},
            ]
        }
        snapshot = snapshot_low_confidence_candidates(rankings)
        rankings["quality_hybrid"] = []
        self.assertEqual(
            [row["item_id"] for row in snapshot["quality_hybrid"]],
            ["a", "b"],
        )

    def test_output_rankings_are_bounded(self) -> None:
        rankings = {"quality_hybrid": [{"item_id": str(index)} for index in range(140)]}
        trimmed = truncate_rankings_for_output(rankings)
        self.assertEqual(len(trimmed["quality_hybrid"]), 100)
        self.assertEqual(len(rankings["quality_hybrid"]), 140)

    def test_result_level_color_floor_removes_tiny_color_matches(self) -> None:
        rankings = {
            "quality_hybrid": [
                {"item_id": "a", "color_score": 0.65},
                {"item_id": "b", "color_score": 0.03},
            ]
        }
        filter_discovery_rankings(
            "visual_discovery",
            rankings,
            set(),
            {"color_result_min_coverage": 0.18},
            "blue",
            True,
        )
        self.assertEqual(
            [row["item_id"] for row in rankings["quality_hybrid"]],
            ["a"],
        )

    def test_compound_color_query_requires_color_and_semantics(self) -> None:
        rankings = {
            "quality_hybrid": [
                {
                    "item_id": "match",
                    "color_score": 0.25,
                    "raw_visual_score": 0.45,
                },
                {
                    "item_id": "wrong_object",
                    "color_score": 0.80,
                    "raw_visual_score": 0.20,
                },
                {
                    "item_id": "weak_object",
                    "color_score": 0.40,
                    "raw_visual_score": 0.33,
                },
            ]
        }
        filter_discovery_rankings(
            "visual_discovery",
            rankings,
            set(),
            {},
            "blue",
            False,
        )
        self.assertEqual(
            [row["item_id"] for row in rankings["quality_hybrid"]],
            ["match"],
        )

    def test_topic_discovery_rejects_weak_partial_match(self) -> None:
        decision = discovery_acceptance_decision(
            "topic_discovery",
            [{"raw_text_score": 0.42}],
            0,
            {"topic_discovery_min_dense_score": 0.44},
            "test",
        )
        self.assertFalse(decision["accepted"])

    def test_topic_discovery_accepts_exact_evidence(self) -> None:
        decision = discovery_acceptance_decision(
            "topic_discovery",
            [{"raw_text_score": 0.30}],
            2,
            {"topic_discovery_min_dense_score": 0.44},
            "test",
        )
        self.assertTrue(decision["accepted"])

    def test_visual_discovery_keeps_low_relevance_floor(self) -> None:
        decision = discovery_acceptance_decision(
            "visual_discovery",
            [{"raw_visual_score": 0.58}],
            0,
            {"visual_discovery_min_score": 0.35},
            "test",
        )
        self.assertTrue(decision["accepted"])

    def test_visual_keyword_auto_enters_discovery(self) -> None:
        route, exploratory, visual_query, entity = resolve_search_intent(
            "quality_hybrid", "美丽"
        )
        self.assertEqual(route, "visual_discovery")
        self.assertTrue(exploratory)
        self.assertIn("自然风景", visual_query)
        self.assertIsNone(entity)

    def test_forest_auto_enters_visual_discovery(self) -> None:
        route, exploratory, visual_query, entity = resolve_search_intent(
            "quality_hybrid", "森林"
        )
        self.assertEqual(route, "visual_discovery")
        self.assertTrue(exploratory)
        self.assertIn("绿色森林", visual_query)
        self.assertIsNone(entity)

    def test_landscape_auto_enters_visual_discovery(self) -> None:
        route, exploratory, visual_query, entity = resolve_search_intent(
            "quality_hybrid", "山水"
        )
        self.assertEqual(route, "visual_discovery")
        self.assertTrue(exploratory)
        self.assertIn("山水", visual_query)
        self.assertIsNone(entity)

    def test_short_topic_keyword_uses_recall_first_hybrid(self) -> None:
        route, exploratory, visual_query, entity = resolve_search_intent(
            "quality_hybrid", "量子力学"
        )
        self.assertEqual(route, "topic_discovery")
        self.assertTrue(exploratory)
        self.assertEqual(visual_query, "量子力学")
        self.assertIsNone(entity)

    def test_short_person_name_remains_strict(self) -> None:
        route, exploratory, visual_query, entity = resolve_search_intent(
            "quality_hybrid", "盛和"
        )
        self.assertEqual(route, "entity_exact")
        self.assertFalse(exploratory)
        self.assertEqual(visual_query, "盛和")
        self.assertEqual(entity, "盛和")

    def test_exact_entity_evidence_reads_ocr_index(self) -> None:
        with TemporaryDirectory() as directory:
            library_dir = Path(directory)
            index_dir = library_dir / "text_index"
            index_dir.mkdir()
            (index_dir / "metadata.jsonl").write_text(
                '{"item_id":"b","text":"报名人：方岩松"}\n',
                encoding="utf-8",
            )
            matches = exact_evidence_item_ids(
                "方岩松",
                library_dir,
                [
                    {"item_id": "a", "display_name_zh": "普通页面"},
                    {"item_id": "b", "display_name_zh": "登记表"},
                ],
            )
            self.assertEqual(matches, {"b"})

    def test_entity_policy_filters_unrelated_results(self) -> None:
        rankings = {
            "quality_hybrid": [
                {"item_id": "a", "score": 0.9},
                {"item_id": "b", "score": 0.8},
            ]
        }
        acceptance = {"quality_hybrid": {"accepted": True}}
        apply_strict_entity_policy(
            "方岩松",
            {"b"},
            rankings,
            acceptance,
            "test",
        )
        self.assertEqual(
            [row["item_id"] for row in rankings["quality_hybrid"]],
            ["b"],
        )
        self.assertTrue(acceptance["quality_hybrid"]["accepted"])

    def test_entity_without_exact_evidence_is_rejected(self) -> None:
        rankings = {"quality_hybrid": [{"item_id": "a"}]}
        acceptance = {"quality_hybrid": {"accepted": True}}
        apply_strict_entity_policy(
            "方岩松",
            set(),
            rankings,
            acceptance,
            "test",
        )
        self.assertFalse(acceptance["quality_hybrid"]["accepted"])
        self.assertEqual(acceptance["quality_hybrid"]["signal"], 0)

    def test_text_evidence_skips_unneeded_visual_model(self) -> None:
        self.assertEqual(
            required_search_branches("quality_hybrid", "text_evidence", False),
            {"text": True, "bm25": True, "visual": False},
        )

    def test_composite_visual_query_detection(self) -> None:
        self.assertTrue(
            is_composite_visual_query(
                "查找钢琴旁摆放手写乐谱和小提琴的室内照片。",
                "visual_metadata",
            )
        )
        self.assertFalse(is_composite_visual_query("蓝色汽车", "visual_discovery"))
        self.assertFalse(is_composite_visual_query("文档包含日期", "text_evidence"))

    def test_composite_visual_guard_rejects_single_term_overlap(self) -> None:
        decision = {"accepted": True, "reason": "old"}
        changed = apply_composite_visual_guard(
            "钢琴旁摆放手写乐谱和小提琴的室内照片",
            "visual_metadata",
            [{"raw_visual_score": 0.341}],
            decision,
            {"composite_visual_min_score": 0.35},
            "test",
        )
        self.assertTrue(changed)
        self.assertFalse(decision["accepted"])
        self.assertEqual(decision["signal_name"], "composite_visual_raw_cosine")

    def test_composite_visual_guard_keeps_strong_visual_candidate(self) -> None:
        decision = {"accepted": True, "reason": "old"}
        changed = apply_composite_visual_guard(
            "猫和狗同时出现在草地照片中",
            "visual_metadata",
            [{"raw_visual_score": 0.41}],
            decision,
            {"composite_visual_min_score": 0.35},
            "test",
        )
        self.assertFalse(changed)
        self.assertTrue(decision["accepted"])

    def test_complete_quoted_evidence_can_satisfy_visual_text_query(self) -> None:
        decision = {"accepted": True, "reason": "exact"}
        changed = apply_composite_visual_guard(
            "哪张场景照片中出现“JINGMI Rd”和“自行车优先”？",
            "visual_metadata",
            [{"item_id": "scene", "raw_visual_score": 0.30}],
            decision,
            {"composite_visual_min_score": 0.40},
            "test",
            {"scene"},
        )
        self.assertFalse(changed)
        self.assertTrue(decision["accepted"])

    def test_relational_visual_query_uses_composite_guard(self) -> None:
        self.assertTrue(
            is_composite_visual_query("查找宠物戴红色帽子的照片", "visual_discovery")
        )
        decision = {"accepted": True, "reason": "old"}
        changed = apply_composite_visual_guard(
            "查找宠物戴红色帽子的照片",
            "visual_discovery",
            [{"raw_visual_score": 0.35}],
            decision,
            {"composite_visual_min_score": 0.40},
            "test",
        )
        self.assertTrue(changed)
        self.assertFalse(decision["accepted"])

    def test_strong_relation_uses_stricter_visual_floor(self) -> None:
        decision = {"accepted": True, "reason": "old"}
        changed = apply_composite_visual_guard(
            "找蓝色鲸鱼游过沙漠峡谷的照片",
            "visual_discovery",
            [{"raw_visual_score": 0.451}],
            decision,
            {
                "composite_visual_min_score": 0.40,
                "strong_relation_visual_min_score": 0.46,
            },
            "test",
        )
        self.assertTrue(changed)
        self.assertFalse(decision["accepted"])
        self.assertEqual(decision["threshold"], 0.46)

    def test_text_gate_requires_aligned_dense_and_keyword_evidence(self) -> None:
        decision = {"accepted": True}
        apply_route_acceptance_guard(
            "查找包含Python代码的页面",
            "text_evidence",
            [
                {"raw_text_score": 0.45, "bm25_raw_score": 28.0},
                {"raw_text_score": 0.55, "bm25_raw_score": 2.0},
            ],
            decision,
            {},
            "test",
        )
        self.assertFalse(decision["accepted"])
        self.assertEqual(decision["signal"], 0)

    def test_text_gate_accepts_same_candidate_joint_evidence(self) -> None:
        decision = {"accepted": False}
        apply_route_acceptance_guard(
            "查找标题里有年度报告的文档",
            "text_evidence",
            [{"raw_text_score": 0.55, "bm25_raw_score": 8.0}],
            decision,
            {},
            "test",
        )
        self.assertTrue(decision["accepted"])

    def test_visual_metadata_gate_does_not_trust_metadata_alone(self) -> None:
        decision = {"accepted": True}
        apply_route_acceptance_guard(
            "查找活动宣传海报",
            "visual_metadata",
            [
                {
                    "raw_visual_score": 0.20,
                    "raw_metadata_score": 0.95,
                    "raw_text_score": 0.60,
                    "bm25_raw_score": 12.0,
                }
            ],
            decision,
            {},
            "test",
        )
        self.assertFalse(decision["accepted"])

    def test_visual_menu_query_also_requires_literal_text_evidence(self) -> None:
        decision = {"accepted": True}
        apply_route_acceptance_guard(
            "查找餐厅菜单或价目表照片",
            "visual_metadata",
            [
                {
                    "raw_visual_score": 0.48,
                    "raw_text_score": 0.51,
                    "bm25_raw_score": 0.0,
                }
            ],
            decision,
            {},
            "test",
        )
        self.assertFalse(decision["accepted"])

    def test_relative_time_without_indexed_evidence_is_rejected(self) -> None:
        decision = {"accepted": True}
        changed = apply_contextual_evidence_guard(
            "找我去年在海边拍的日落照片",
            [{"item_id": "sunset"}],
            decision,
            set(),
            "test",
        )
        self.assertTrue(changed)
        self.assertFalse(decision["accepted"])

    def test_route_ranking_policy_keeps_specialized_color_blend(self) -> None:
        rankings = {
            "quality_hybrid": [{"item_id": "color"}],
            "adaptive": [{"item_id": "visual"}],
        }
        source = apply_route_ranking_policy(
            rankings,
            "visual_discovery",
            "blue",
            False,
            False,
            {"route_ranking_sources": {"visual_discovery": "adaptive"}},
        )
        self.assertEqual(source, "quality_hybrid")
        self.assertEqual(rankings["quality_hybrid"][0]["item_id"], "color")

    def test_route_ranking_policy_can_select_adaptive_visual_metadata(
        self,
    ) -> None:
        rankings = {
            "quality_hybrid": [{"item_id": "metadata"}],
            "adaptive": [{"item_id": "joint"}],
        }
        source = apply_route_ranking_policy(
            rankings,
            "visual_metadata",
            None,
            False,
            False,
            {"route_ranking_sources": {"visual_metadata": "adaptive"}},
        )
        self.assertEqual(source, "adaptive")
        self.assertEqual(rankings["quality_hybrid"][0]["item_id"], "joint")

    def test_attribute_coverage_preserves_v17_quality_hybrid(self) -> None:
        rankings = {
            "quality_hybrid": [{"item_id": "attribute_complete"}],
            "quality_rrf": [{"item_id": "global_only"}],
        }
        source = apply_route_ranking_policy(
            rankings,
            "mixed",
            None,
            False,
            False,
            {"route_ranking_sources": {"mixed": "quality_rrf"}},
            preserve_quality_hybrid=True,
        )
        self.assertEqual(source, "quality_hybrid")
        self.assertEqual(rankings["quality_hybrid"][0]["item_id"], "attribute_complete")

    def test_plain_object_photo_can_use_metadata_visual_blend(self) -> None:
        rankings = {
            "quality_hybrid": [{"item_id": "old"}],
            "adaptive": [{"item_id": "joint"}],
            "visual_metadata_blend": [{"item_id": "phone"}],
        }
        source = apply_route_ranking_policy(
            rankings,
            "visual_metadata",
            None,
            False,
            False,
            {
                "route_ranking_sources": {
                    "visual_metadata": "adaptive",
                    "visual_metadata_plain_object": "visual_metadata_blend",
                }
            },
            query="找一张手机照片",
        )
        self.assertEqual(source, "visual_metadata_blend")
        self.assertEqual(rankings["quality_hybrid"][0]["item_id"], "phone")

    def test_structured_visual_query_keeps_adaptive_ranking(self) -> None:
        rankings = {
            "quality_hybrid": [{"item_id": "old"}],
            "adaptive": [{"item_id": "layout"}],
            "visual_metadata_blend": [{"item_id": "metadata"}],
        }
        source = apply_route_ranking_policy(
            rankings,
            "visual_metadata",
            None,
            False,
            False,
            {
                "route_ranking_sources": {
                    "visual_metadata": "adaptive",
                    "visual_metadata_plain_object": "visual_metadata_blend",
                }
            },
            query="查找有公式和图表的论文",
        )
        self.assertEqual(source, "adaptive")
        self.assertEqual(rankings["quality_hybrid"][0]["item_id"], "layout")

    def test_topic_without_literal_evidence_can_use_adaptive_ranking(self) -> None:
        rankings = {
            "quality_hybrid": [{"item_id": "sparse"}],
            "adaptive": [{"item_id": "semantic"}],
        }
        source = apply_route_ranking_policy(
            rankings,
            "topic_discovery",
            None,
            False,
            False,
            {"route_ranking_sources": {"topic_discovery_without_exact": "adaptive"}},
        )
        self.assertEqual(source, "adaptive")
        self.assertEqual(rankings["quality_hybrid"][0]["item_id"], "semantic")

    def test_exact_quoted_text_can_pass_without_dense_floor(self) -> None:
        decision = {"accepted": False}
        apply_route_acceptance_guard(
            "哪份文档中提到了“rare literal phrase”？",
            "text_evidence",
            [
                {
                    "item_id": "exact",
                    "raw_text_score": 0.40,
                    "bm25_raw_score": 30.0,
                }
            ],
            decision,
            {},
            "test",
            {"exact"},
        )
        self.assertTrue(decision["accepted"])

    def test_missing_complete_quote_rejects_approximate_text_match(self) -> None:
        decision = {"accepted": True}
        apply_route_acceptance_guard(
            "找写着“QCM-8842量子咖啡机验收通过”的表格",
            "text_evidence",
            [{"raw_text_score": 0.70, "bm25_raw_score": 20.0}],
            decision,
            {},
            "test",
            set(),
        )
        self.assertFalse(decision["accepted"])
        self.assertEqual(decision["signal_name"], "complete_quoted_evidence_count")

    def test_complete_quote_is_promoted_ahead_of_approximation(self) -> None:
        rankings = {
            "quality_hybrid": [
                {"item_id": "approximate"},
                {"item_id": "exact"},
            ]
        }
        promote_complete_quoted_evidence(rankings, {"exact"})
        self.assertEqual(
            [row["item_id"] for row in rankings["quality_hybrid"]],
            ["exact", "approximate"],
        )

    def test_quoted_visual_metadata_keeps_metadata_ranking(self) -> None:
        rankings = {
            "quality_hybrid": [{"item_id": "metadata"}],
            "adaptive": [{"item_id": "visual"}],
        }
        source = apply_route_ranking_policy(
            rankings,
            "visual_metadata",
            None,
            False,
            True,
            {"route_ranking_sources": {"visual_metadata": "adaptive"}},
        )
        self.assertEqual(source, "quality_hybrid")
        self.assertEqual(rankings["quality_hybrid"][0]["item_id"], "metadata")

    def test_visual_discovery_blends_all_three_branches(self) -> None:
        self.assertEqual(
            required_search_branches("quality_hybrid", "visual_discovery", True),
            {"text": True, "bm25": True, "visual": True},
        )

    def test_precision_mode_also_blends_visual_discovery(self) -> None:
        self.assertEqual(
            required_search_branches("reranker", "visual_discovery", True),
            {"text": True, "bm25": True, "visual": True},
        )

    def test_topic_discovery_uses_text_and_keywords(self) -> None:
        self.assertEqual(
            required_search_branches("quality_hybrid", "topic_discovery", True),
            {"text": True, "bm25": True, "visual": False},
        )

    def test_precision_search_remains_route_aware(self) -> None:
        self.assertEqual(
            required_search_branches("reranker", "text_evidence", False),
            {"text": True, "bm25": True, "visual": False},
        )

    def test_query_key_is_stable(self) -> None:
        self.assertEqual(query_key("同一个问题"), query_key("同一个问题"))
        self.assertNotEqual(query_key("问题A"), query_key("问题B"))

    def test_query_key_changes_with_library_revision(self) -> None:
        self.assertNotEqual(
            query_key("同一个问题", "revision-a"),
            query_key("同一个问题", "revision-b"),
        )

    def test_retrieval_config_has_separate_revision(self) -> None:
        revision = retrieval_config_revision()
        self.assertTrue(revision == "fallback" or len(revision) == 12)

    def test_category_only_edit_keeps_retrieval_revision(self) -> None:
        with TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.jsonl"
            manifest.write_text(
                '{"item_id":"a","source_path":"a.png",'
                '"category":"general_text_document"}\n',
                encoding="utf-8",
            )
            before = library_revision(Path(directory))
            manifest.write_text(
                '{"item_id":"a","source_path":"a.png",'
                '"category":"table_form_ticket"}\n',
                encoding="utf-8",
            )
            self.assertEqual(before, library_revision(Path(directory)))

    def test_align_scores_follows_manifest_order(self) -> None:
        payload = {
            "item_ids": ["b", "a", "c"],
            "scores": [0.2, 0.9, -1.0],
        }
        aligned = align_scores(payload, ["a", "b", "c"])
        np.testing.assert_allclose(aligned, [0.9, 0.2, -1.0])

    def test_cached_query_must_match(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "cache.json"
            path.write_text('{"query":"问题A"}', encoding="utf-8")
            self.assertTrue(cached_query_matches(path, "问题A"))
            self.assertFalse(cached_query_matches(path, "问题B"))

    def test_visual_cache_must_match_expanded_query(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "visual.json"
            path.write_text(
                '{"query":"森林","encoded_query":"绿色森林","library_revision":"r1"}',
                encoding="utf-8",
            )
            self.assertTrue(cached_visual_query_matches(path, "森林", "绿色森林", "r1"))
            self.assertFalse(
                cached_visual_query_matches(path, "森林", "原始森林", "r1")
            )

    def test_reranker_reorders_only_scored_candidates(self) -> None:
        rankings = {
            "adaptive": [
                {"item_id": "a", "score": 0.9},
                {"item_id": "b", "score": 0.8},
                {"item_id": "c", "score": 0.7},
            ]
        }
        add_reranker_ranking(
            rankings,
            {
                "item_ids": ["a", "b"],
                "scores": [0.1, 0.9],
            },
        )
        reranked = rankings["reranker"]
        self.assertEqual(
            [row["item_id"] for row in reranked],
            ["b", "a", "c"],
        )
        self.assertEqual(reranked[0]["reranker_score"], 0.9)
        self.assertIsNone(reranked[2]["reranker_score"])


if __name__ == "__main__":
    unittest.main()
