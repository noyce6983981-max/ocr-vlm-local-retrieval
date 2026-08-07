from __future__ import annotations

import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.evaluate_live_search_blind import (
    aggregate_metrics,
    apply_judgment_overrides,
    classify_errors,
    live_search_command,
    ndcg_at_k,
    percentile,
    query_set_fingerprint,
    read_frozen_queries,
    summarize_live_payload,
)


class LiveSearchBlindEvaluationTests(unittest.TestCase):
    def test_read_frozen_queries_filters_and_validates(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "queries.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        "query_id",
                        "query",
                        "query_type",
                        "review_status",
                        "split",
                        "is_no_answer",
                        "relevant_item_ids",
                    ),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "query_id": "q2",
                        "query": "missing",
                        "query_type": "no_answer",
                        "review_status": "已确认",
                        "split": "test",
                        "is_no_answer": "True",
                        "relevant_item_ids": "",
                    }
                )
                writer.writerow(
                    {
                        "query_id": "q1",
                        "query": "answerable",
                        "query_type": "text_explicit",
                        "review_status": "已确认",
                        "split": "test",
                        "is_no_answer": "False",
                        "relevant_item_ids": "a;b",
                    }
                )
                writer.writerow(
                    {
                        "query_id": "q3",
                        "query": "development",
                        "query_type": "text_explicit",
                        "review_status": "已确认",
                        "split": "train",
                        "is_no_answer": "False",
                        "relevant_item_ids": "c",
                    }
                )
            rows = read_frozen_queries(
                path, split="test", required_review_status="已确认"
            )
            self.assertEqual([row["query_id"] for row in rows], ["q1", "q2"])
            self.assertEqual(rows[0]["relevant_item_ids"], {"a", "b"})
            self.assertTrue(rows[1]["is_no_answer"])

    def test_query_fingerprint_is_stable_for_item_order(self) -> None:
        row = {
            "query_id": "q1",
            "query": "query",
            "query_type": "text",
            "is_no_answer": False,
            "relevant_item_ids": {"b", "a"},
        }
        self.assertEqual(query_set_fingerprint([row]), query_set_fingerprint([row]))

    def test_human_overrides_preserve_source_and_update_judgments(self) -> None:
        source = [
            {
                "query_id": "q1",
                "query": "place",
                "query_type": "visual_only",
                "is_no_answer": False,
                "relevant_item_ids": {"old"},
            },
            {
                "query_id": "q2",
                "query": "wall",
                "query_type": "visual_only",
                "is_no_answer": False,
                "relevant_item_ids": {"expected"},
            },
        ]
        revised = apply_judgment_overrides(
            source,
            [
                {
                    "query_id": "q1",
                    "is_no_answer": True,
                    "decision": "not_answerable_from_indexed_evidence",
                },
                {
                    "query_id": "q2",
                    "add_relevant_item_ids": ["also_relevant"],
                    "decision": "expanded_relevance_set",
                },
            ],
        )
        self.assertFalse(source[0]["is_no_answer"])
        self.assertEqual(source[0]["relevant_item_ids"], {"old"})
        self.assertTrue(revised[0]["is_no_answer"])
        self.assertEqual(revised[0]["relevant_item_ids"], set())
        self.assertEqual(
            revised[1]["relevant_item_ids"],
            {"expected", "also_relevant"},
        )

    def test_ndcg_rewards_all_relevant_items(self) -> None:
        ranking = [{"item_id": "a"}, {"item_id": "x"}, {"item_id": "b"}]
        score = ndcg_at_k(ranking, {"a", "b"}, 10)
        self.assertGreater(score, 0.9)
        self.assertLess(score, 1.0)

    def test_percentile_interpolates(self) -> None:
        self.assertEqual(percentile([1.0, 3.0], 0.5), 2.0)
        self.assertIsNone(percentile([], 0.95))

    def test_error_classification_separates_gate_and_ranker(self) -> None:
        record = {
            "is_no_answer": False,
            "accepted": False,
            "relevant_rank": 12,
        }
        self.assertEqual(
            classify_errors(record),
            ["false_reject", "recall_miss_top_10"],
        )
        self.assertEqual(
            classify_errors(
                {
                    "is_no_answer": True,
                    "accepted": True,
                    "relevant_rank": None,
                }
            ),
            ["false_accept"],
        )

    def test_summary_and_metrics_apply_product_gate(self) -> None:
        row = {
            "query_id": "q1",
            "query": "needle",
            "query_type": "text_explicit",
            "is_no_answer": False,
            "relevant_item_ids": {"target"},
        }
        payload = {
            "rankings": {
                "quality_hybrid": [
                    {"item_id": "wrong"},
                    {"item_id": "target"},
                ]
            },
            "acceptance": {"quality_hybrid": {"accepted": False}},
            "search_policy_version": 14,
            "timings": {"total_seconds": 0.5},
        }
        record = summarize_live_payload(
            row,
            payload,
            method="quality_hybrid",
            wall_seconds=0.6,
        )
        metrics = aggregate_metrics([record])
        self.assertEqual(metrics["ranker_recall_at_3"], 1.0)
        self.assertEqual(metrics["product_recall_at_3"], 0.0)
        self.assertEqual(metrics["answerable_acceptance_rate"], 0.0)
        self.assertEqual(record["component_cache_state"], "cold")
        self.assertEqual(metrics["component_cache_state_counts"], {"cold": 1})

    def test_protocol_controls_component_cache_forcing(self) -> None:
        row = {"query": "needle", "query_id": "q1"}
        runtime_path = Path("runtime.json")
        protocol = {
            "method": "quality_hybrid",
            "library_dir": "outputs/user_library",
            "rerank_top_k": 0,
            "force_components": True,
        }
        self.assertIn("--force", live_search_command(row, protocol, runtime_path))
        protocol["force_components"] = False
        self.assertNotIn(
            "--force", live_search_command(row, protocol, runtime_path)
        )


if __name__ == "__main__":
    unittest.main()
