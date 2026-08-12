from __future__ import annotations

import json

from ocr_vlm_retrieval.routing import HybridRouter, LLMRouter, RuleRouter
from scripts.route_v19_downstream_pilot import development_rows, route_rows


class StubBackend:
    name = "stub"

    def generate_intent_json(self, query: str) -> str:
        assert query
        return json.dumps(
            {
                "needs_literal_text": False,
                "needs_visual_semantics": True,
                "needs_layout_structure": False,
                "needs_exact_entity": False,
                "needs_topic_discovery": False,
                "is_compositional": False,
            }
        )


def test_downstream_assignment_keeps_full_route_provenance() -> None:
    rule = RuleRouter(lambda query: "mixed")
    hybrid = HybridRouter(rule, LLMRouter(StubBackend()))
    assignments, latencies = route_rows(
        [
            {
                "query_id": "q1",
                "query": "找相似画面",
                "query_role": "positive",
                "source_item_id": "item-1",
            }
        ],
        hybrid,
        rule,
    )
    assert len(latencies) == 1
    assert assignments[0]["legacy_route"] == "mixed"
    assert assignments[0]["calibrated_rule_route"] == "mixed"
    assert assignments[0]["hybrid_route"] == "visual_discovery"
    assert assignments[0]["route_changed"] is True
    assert assignments[0]["guarded_route"] == "mixed"
    assert assignments[0]["guarded_route_changed"] is False
    assert assignments[0]["llm_invoked"] is True
    assert assignments[0]["evidence"]["needs_visual_semantics"] is True


def test_reviewed_e2e_rows_use_query_text_and_target_item() -> None:
    rule = RuleRouter(lambda query: "mixed")
    hybrid = HybridRouter(rule, LLMRouter(StubBackend()))
    assignments, _ = route_rows(
        [
            {
                "query_id": "q2",
                "query_text": "  找右上角带校徽的封面  ",
                "query_role": "answerable_positive",
                "target_item_id": "item-2",
                "neighbor_item_id": "item-3",
                "gold_answerable": True,
                "gold_relevant_item_ids": ["item-2", "item-4"],
                "family_id": "family-1",
                "split": "development",
                "content_stratum": "text_visual_compositional",
            }
        ],
        hybrid,
        rule,
    )
    assert assignments[0]["query"] == "找右上角带校徽的封面"
    assert assignments[0]["source_item_id"] == "item-2"
    assert assignments[0]["neighbor_item_id"] == "item-3"
    assert assignments[0]["gold_answerable"] is True
    assert assignments[0]["gold_relevant_item_ids"] == ["item-2", "item-4"]
    assert assignments[0]["family_id"] == "family-1"
    assert assignments[0]["split"] == "development"
    assert assignments[0]["route_latency_ms"] >= 0.0


def test_development_selector_never_returns_holdout() -> None:
    selected = development_rows(
        [
            {"query_id": "dev", "split": "development"},
            {"query_id": "held", "split": "holdout"},
        ]
    )
    assert [row["query_id"] for row in selected] == ["dev"]


def test_guarded_assignment_blocks_factual_to_discovery_transition() -> None:
    rule = RuleRouter(lambda query: "mixed")
    hybrid = HybridRouter(rule, LLMRouter(StubBackend()))
    assignments, _ = route_rows(
        [
            {
                "query_id": "q3",
                "query_text": "查找文件里明确写有金额的红色印章页面",
                "target_item_id": "item-3",
            }
        ],
        hybrid,
        rule,
    )
    row = assignments[0]
    assert row["hybrid_route"] == "visual_discovery"
    assert row["guarded_route"] == "mixed"
    assert row["intervention_guard_reason"] == "preserve_factual_acceptance"
