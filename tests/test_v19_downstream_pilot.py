from __future__ import annotations

import json

from ocr_vlm_retrieval.routing import HybridRouter, LLMRouter, RuleRouter
from scripts.route_v19_downstream_pilot import route_rows


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
    assert assignments[0]["llm_invoked"] is True
    assert assignments[0]["evidence"]["needs_visual_semantics"] is True
