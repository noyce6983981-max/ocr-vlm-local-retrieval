from __future__ import annotations

from scripts.evaluate_v19_2_condition_aggregation_retrieval import explicit_phrases


def test_explicit_phrases_requires_two_quoted_conditions() -> None:
    query = "帮我找同时写有“Alpha Project”和“Budget 2026”的那一页。"
    assert explicit_phrases(query) == ["Alpha Project", "Budget 2026"]
