from __future__ import annotations

from scripts.run_v19_downstream_retrieval_pilot import (
    source_rank,
    summarize_source_neighbor,
)


def test_source_rank_uses_only_accepted_quality_ranking() -> None:
    payload = {
        "rankings": {"quality_hybrid": [{"item_id": "a"}, {"item_id": "b"}]},
        "low_confidence_rankings": {
            "quality_hybrid": [{"item_id": "rejected"}]
        },
    }
    assert source_rank(payload, "b") == 2
    assert source_rank(payload, "rejected") is None


def test_source_neighbor_summary_separates_positive_hit_and_negative_far() -> None:
    rows = [
        {"query_role": "positive", "rank": 1},
        {"query_role": "positive", "rank": None},
        {"query_role": "single_condition_hard_negative", "rank": 1},
        {"query_role": "single_condition_hard_negative", "rank": 4},
    ]
    summary = summarize_source_neighbor(rows, "rank")
    assert summary["positive_source_hit_at_1"] == 0.5
    assert summary["hard_negative_source_far_at_1"] == 0.5
    assert summary["hard_negative_source_far_at_3"] == 0.5
    assert summary["hard_negative_source_far_at_10"] == 1.0
