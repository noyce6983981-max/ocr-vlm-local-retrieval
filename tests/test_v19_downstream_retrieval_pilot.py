from __future__ import annotations

from scripts.run_v19_downstream_retrieval_pilot import (
    retrieval_snapshot,
    source_rank,
    summarize_behavior,
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


def test_retrieval_snapshot_keeps_acceptance_policy_and_wall_clock() -> None:
    payload = {
        "rankings": {"quality_hybrid": [{"item_id": "top"}]},
        "acceptance": {
            "quality_hybrid": {
                "accepted": False,
                "reason": "open-set",
                "signal_name": "probability",
                "signal": 0.2,
                "threshold": 0.4,
            }
        },
        "executed_branches": {"text": True, "bm25": True, "visual": True},
        "exploratory_query": False,
        "retrieval_route": "mixed",
        "timings": {"total_seconds": 1.2},
        "v18_1_intent_routing": {
            "route_latency_ms": 75.0,
            "wall_total_seconds": 1.3,
        },
    }
    snapshot = retrieval_snapshot(payload)
    assert snapshot["top1_item_id"] == "top"
    assert snapshot["accepted"] is False
    assert snapshot["route_latency_ms"] == 75.0
    assert snapshot["wall_total_seconds"] == 1.3


def test_behavior_summary_detects_ranking_acceptance_and_policy_changes() -> None:
    records = [
        {
            "baseline": {
                "top1_item_id": "a",
                "accepted": False,
                "executed_branches": {"visual": True},
                "exploratory_query": False,
                "retrieval_route": "mixed",
            },
            "candidate": {
                "top1_item_id": "a",
                "accepted": True,
                "executed_branches": {"visual": True},
                "exploratory_query": True,
                "retrieval_route": "visual_discovery",
            },
        }
    ]
    summary = summarize_behavior(records)
    assert summary["top1_changed_count"] == 0
    assert summary["acceptance_changed_count"] == 1
    assert summary["exploratory_changed_count"] == 1
    assert summary["route_transition_counts"] == {
        "mixed->visual_discovery": 1
    }
