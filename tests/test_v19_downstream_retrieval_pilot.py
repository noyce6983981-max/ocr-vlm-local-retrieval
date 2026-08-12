from __future__ import annotations

from scripts.run_v19_downstream_retrieval_pilot import (
    retrieval_snapshot,
    source_rank,
    summarize_behavior,
    summarize_e2e,
    summarize_source_neighbor,
    with_route_latency,
)


def test_source_rank_uses_only_accepted_quality_ranking() -> None:
    payload = {
        "rankings": {"quality_hybrid": [{"item_id": "a"}, {"item_id": "b"}]},
        "low_confidence_rankings": {"quality_hybrid": [{"item_id": "rejected"}]},
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


def test_source_neighbor_summary_accepts_all_reviewed_e2e_roles() -> None:
    rows = [
        {"query_role": "answerable_positive", "rank": 1},
        {"query_role": "paraphrase_positive", "rank": 2},
        {"query_role": "single_condition_hard_negative", "rank": None},
        {"query_role": "unanswerable_neighbor", "rank": 1},
    ]
    summary = summarize_source_neighbor(rows, "rank")
    assert summary["positive_count"] == 2
    assert summary["hard_negative_count"] == 2
    assert summary["positive_source_hit_at_1"] == 0.5
    assert summary["hard_negative_source_far_at_1"] == 0.5


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


def test_e2e_summary_uses_acceptance_and_any_false_accept() -> None:
    records = [
        {
            "gold_answerable": True,
            "query_role": "answerable_positive",
            "source_item_id": "target",
            "gold_relevant_item_ids": ["target", "also-relevant"],
            "result": {
                "accepted": True,
                "top1_item_id": "also-relevant",
                "top3_item_ids": ["also-relevant"],
                "wall_total_seconds": 1.0,
            },
        },
        {
            "gold_answerable": False,
            "query_role": "single_condition_hard_negative",
            "source_item_id": "target",
            "result": {
                "accepted": True,
                "top1_item_id": "other",
                "top3_item_ids": ["other"],
                "wall_total_seconds": 2.0,
            },
        },
        {
            "gold_answerable": False,
            "query_role": "unanswerable_neighbor",
            "source_item_id": "target",
            "result": {
                "accepted": False,
                "top1_item_id": None,
                "top3_item_ids": [],
                "wall_total_seconds": 3.0,
            },
        },
    ]
    summary = summarize_e2e(records, "result")
    assert summary["positive_top1_accuracy"] == 1.0
    assert summary["end_to_end_accuracy"] == 2 / 3
    assert summary["recall_at_3"] == 1.0
    assert summary["negative_far"] == 0.5
    assert summary["hard_negative_far"] == 1.0
    assert summary["neighbor_negative_far"] == 0.0
    assert summary["warm_wall_p95_seconds"] == 3.0


def test_candidate_wall_time_adds_measured_route_latency() -> None:
    snapshot = with_route_latency(
        {"core_total_seconds": 1.25, "wall_total_seconds": 1.25}, 250.0
    )
    assert snapshot["wall_total_seconds"] == 1.5


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
    assert summary["route_transition_counts"] == {"mixed->visual_discovery": 1}
