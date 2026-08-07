from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from ocr_vlm_retrieval.gating.gate_calibration import (
    evaluate_topk_operating_point,
    mark_eligibility,
    select_near_optimal_operating_point,
    select_rank_first_passed,
)


def candidate(
    item_id: str,
    score: float,
    *,
    relevant: bool = False,
    positive_relation: float | None = None,
    negative_relation: float | None = None,
) -> dict:
    row = {
        "item_id": item_id,
        "full_query_score": score,
        "requirement_scores": {"object": score},
        "relevant": relevant,
    }
    if positive_relation is not None and negative_relation is not None:
        row["contrastive_relation_evidence"] = [
            {
                "positive_score": positive_relation,
                "negative_score": negative_relation,
                "absolute_threshold": 0.44,
            }
        ]
    return row


def test_rank_first_policy_does_not_reorder_two_passing_candidates() -> None:
    decision = select_rank_first_passed(
        [candidate("rank1", 0.6), candidate("rank2", 0.9)],
        top_k=2,
        method="attribute_geometric",
        threshold=0.5,
        contrastive_relations=False,
        relation_margin_threshold=0.0,
    )

    assert decision["selected_item_id"] == "rank1"
    assert decision["selected_rank"] == 1
    assert decision["verified_candidate_count"] == 1


def test_top3_recovers_relevant_candidate_and_contrastive_rejects_inverse() -> None:
    rows = [
        {
            "query_id": "q1",
            "group_id": "g1",
            "pool_relevance": "relevant_candidate_in_pool",
            "candidates": [
                candidate("q1_wrong", 0.2),
                candidate("q1_right", 0.8, relevant=True),
                candidate("q1_tail", 0.1),
            ],
        },
        {
            "query_id": "q2",
            "group_id": "g2",
            "pool_relevance": "no_relevant_candidate_in_pool",
            "candidates": [
                candidate(
                    "q2_inverse",
                    0.9,
                    positive_relation=0.52,
                    negative_relation=0.58,
                ),
                candidate("q2_tail1", 0.1),
                candidate("q2_tail2", 0.1),
            ],
        },
        {
            "query_id": "q3",
            "group_id": "g3",
            "pool_relevance": "no_relevant_candidate_in_pool",
            "candidates": [
                candidate("q3_1", 0.1),
                candidate("q3_2", 0.1),
                candidate("q3_3", 0.1),
            ],
        },
        {
            "query_id": "q4",
            "group_id": "g4",
            "pool_relevance": "relevant_candidate_in_pool",
            "candidates": [
                candidate("q4_right", 0.7, relevant=True),
                candidate("q4_2", 0.1),
                candidate("q4_3", 0.1),
            ],
        },
    ]

    metrics, decisions = evaluate_topk_operating_point(
        rows,
        top_k=3,
        method="attribute_geometric",
        threshold=0.5,
        contrastive_relations=True,
        relation_margin_threshold=0.05,
    )

    assert metrics["retrieval_recall_at_k"] == 1.0
    assert metrics["selected_relevant_query_rate"] == 1.0
    assert metrics["pool_conditioned_false_accept_rate"] == 0.0
    assert metrics["pool_conditioned_end_to_end_accuracy"] == 1.0
    assert decisions[0]["selected_rank"] == 2
    assert not decisions[1]["accepted"]


def test_near_optimal_rule_prefers_smallest_k() -> None:
    base = {
        "eligible": True,
        "selected_relevant_query_rate": 0.8,
        "pool_conditioned_false_accept_rate": 0.1,
        "contrastive_relations": True,
        "method": "attribute_geometric",
        "threshold": 0.5,
        "relation_margin_threshold": 0.05,
    }
    selected = select_near_optimal_operating_point(
        [
            {**base, "top_k": 1, "pool_conditioned_end_to_end_accuracy": 0.88},
            {**base, "top_k": 3, "pool_conditioned_end_to_end_accuracy": 0.90},
            {**base, "top_k": 5, "pool_conditioned_end_to_end_accuracy": 0.91},
        ],
        accuracy_tolerance=0.03,
    )

    assert selected["top_k"] == 1


def test_mark_eligibility_records_relative_reduction() -> None:
    point = mark_eligibility(
        {
            "pool_conditioned_false_accept_rate": 0.4,
            "selected_relevant_query_rate": 0.5,
            "degenerate_reject_all": False,
        },
        baseline_false_accept_rate=0.8,
        minimum_false_accept_relative_reduction=0.3,
        minimum_selected_relevant_rate=0.3,
    )
    assert point["eligible"]
    assert point["pool_conditioned_false_accept_relative_reduction_vs_v16"] == 0.5
