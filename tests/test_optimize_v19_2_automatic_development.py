from __future__ import annotations

from scripts.optimize_v19_2_automatic_development import (
    stop_condition_audit,
    truncate_decision,
)


CONDITIONS = {
    "minimum_end_to_end_accuracy": 0.8,
    "minimum_topic_positive_selected_relevant": 0.7,
    "maximum_hard_negative_far": 0.2,
    "minimum_positive_recall_at_3": 0.9,
}


def test_stop_condition_audit_accepts_boundary_values() -> None:
    metrics = {
        "end_to_end_accuracy": 0.8,
        "topic_positive_selected_relevant": 0.7,
        "hard_negative_far": 0.2,
        "positive_recall_at_3": 0.9,
    }
    assert stop_condition_audit(metrics, CONDITIONS)["all_met"] is True


def test_stop_condition_audit_reports_failed_far() -> None:
    metrics = {
        "end_to_end_accuracy": 0.9,
        "topic_positive_selected_relevant": 1.0,
        "hard_negative_far": 0.25,
        "positive_recall_at_3": 1.0,
    }
    audit = stop_condition_audit(metrics, CONDITIONS)
    assert audit["all_met"] is False
    assert audit["checks"]["maximum_hard_negative_far"] is False


def test_truncate_decision_selects_first_complete_evidence_in_prefix() -> None:
    decision = {
        "candidate_evidence": [
            {"item_id": "a", "retrieval_rank": 1, "all_constraints_matched": False},
            {"item_id": "b", "retrieval_rank": 2, "all_constraints_matched": True},
        ],
        "accepted": True,
        "selected_item_id": "b",
    }
    top_one = truncate_decision(decision, top_k=1)
    top_two = truncate_decision(decision, top_k=2)
    assert top_one["accepted"] is False
    assert top_one["selected_item_id"] is None
    assert top_two["accepted"] is True
    assert top_two["selected_item_id"] == "b"
