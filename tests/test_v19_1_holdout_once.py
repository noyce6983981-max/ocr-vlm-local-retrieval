from __future__ import annotations

import pytest

from scripts.run_v19_1_condition_completeness_holdout_once import (
    hard_negative_far,
    release_gates,
)


def test_v19_1_release_gates_require_gain_without_far_regression() -> None:
    baseline = {
        "end_to_end_accuracy": 0.50,
        "positive_recall_at_3": 0.75,
        "hard_negative_far": 0.25,
    }
    predecessor = {
        "end_to_end_accuracy": 0.65,
        "positive_selected_relevant": 0.70,
    }
    candidate = {
        "end_to_end_accuracy": 0.75,
        "positive_recall_at_3": 0.90,
        "hard_negative_far": 0.10,
        "positive_selected_relevant": 0.80,
    }
    assert all(release_gates(baseline, predecessor, candidate).values())


def test_v19_1_release_gates_reject_equal_predecessor_e2e() -> None:
    baseline = {
        "end_to_end_accuracy": 0.50,
        "positive_recall_at_3": 0.75,
        "hard_negative_far": 0.25,
    }
    predecessor = {
        "end_to_end_accuracy": 0.75,
        "positive_selected_relevant": 0.80,
    }
    candidate = {
        "end_to_end_accuracy": 0.75,
        "positive_recall_at_3": 0.90,
        "hard_negative_far": 0.10,
        "positive_selected_relevant": 0.80,
    }
    gates = release_gates(baseline, predecessor, candidate)
    assert gates["e2e_improved_vs_v19"] is False


def test_hard_negative_far_ignores_other_negative_role() -> None:
    rows = [
        {"query_role": "single_condition_hard_negative", "accepted": True},
        {"query_role": "single_condition_hard_negative", "accepted": False},
        {"query_role": "unanswerable_neighbor", "accepted": True},
    ]
    assert hard_negative_far(rows) == 0.5


def test_hard_negative_far_requires_hard_negatives() -> None:
    with pytest.raises(ValueError, match="hard-negative"):
        hard_negative_far([])
