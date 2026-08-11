from __future__ import annotations

from scripts.run_intent_routing_study import run_human_oracle
from scripts.run_v19_rf_baseline import build_classifier


def test_human_oracle_is_explicitly_diagnostic() -> None:
    rows = [
        {
            "query_id": "q1",
            "family_id": "f1",
            "query_text": "查金额",
            "gold_route": "text_evidence",
        },
        {
            "query_id": "q2",
            "family_id": "f2",
            "query_text": "找蓝色海边照片",
            "gold_route": "visual_discovery",
        },
    ]
    payload = run_human_oracle(rows)
    assert payload["method"] == "B3_human_route_oracle"
    assert payload["diagnostic_only"] is True
    assert payload["metrics"]["accuracy"] == 1.0
    assert all(row["correct"] for row in payload["predictions"])


def test_rf_baseline_has_fixed_random_state() -> None:
    classifier = build_classifier()
    forest = classifier.named_steps["classifier"]
    assert forest.random_state == 20260811
    assert forest.n_estimators == 500
