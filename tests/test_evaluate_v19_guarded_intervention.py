from __future__ import annotations

from ocr_vlm_retrieval.gating.selective_intervention import SelectiveInterventionPolicy
from scripts.evaluate_v19_guarded_intervention import evaluate, summarize


def test_summary_counts_positive_and_rejection_in_one_e2e_denominator() -> None:
    rows = [
        {
            "gold_answerable": True,
            "gold_relevant_item_ids": ["a"],
            "accepted": True,
            "selected_item_id": "a",
            "query_role": "answerable_positive",
        },
        {
            "gold_answerable": False,
            "gold_relevant_item_ids": [],
            "accepted": False,
            "selected_item_id": None,
            "query_role": "single_condition_hard_negative",
        },
    ]
    assert summarize(rows)["end_to_end_accuracy"] == 1.0


def test_evaluator_uses_v18_as_default_and_admits_only_verified_rescue() -> None:
    assignments = {
        "split": "v19_reviewed_development_only",
        "assignments": [
            {
                "query_id": "q",
                "query_role": "answerable_positive",
                "content_stratum": "layout_table",
            }
        ],
    }
    baseline = {
        "method": "frozen_v18_L1_max_verifier_score",
        "results": [
            {
                "query_id": "q",
                "gold_answerable": True,
                "gold_relevant_item_ids": ["target"],
                "accepted": False,
                "selected_item_id": None,
                "top_score": 0.57,
            }
        ],
    }
    late = {
        "split": "development_only",
        "results": [
            {
                "query_id": "q",
                "attribute_plan": {
                    "compositional": True,
                    "requirements": [
                        {"requirement_id": "o", "kind": "object", "threshold": 0.4},
                        {"requirement_id": "r", "kind": "relation", "threshold": 0.44},
                    ],
                },
                "candidates": [
                    {
                        "item_id": "target",
                        "full_query_score": 0.7,
                        "resolved_requirement_scores": {"o": 0.6, "r": 0.5},
                        "contrastive_relation_evidence": [
                            {"requirement_id": "r", "margin": 0.02}
                        ],
                    }
                ],
            }
        ],
    }
    guarded = {"q": {"rankings": {"quality_hybrid": [{"item_id": "target"}]}}}
    report = evaluate(
        assignments,
        baseline,
        late,
        guarded,
        SelectiveInterventionPolicy(),
    )
    assert report["baseline"]["end_to_end_accuracy"] == 0.0
    assert report["candidate"]["end_to_end_accuracy"] == 1.0
    assert report["intervention_count"] == 1
