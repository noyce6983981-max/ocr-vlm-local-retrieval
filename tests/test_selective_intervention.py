from __future__ import annotations

from ocr_vlm_retrieval.gating.selective_intervention import (
    SelectiveInterventionPolicy,
    admit_guarded_intervention,
)


def plan() -> dict:
    return {
        "compositional": True,
        "requirements": [
            {
                "requirement_id": "object",
                "kind": "object",
                "threshold": 0.4,
                "mandatory": True,
            },
            {
                "requirement_id": "relation",
                "kind": "relation",
                "threshold": 0.44,
                "mandatory": True,
            },
        ],
    }


def candidate(*, relation_margin: float = 0.02) -> dict:
    return {
        "item_id": "rescued",
        "retrieval_rank": 1,
        "full_query_score": 0.7,
        "resolved_requirement_scores": {"object": 0.6, "relation": 0.5},
        "contrastive_relation_evidence": [
            {"requirement_id": "relation", "margin": relation_margin}
        ],
    }


def test_guarded_policy_can_rescue_a_rejected_baseline() -> None:
    decision = admit_guarded_intervention(
        {"accepted": False, "selected_item_id": None, "top_score": 0.6},
        [candidate(), {**candidate(), "item_id": "second", "full_query_score": 0.5}],
        plan(),
        SelectiveInterventionPolicy(),
    )
    assert decision.intervention_applied is True
    assert decision.accepted is True
    assert decision.selected_item_id == "rescued"


def test_guarded_policy_never_replaces_an_accepted_baseline() -> None:
    decision = admit_guarded_intervention(
        {"accepted": True, "selected_item_id": "frozen", "top_score": 0.64},
        [candidate()],
        plan(),
        SelectiveInterventionPolicy(),
    )
    assert decision.intervention_applied is False
    assert decision.selected_item_id == "frozen"


def test_guarded_policy_rejects_weak_counterfactual_relation() -> None:
    decision = admit_guarded_intervention(
        {"accepted": False, "selected_item_id": None, "top_score": 0.6},
        [candidate(relation_margin=0.001)],
        plan(),
        SelectiveInterventionPolicy(),
    )
    assert decision.intervention_applied is False
    assert decision.reason == "keep_v18_relation_counterfactual_not_separated"
