from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from ocr_vlm_retrieval.gating.attribute_coverage import (
    AttributePlan,
    AttributeRequirement,
)
from ocr_vlm_retrieval.gating.listwise_selection import (
    CandidateEvidence,
    select_candidate,
)


def _plan() -> AttributePlan:
    requirements = (
        AttributeRequirement("object_1", "object", "海豚", "海豚", 0.5, 0.8),
        AttributeRequirement("color_1", "color", "黄色", "黄色", 0.5, 0.8),
        AttributeRequirement(
            "relation_1", "relation", "海豚跃过金字塔", "跃过", 0.5, 0.8
        ),
    )
    return AttributePlan("黄色海豚跃过金字塔", requirements, True, 1, "plan")


def _candidate(
    item_id: str,
    rank: int,
    full: float,
    scores: dict[str, float],
    *,
    relation_margin: float | None = None,
) -> CandidateEvidence:
    margins = {} if relation_margin is None else {"relation_1": relation_margin}
    return CandidateEvidence(item_id, rank, 1.0 / rank, full, scores, margins)


def _l3_parameters() -> dict[str, object]:
    return {
        "top_k": 3,
        "full_query_floor": 0.4,
        "score_threshold": 0.45,
        "selection_margin_threshold": 0.0,
        "relation_margin_threshold": 0.0,
        "weights": {
            "retrieval": 0.15,
            "full_query": 0.25,
            "geometric_mean": 0.3,
            "weakest": 0.3,
        },
    }


def test_l0_keeps_rank_first_behavior() -> None:
    candidates = [
        _candidate("rank1", 1, 0.52, {}),
        _candidate("rank2", 2, 0.90, {}),
    ]
    decision = select_candidate(
        "L0", candidates, {"top_k": 3, "full_query_threshold": 0.51}
    )
    assert decision.accepted
    assert decision.selected_item_id == "rank1"


def test_l1_selects_max_verifier_instead_of_first_pass() -> None:
    candidates = [
        _candidate("rank1", 1, 0.52, {}),
        _candidate("rank2", 2, 0.90, {}),
    ]
    decision = select_candidate(
        "L1", candidates, {"top_k": 3, "full_query_threshold": 0.51}
    )
    assert decision.accepted
    assert decision.selected_item_id == "rank2"


def test_l2_fuses_retrieval_and_verifier_evidence() -> None:
    candidates = [
        _candidate("rank1", 1, 0.55, {}),
        _candidate("rank2", 2, 0.90, {}),
    ]
    decision = select_candidate(
        "L2",
        candidates,
        {
            "top_k": 3,
            "retrieval_weight": 0.2,
            "verifier_weight": 0.8,
            "full_query_floor": 0.4,
            "score_threshold": 0.5,
        },
    )
    assert decision.accepted
    assert decision.selected_item_id == "rank2"


def test_l3_rejects_high_global_score_when_one_condition_is_missing() -> None:
    candidates = [
        _candidate(
            "yellow_pyramid",
            1,
            0.95,
            {"object_1": 0.2, "color_1": 0.9, "relation_1": 0.2},
            relation_margin=-0.2,
        )
    ]
    decision = select_candidate(
        "L3", candidates, _l3_parameters(), attribute_plan=_plan()
    )
    assert not decision.accepted
    assert decision.reason == "no_candidate_covers_all_requirements"
    assert decision.assessments[0].coverage_ratio < 1.0
    assert "object_1" in decision.assessments[0].missing_requirement_ids


def test_l3_selects_complete_lower_ranked_candidate() -> None:
    candidates = [
        _candidate(
            "partial",
            1,
            0.95,
            {"object_1": 0.2, "color_1": 0.9, "relation_1": 0.2},
        ),
        _candidate(
            "complete",
            2,
            0.75,
            {"object_1": 0.8, "color_1": 0.8, "relation_1": 0.8},
            relation_margin=0.1,
        ),
    ]
    decision = select_candidate(
        "L3", candidates, _l3_parameters(), attribute_plan=_plan()
    )
    assert decision.accepted
    assert decision.selected_item_id == "complete"


def test_l3_abstains_when_two_complete_candidates_are_too_close() -> None:
    parameters = _l3_parameters()
    parameters["selection_margin_threshold"] = 0.2
    candidates = [
        _candidate(
            "first",
            1,
            0.8,
            {"object_1": 0.8, "color_1": 0.8, "relation_1": 0.8},
        ),
        _candidate(
            "second",
            2,
            0.8,
            {"object_1": 0.8, "color_1": 0.8, "relation_1": 0.8},
        ),
    ]
    decision = select_candidate(
        "L3", candidates, parameters, attribute_plan=_plan()
    )
    assert not decision.accepted
    assert decision.reason == "selection_margin_too_small"
