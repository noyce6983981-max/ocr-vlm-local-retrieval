"""Conservative admission policy for V19 development interventions.

The policy is deliberately asymmetric: it may rescue a rejected V18 L1
decision, but it cannot revoke or replace an accepted V18 result.  This keeps
the frozen method as the default and makes every V19 behavioral change
explicit and auditable.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class SelectiveInterventionPolicy:
    """Label-blind requirements for admitting one guarded intervention."""

    full_query_threshold: float = 0.63
    score_gain_threshold: float = 0.05
    selection_margin_threshold: float = 0.03
    relation_margin_threshold: float = 0.015
    minimum_requirement_count: int = 2

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SelectiveInterventionDecision:
    """Auditable outcome of applying the V19 admission policy."""

    accepted: bool
    selected_item_id: str | None
    intervention_applied: bool
    reason: str
    diagnostics: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "selected_item_id": self.selected_item_id,
            "intervention_applied": self.intervention_applied,
            "reason": self.reason,
            "diagnostics": dict(self.diagnostics),
        }


def _baseline_decision(
    baseline: Mapping[str, Any], *, reason: str, diagnostics: Mapping[str, Any]
) -> SelectiveInterventionDecision:
    accepted = bool(baseline.get("accepted"))
    return SelectiveInterventionDecision(
        accepted=accepted,
        selected_item_id=(
            str(baseline["selected_item_id"])
            if accepted and baseline.get("selected_item_id")
            else None
        ),
        intervention_applied=False,
        reason=reason,
        diagnostics=diagnostics,
    )


def admit_guarded_intervention(
    baseline: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    attribute_plan: Mapping[str, Any],
    policy: SelectiveInterventionPolicy,
) -> SelectiveInterventionDecision:
    """Admit a guarded reranking change only when all evidence agrees.

    The function never reads relevance labels.  Candidate rows need an item
    ID, a full-query verifier score, resolved per-requirement scores, and any
    contrastive relation margins produced by the late-interaction scorer.
    """

    if bool(baseline.get("accepted")):
        return _baseline_decision(
            baseline,
            reason="keep_accepted_v18_l1",
            diagnostics={"baseline_preserved": True},
        )
    if not candidates:
        return _baseline_decision(
            baseline,
            reason="keep_v18_no_guarded_candidates",
            diagnostics={"candidate_count": 0},
        )

    ordered = sorted(
        candidates,
        key=lambda row: (
            -float(row.get("full_query_score", float("-inf"))),
            int(row.get("retrieval_rank", 10**9)),
            str(row.get("item_id", "")),
        ),
    )
    selected = ordered[0]
    selected_score = float(selected.get("full_query_score", float("-inf")))
    runner_up_score = (
        float(ordered[1].get("full_query_score", float("-inf")))
        if len(ordered) > 1
        else 0.0
    )
    baseline_score = float(baseline.get("top_score") or 0.0)
    requirements = [
        row
        for row in attribute_plan.get("requirements", [])
        if bool(row.get("mandatory", True))
    ]
    requirement_scores = selected.get("resolved_requirement_scores", {})
    passed_requirement_ids = [
        str(row.get("requirement_id"))
        for row in requirements
        if float(requirement_scores.get(str(row.get("requirement_id")), -1.0))
        >= float(row.get("threshold", 1.0))
    ]
    missing_requirement_ids = [
        str(row.get("requirement_id"))
        for row in requirements
        if str(row.get("requirement_id")) not in passed_requirement_ids
    ]
    relation_margins = {
        str(row.get("requirement_id")): float(row.get("margin", 0.0))
        for row in selected.get("contrastive_relation_evidence", [])
    }
    relation_ids = [
        str(row.get("requirement_id"))
        for row in requirements
        if row.get("kind") == "relation"
    ]
    weak_relation_ids = [
        requirement_id
        for requirement_id in relation_ids
        if relation_margins.get(requirement_id, float("-inf"))
        < policy.relation_margin_threshold
    ]
    diagnostics = {
        "candidate_count": len(ordered),
        "selected_full_query_score": round(selected_score, 8),
        "baseline_top_score": round(baseline_score, 8),
        "score_gain": round(selected_score - baseline_score, 8),
        "selection_margin": round(selected_score - runner_up_score, 8),
        "mandatory_requirement_count": len(requirements),
        "passed_requirement_ids": passed_requirement_ids,
        "missing_requirement_ids": missing_requirement_ids,
        "relation_requirement_ids": relation_ids,
        "weak_relation_ids": weak_relation_ids,
    }
    checks = (
        (
            len(requirements) >= policy.minimum_requirement_count,
            "insufficient_independent_requirements",
        ),
        (bool(attribute_plan.get("compositional")), "not_compositional"),
        (not missing_requirement_ids, "mandatory_requirement_missing"),
        (bool(relation_ids), "no_contrastive_relation_requirement"),
        (not weak_relation_ids, "relation_counterfactual_not_separated"),
        (
            selected_score >= policy.full_query_threshold,
            "full_query_below_threshold",
        ),
        (
            selected_score - baseline_score >= policy.score_gain_threshold,
            "insufficient_gain_over_v18",
        ),
        (
            selected_score - runner_up_score >= policy.selection_margin_threshold,
            "selection_margin_too_small",
        ),
    )
    for passed, reason in checks:
        if not passed:
            return _baseline_decision(
                baseline, reason=f"keep_v18_{reason}", diagnostics=diagnostics
            )
    return SelectiveInterventionDecision(
        accepted=True,
        selected_item_id=str(selected["item_id"]),
        intervention_applied=True,
        reason="admit_guarded_intervention",
        diagnostics=diagnostics,
    )
