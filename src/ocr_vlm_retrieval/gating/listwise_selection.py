"""Calibration-safe candidate selection for the V18 L0-L3 comparison."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from ocr_vlm_retrieval.gating.attribute_coverage import AttributePlan

METHOD_IDS = frozenset({"L0", "L1", "L2", "L3"})


@dataclass(frozen=True)
class CandidateEvidence:
    """All method-independent evidence available for one ranked candidate."""

    item_id: str
    retrieval_rank: int
    retrieval_score: float
    full_query_score: float
    resolved_requirement_scores: Mapping[str, float]
    contrastive_relation_margins: Mapping[str, float]


@dataclass(frozen=True)
class CandidateAssessment:
    """One candidate's method-specific score and coverage diagnostics."""

    item_id: str
    retrieval_rank: int
    score: float
    full_query_score: float
    coverage_ratio: float
    weakest_requirement_strength: float
    requirement_geometric_mean: float
    missing_requirement_ids: tuple[str, ...]
    eligible: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SelectionDecision:
    """Auditable best-or-none decision for a single query."""

    method_id: str
    accepted: bool
    selected_item_id: str | None
    selected_score: float | None
    runner_up_score: float | None
    selection_margin: float | None
    reason: str
    assessments: tuple[CandidateAssessment, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "accepted": self.accepted,
            "selected_item_id": self.selected_item_id,
            "selected_score": self.selected_score,
            "runner_up_score": self.runner_up_score,
            "selection_margin": self.selection_margin,
            "reason": self.reason,
            "assessments": [row.to_dict() for row in self.assessments],
        }


def _parameter_float(parameters: Mapping[str, Any], name: str) -> float:
    value = float(parameters[name])
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _top_candidates(
    candidates: Sequence[CandidateEvidence], parameters: Mapping[str, Any]
) -> list[CandidateEvidence]:
    top_k = int(parameters["top_k"])
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if any(not row.item_id or row.retrieval_rank < 1 for row in candidates):
        raise ValueError("candidates need item IDs and positive retrieval ranks")
    if len({row.item_id for row in candidates}) != len(candidates):
        raise ValueError("candidate item IDs must be unique")
    if len({row.retrieval_rank for row in candidates}) != len(candidates):
        raise ValueError("candidate retrieval ranks must be unique")
    return sorted(candidates, key=lambda row: row.retrieval_rank)[:top_k]


def _rank_strength(rank: int) -> float:
    return 1.0 / math.log2(rank + 1.0)


def _requirement_strength(
    raw_score: float, threshold: float, full_score: float
) -> float:
    lower = threshold - (full_score - threshold)
    denominator = full_score - lower
    if denominator <= 0.0:
        raise ValueError("requirement full_score must exceed its threshold")
    return min(1.0, max(0.0, (raw_score - lower) / denominator))


def _empty_assessment(
    candidate: CandidateEvidence, *, score: float
) -> CandidateAssessment:
    return CandidateAssessment(
        item_id=candidate.item_id,
        retrieval_rank=candidate.retrieval_rank,
        score=round(score, 8),
        full_query_score=round(candidate.full_query_score, 8),
        coverage_ratio=1.0,
        weakest_requirement_strength=1.0,
        requirement_geometric_mean=1.0,
        missing_requirement_ids=(),
        eligible=True,
    )


def _coverage_assessment(
    candidate: CandidateEvidence,
    plan: AttributePlan,
    *,
    full_query_floor: float,
    relation_margin_threshold: float,
    weights: Mapping[str, float],
) -> CandidateAssessment:
    mandatory = [row for row in plan.requirements if row.mandatory]
    if not mandatory:
        raise ValueError("L3 requires at least one mandatory condition")

    strengths: list[float] = []
    missing: list[str] = []
    for requirement in mandatory:
        raw = candidate.resolved_requirement_scores.get(requirement.requirement_id)
        if raw is None:
            strength = 0.0
            passed = False
        else:
            strength = _requirement_strength(
                float(raw), requirement.threshold, requirement.full_score
            )
            passed = float(raw) >= requirement.threshold
        if requirement.kind == "relation":
            contrastive_margin = candidate.contrastive_relation_margins.get(
                requirement.requirement_id
            )
            if (
                contrastive_margin is not None
                and float(contrastive_margin) < relation_margin_threshold
            ):
                passed = False
                strength = min(strength, 0.5)
        strengths.append(strength)
        if not passed:
            missing.append(requirement.requirement_id)

    coverage_ratio = (len(mandatory) - len(missing)) / len(mandatory)
    weakest = min(strengths)
    geometric = math.exp(
        sum(math.log(max(value, 1e-6)) for value in strengths) / len(strengths)
    )
    full_strength = min(1.0, max(0.0, candidate.full_query_score))
    score = (
        float(weights["retrieval"]) * _rank_strength(candidate.retrieval_rank)
        + float(weights["full_query"]) * full_strength
        + float(weights["geometric_mean"]) * geometric
        + float(weights["weakest"]) * weakest
    )
    eligible = not missing and candidate.full_query_score >= full_query_floor
    return CandidateAssessment(
        item_id=candidate.item_id,
        retrieval_rank=candidate.retrieval_rank,
        score=round(score, 8),
        full_query_score=round(candidate.full_query_score, 8),
        coverage_ratio=round(coverage_ratio, 8),
        weakest_requirement_strength=round(weakest, 8),
        requirement_geometric_mean=round(geometric, 8),
        missing_requirement_ids=tuple(missing),
        eligible=eligible,
    )


def _validate_weights(weights: Mapping[str, Any]) -> dict[str, float]:
    expected = {"retrieval", "full_query", "geometric_mean", "weakest"}
    if set(weights) != expected:
        raise ValueError(f"L3 weights must be exactly {sorted(expected)}")
    resolved = {key: float(value) for key, value in weights.items()}
    if any(value < 0.0 or not math.isfinite(value) for value in resolved.values()):
        raise ValueError("L3 weights must be finite and non-negative")
    if not math.isclose(sum(resolved.values()), 1.0, abs_tol=1e-9):
        raise ValueError("L3 weights must sum to one")
    return resolved


def _decision(
    method_id: str,
    assessments: Sequence[CandidateAssessment],
    *,
    accepted: bool,
    reason: str,
    selected: CandidateAssessment | None,
    runner_up: CandidateAssessment | None,
) -> SelectionDecision:
    selected_score = selected.score if selected is not None else None
    runner_up_score = runner_up.score if runner_up is not None else None
    margin = (
        round(selected_score - runner_up_score, 8)
        if selected_score is not None and runner_up_score is not None
        else selected_score
    )
    return SelectionDecision(
        method_id=method_id,
        accepted=accepted,
        selected_item_id=selected.item_id if accepted and selected else None,
        selected_score=selected_score,
        runner_up_score=runner_up_score,
        selection_margin=margin,
        reason=reason,
        assessments=tuple(assessments),
    )


def select_candidate(
    method_id: str,
    candidates: Sequence[CandidateEvidence],
    parameters: Mapping[str, Any],
    *,
    attribute_plan: AttributePlan | None = None,
) -> SelectionDecision:
    """Apply one preregistered V18 selector to a shared ranked candidate list."""

    if method_id not in METHOD_IDS:
        raise ValueError(f"unsupported V18 method: {method_id}")
    ranked = _top_candidates(candidates, parameters)
    if not ranked:
        return _decision(
            method_id,
            (),
            accepted=False,
            reason="no_candidates",
            selected=None,
            runner_up=None,
        )

    if method_id == "L0":
        threshold = _parameter_float(parameters, "full_query_threshold")
        assessments = [
            _empty_assessment(row, score=row.full_query_score) for row in ranked
        ]
        selected = next(
            (row for row in assessments if row.full_query_score >= threshold), None
        )
        return _decision(
            method_id,
            assessments,
            accepted=selected is not None,
            reason="accepted" if selected is not None else "no_full_query_pass",
            selected=selected,
            runner_up=None,
        )

    if method_id == "L1":
        threshold = _parameter_float(parameters, "full_query_threshold")
        assessments = [
            _empty_assessment(row, score=row.full_query_score) for row in ranked
        ]
        ordered = sorted(
            assessments, key=lambda row: (-row.score, row.retrieval_rank)
        )
        selected = ordered[0]
        runner_up = ordered[1] if len(ordered) > 1 else None
        accepted = selected.full_query_score >= threshold
        return _decision(
            method_id,
            assessments,
            accepted=accepted,
            reason="accepted" if accepted else "full_query_below_threshold",
            selected=selected,
            runner_up=runner_up,
        )

    if method_id == "L2":
        retrieval_weight = _parameter_float(parameters, "retrieval_weight")
        verifier_weight = _parameter_float(parameters, "verifier_weight")
        if not math.isclose(retrieval_weight + verifier_weight, 1.0, abs_tol=1e-9):
            raise ValueError("L2 retrieval and verifier weights must sum to one")
        if retrieval_weight < 0.0 or verifier_weight < 0.0:
            raise ValueError("L2 weights must be non-negative")
        assessments = [
            _empty_assessment(
                row,
                score=(
                    retrieval_weight * _rank_strength(row.retrieval_rank)
                    + verifier_weight * min(1.0, max(0.0, row.full_query_score))
                ),
            )
            for row in ranked
        ]
        ordered = sorted(
            assessments, key=lambda row: (-row.score, row.retrieval_rank)
        )
        selected = ordered[0]
        runner_up = ordered[1] if len(ordered) > 1 else None
        score_threshold = _parameter_float(parameters, "score_threshold")
        verifier_floor = _parameter_float(parameters, "full_query_floor")
        accepted = (
            selected.score >= score_threshold
            and selected.full_query_score >= verifier_floor
        )
        reason = "accepted"
        if selected.full_query_score < verifier_floor:
            reason = "full_query_below_floor"
        elif selected.score < score_threshold:
            reason = "fused_score_below_threshold"
        return _decision(
            method_id,
            assessments,
            accepted=accepted,
            reason=reason,
            selected=selected,
            runner_up=runner_up,
        )

    if attribute_plan is None:
        raise ValueError("L3 requires an attribute plan")
    weights = _validate_weights(parameters["weights"])
    full_query_floor = _parameter_float(parameters, "full_query_floor")
    relation_margin_threshold = _parameter_float(
        parameters, "relation_margin_threshold"
    )
    assessments = [
        _coverage_assessment(
            row,
            attribute_plan,
            full_query_floor=full_query_floor,
            relation_margin_threshold=relation_margin_threshold,
            weights=weights,
        )
        for row in ranked
    ]
    eligible = sorted(
        (row for row in assessments if row.eligible),
        key=lambda row: (-row.score, row.retrieval_rank),
    )
    if not eligible:
        return _decision(
            method_id,
            assessments,
            accepted=False,
            reason="no_candidate_covers_all_requirements",
            selected=None,
            runner_up=None,
        )
    selected = eligible[0]
    runner_up = eligible[1] if len(eligible) > 1 else None
    score_threshold = _parameter_float(parameters, "score_threshold")
    margin_threshold = _parameter_float(parameters, "selection_margin_threshold")
    margin = selected.score - runner_up.score if runner_up else selected.score
    accepted = selected.score >= score_threshold and margin >= margin_threshold
    reason = "accepted"
    if selected.score < score_threshold:
        reason = "listwise_score_below_threshold"
    elif margin < margin_threshold:
        reason = "selection_margin_too_small"
    return _decision(
        method_id,
        assessments,
        accepted=accepted,
        reason=reason,
        selected=selected,
        runner_up=runner_up,
    )
