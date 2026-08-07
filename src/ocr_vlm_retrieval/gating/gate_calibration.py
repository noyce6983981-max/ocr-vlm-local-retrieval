"""Top-K operating points for V17 candidate verification."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from ocr_vlm_retrieval.evaluation.judgments import (
    NO_RELEVANT_CANDIDATE_IN_POOL,
    RELEVANT_CANDIDATE_IN_POOL,
)
from ocr_vlm_retrieval.gating.contrastive_relations import (
    relation_evidence_passes,
)

AGGREGATORS = ("full_query", "attribute_mean", "attribute_geometric", "weakest")
METHOD_PRIORITY = {
    "attribute_geometric": 0,
    "attribute_mean": 1,
    "full_query": 2,
    "weakest": 3,
}


def mean(values: Iterable[float]) -> float:
    collected = list(values)
    return sum(collected) / len(collected) if collected else 0.0


def geometric_mean(values: Iterable[float]) -> float:
    collected = [max(float(value), 1e-9) for value in values]
    if not collected:
        return 0.0
    return math.exp(sum(math.log(value) for value in collected) / len(collected))


def candidate_method_scores(candidate: Mapping[str, Any]) -> dict[str, float]:
    source = candidate.get(
        "resolved_requirement_scores",
        candidate.get("requirement_scores", {}),
    )
    if not isinstance(source, Mapping) or not source:
        raise ValueError("Candidate has no requirement scores")
    requirement_scores = [float(value) for value in source.values()]
    return {
        "full_query": float(candidate["full_query_score"]),
        "attribute_mean": mean(requirement_scores),
        "attribute_geometric": geometric_mean(requirement_scores),
        "weakest": min(requirement_scores),
    }


def contrastive_relations_pass(
    candidate: Mapping[str, Any],
    *,
    enabled: bool,
    margin_threshold: float,
) -> bool:
    if not enabled:
        return True
    evidence_rows = candidate.get("contrastive_relation_evidence", [])
    if not isinstance(evidence_rows, list):
        raise ValueError("contrastive_relation_evidence must be a list")
    return all(
        relation_evidence_passes(
            positive_score=float(row["positive_score"]),
            negative_score=float(row["negative_score"]),
            absolute_threshold=float(row["absolute_threshold"]),
            margin_threshold=margin_threshold,
        )
        for row in evidence_rows
    )


def select_rank_first_passed(
    candidates: Sequence[Mapping[str, Any]],
    *,
    top_k: int,
    method: str,
    threshold: float,
    contrastive_relations: bool,
    relation_margin_threshold: float,
) -> dict[str, Any]:
    """Return the highest retrieval-ranked candidate that passes verification."""

    if top_k < 1:
        raise ValueError("top_k must be positive")
    if method not in AGGREGATORS:
        raise ValueError(f"Unsupported aggregation method: {method}")
    inspected = list(candidates[:top_k])
    for rank, candidate in enumerate(inspected, start=1):
        score = candidate_method_scores(candidate)[method]
        relation_passed = contrastive_relations_pass(
            candidate,
            enabled=contrastive_relations,
            margin_threshold=relation_margin_threshold,
        )
        if score >= threshold and relation_passed:
            return {
                "accepted": True,
                "selected_item_id": str(candidate.get("item_id", "")),
                "selected_rank": rank,
                "selected_score": score,
                "selected_relevant": bool(candidate.get("relevant", False)),
                "verified_candidate_count": rank,
                "relation_passed": relation_passed,
            }
    return {
        "accepted": False,
        "selected_item_id": None,
        "selected_rank": None,
        "selected_score": None,
        "selected_relevant": False,
        "verified_candidate_count": len(inspected),
        "relation_passed": None,
    }


def evaluate_topk_operating_point(
    rows: Sequence[Mapping[str, Any]],
    *,
    top_k: int,
    method: str,
    threshold: float,
    contrastive_relations: bool,
    relation_margin_threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not rows:
        raise ValueError("At least one calibration row is required")
    decisions: list[dict[str, Any]] = []
    for row in rows:
        candidates = row.get("candidates", [])
        if not isinstance(candidates, list) or len(candidates) < top_k:
            query_id = row.get("query_id")
            raise ValueError(
                f"Calibration row {query_id} has fewer than {top_k} candidates"
            )
        decision = select_rank_first_passed(
            candidates,
            top_k=top_k,
            method=method,
            threshold=threshold,
            contrastive_relations=contrastive_relations,
            relation_margin_threshold=relation_margin_threshold,
        )
        relevant_in_top_k = any(
            bool(candidate.get("relevant", False)) for candidate in candidates[:top_k]
        )
        pool_relevance = str(row["pool_relevance"])
        correct = bool(
            (
                pool_relevance == RELEVANT_CANDIDATE_IN_POOL
                and decision["accepted"]
                and decision["selected_relevant"]
            )
            or (
                pool_relevance == NO_RELEVANT_CANDIDATE_IN_POOL
                and not decision["accepted"]
            )
        )
        decisions.append(
            {
                "query_id": row["query_id"],
                "group_id": row["group_id"],
                "pool_relevance": pool_relevance,
                "relevant_in_top_k": relevant_in_top_k,
                "pool_conditioned_correct": correct,
                **decision,
            }
        )

    relevant_rows = [
        row
        for row in decisions
        if row["pool_relevance"] == RELEVANT_CANDIDATE_IN_POOL
    ]
    no_relevant_rows = [
        row
        for row in decisions
        if row["pool_relevance"] == NO_RELEVANT_CANDIDATE_IN_POOL
    ]
    if not relevant_rows or not no_relevant_rows:
        raise ValueError("Calibration needs both pooled relevance classes")
    false_accepts = sum(bool(row["accepted"]) for row in no_relevant_rows)
    selected_relevant = sum(bool(row["selected_relevant"]) for row in relevant_rows)
    metrics = {
        "top_k": top_k,
        "selection_policy": "highest_retrieval_rank_among_passed",
        "method": method,
        "threshold": round(threshold, 2),
        "contrastive_relations": contrastive_relations,
        "relation_margin_threshold": round(relation_margin_threshold, 3),
        "accepted_query_count": sum(bool(row["accepted"]) for row in decisions),
        "acceptance_coverage": mean(bool(row["accepted"]) for row in decisions),
        "retrieval_recall_at_k": mean(
            bool(row["relevant_in_top_k"]) for row in relevant_rows
        ),
        "selected_relevant_query_count": selected_relevant,
        "selected_relevant_query_rate": selected_relevant / len(relevant_rows),
        "pool_conditioned_false_accept_count": false_accepts,
        "pool_conditioned_false_accept_rate": false_accepts
        / len(no_relevant_rows),
        "no_relevant_in_pool_rejection_rate": 1.0
        - false_accepts / len(no_relevant_rows),
        "pool_conditioned_end_to_end_accuracy": mean(
            bool(row["pool_conditioned_correct"]) for row in decisions
        ),
        "mean_verified_candidate_count": mean(
            int(row["verified_candidate_count"]) for row in decisions
        ),
        "degenerate_reject_all": not any(row["accepted"] for row in decisions),
    }
    return metrics, decisions


def mark_eligibility(
    metrics: Mapping[str, Any],
    *,
    baseline_false_accept_rate: float,
    minimum_false_accept_relative_reduction: float,
    minimum_selected_relevant_rate: float,
) -> dict[str, Any]:
    result = dict(metrics)
    current_far = float(result["pool_conditioned_false_accept_rate"])
    relative_reduction = (
        (baseline_false_accept_rate - current_far) / baseline_false_accept_rate
        if baseline_false_accept_rate
        else None
    )
    result["pool_conditioned_false_accept_relative_reduction_vs_v16"] = (
        relative_reduction
    )
    result["eligible"] = bool(
        relative_reduction is not None
        and relative_reduction >= minimum_false_accept_relative_reduction
        and float(result["selected_relevant_query_rate"])
        >= minimum_selected_relevant_rate
        and not result["degenerate_reject_all"]
    )
    return result


def select_near_optimal_operating_point(
    points: Sequence[Mapping[str, Any]],
    *,
    accuracy_tolerance: float = 0.03,
) -> dict[str, Any]:
    """Prefer the smallest K whose accuracy is within tolerance of the best."""

    eligible = [dict(point) for point in points if point.get("eligible")]
    if not eligible:
        raise ValueError("No non-degenerate gate satisfies calibration constraints")
    best_accuracy = max(
        float(point["pool_conditioned_end_to_end_accuracy"])
        for point in eligible
    )
    near_optimal = [
        point
        for point in eligible
        if float(point["pool_conditioned_end_to_end_accuracy"])
        >= best_accuracy - accuracy_tolerance
    ]
    minimum_k = min(int(point["top_k"]) for point in near_optimal)
    smallest_k = [point for point in near_optimal if int(point["top_k"]) == minimum_k]
    selected = min(
        smallest_k,
        key=lambda point: (
            -float(point["pool_conditioned_end_to_end_accuracy"]),
            -float(point["selected_relevant_query_rate"]),
            float(point["pool_conditioned_false_accept_rate"]),
            0 if bool(point["contrastive_relations"]) else 1,
            METHOD_PRIORITY[str(point["method"])],
            float(point["threshold"]),
            float(point["relation_margin_threshold"]),
        ),
    )
    selected["best_eligible_accuracy"] = best_accuracy
    selected["accuracy_tolerance_for_smallest_k"] = accuracy_tolerance
    return selected
