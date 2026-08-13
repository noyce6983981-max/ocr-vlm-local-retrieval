"""Query-component rank aggregation for necessary-condition retrieval."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

SUPPORTED_METHODS = (
    "full_query",
    "condition_min_rank",
    "condition_mean_rank",
    "full_25_condition_min_75",
    "full_25_condition_mean_75",
)


def rank_percentiles(scores: Sequence[float]) -> list[float]:
    """Convert scores to deterministic [0, 1] rank percentiles."""

    if not scores:
        raise ValueError("at least one score is required")
    order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    denominator = max(len(scores) - 1, 1)
    result = [0.0] * len(scores)
    for rank, index in enumerate(order):
        result[index] = 1.0 - rank / denominator
    return result


def aggregate_rank_percentiles(
    full_scores: Sequence[float],
    condition_scores: Sequence[Sequence[float]],
    *,
    method: str,
) -> list[float]:
    """Aggregate one full-query ranking and two or more condition rankings."""

    if method not in SUPPORTED_METHODS:
        raise ValueError(f"unsupported condition aggregation method: {method}")
    if len(condition_scores) < 2:
        raise ValueError("at least two necessary-condition rankings are required")
    width = len(full_scores)
    if width == 0 or any(len(scores) != width for scores in condition_scores):
        raise ValueError("all score vectors must have the same non-zero length")
    full = rank_percentiles(full_scores)
    conditions = [rank_percentiles(scores) for scores in condition_scores]
    minimum = [min(scores[index] for scores in conditions) for index in range(width)]
    mean = [
        sum(scores[index] for scores in conditions) / len(conditions)
        for index in range(width)
    ]
    methods: Mapping[str, Sequence[float]] = {
        "full_query": full,
        "condition_min_rank": minimum,
        "condition_mean_rank": mean,
        "full_25_condition_min_75": [
            0.25 * full[index] + 0.75 * minimum[index] for index in range(width)
        ],
        "full_25_condition_mean_75": [
            0.25 * full[index] + 0.75 * mean[index] for index in range(width)
        ],
    }
    return list(methods[method])


def ranked_indices(scores: Sequence[float]) -> list[int]:
    """Return stable descending indices for an aggregated score vector."""

    return sorted(range(len(scores)), key=lambda index: (-scores[index], index))
