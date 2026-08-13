from __future__ import annotations

import pytest

from ocr_vlm_retrieval.runtime.condition_aggregation import (
    aggregate_rank_percentiles,
    rank_percentiles,
    ranked_indices,
)


def test_rank_percentiles_is_stable_for_ties() -> None:
    assert rank_percentiles([0.7, 0.7, 0.2]) == [1.0, 0.5, 0.0]


def test_min_rank_prefers_candidate_covering_both_conditions() -> None:
    scores = aggregate_rank_percentiles(
        [0.9, 0.8, 0.7],
        [[0.9, 0.8, 0.1], [0.1, 0.8, 0.9]],
        method="condition_min_rank",
    )
    assert ranked_indices(scores)[0] == 1


def test_aggregation_rejects_one_condition() -> None:
    with pytest.raises(ValueError, match="at least two"):
        aggregate_rank_percentiles([1.0], [[1.0]], method="condition_min_rank")
