from __future__ import annotations

import pytest

from scripts.benchmark_v19_colqwen2_warm_latency import percentile


def test_nearest_rank_percentile() -> None:
    values = [50.0, 10.0, 40.0, 20.0, 30.0]
    assert percentile(values, 0.50) == 30.0
    assert percentile(values, 0.95) == 50.0


def test_percentile_rejects_invalid_input() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        percentile([], 0.5)
    with pytest.raises(ValueError, match="between zero and one"):
        percentile([1.0], 1.1)
