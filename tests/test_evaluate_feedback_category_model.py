"""Tests for frozen feedback-model evaluation guards."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.evaluate_feedback_category_model import (
    conservative_predictions,
    ensure_split_allowed,
    expected_calibration_error,
)


def test_test_split_is_locked_by_default() -> None:
    with pytest.raises(ValueError, match="Test evaluation is locked"):
        ensure_split_allowed("test", allow_test=False)
    ensure_split_allowed("validation", allow_test=False)
    ensure_split_allowed("test", allow_test=True)


def test_expected_calibration_error_is_zero_for_perfect_confidence() -> None:
    probabilities = np.array(
        [[1.0, 0.0], [0.0, 1.0]], dtype=np.float32
    )
    truth = np.array(["a", "b"], dtype=object)
    predictions = np.array(["a", "b"], dtype=object)
    assert (
        expected_calibration_error(
            probabilities, truth, predictions, bins=5
        )
        == 0.0
    )


def test_expected_calibration_error_detects_overconfidence() -> None:
    probabilities = np.array(
        [[0.9, 0.1], [0.9, 0.1]], dtype=np.float32
    )
    truth = np.array(["a", "b"], dtype=object)
    predictions = np.array(["a", "a"], dtype=object)
    error = expected_calibration_error(
        probabilities, truth, predictions, bins=5
    )
    assert 0.39 < error < 0.41


def test_conservative_gate_only_overrides_above_threshold() -> None:
    baseline = np.array(["a", "a", "b"], dtype=object)
    model_predictions = np.array(["b", "b", "b"], dtype=object)
    probabilities = np.array(
        [
            [0.1, 0.9],
            [0.3, 0.7],
            [0.2, 0.8],
        ],
        dtype=np.float32,
    )
    predictions, override_mask = conservative_predictions(
        baseline,
        model_predictions,
        probabilities,
        threshold=0.8,
    )
    assert list(predictions) == ["b", "a", "b"]
    assert list(override_mask) == [True, False, False]
