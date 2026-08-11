from __future__ import annotations

import pytest

from ocr_vlm_retrieval.routing.evaluator import (
    RoutingSample,
    evaluate_routing,
)


def test_evaluator_counts_missing_predictions_and_operational_rates() -> None:
    samples = [
        RoutingSample(
            gold_route="text_evidence",
            predicted_route="text_evidence",
            rule_route="text_evidence",
            llm_invoked=False,
        ),
        RoutingSample(
            gold_route="visual_discovery",
            predicted_route="mixed",
            rule_route="mixed",
            llm_invoked=True,
            used_fallback=True,
            fallback_error_type="IntentSchemaError",
        ),
        RoutingSample(
            gold_route="mixed",
            predicted_route=None,
            rule_route="mixed",
            llm_invoked=True,
            fallback_error_type="TimeoutError",
        ),
    ]
    metrics = evaluate_routing(samples)
    assert metrics.query_count == 3
    assert metrics.accuracy == pytest.approx(1 / 3)
    assert metrics.llm_call_rate == pytest.approx(2 / 3)
    assert metrics.fallback_rate == pytest.approx(1 / 3)
    assert metrics.invalid_json_rate_per_llm_call == pytest.approx(1 / 2)
    assert metrics.per_route["mixed"].precision == 0.0
    assert metrics.per_route["mixed"].support == 1


def test_evaluator_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        evaluate_routing([])
