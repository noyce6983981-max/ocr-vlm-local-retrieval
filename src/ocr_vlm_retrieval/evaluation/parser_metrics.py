"""Independent requirement-level metrics for the deterministic V17 parser."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from ocr_vlm_retrieval.gating.contrastive_relations import (
    build_relation_counterfactual,
)

REFERENCE_FIELDS = {
    "object": "objects",
    "scene": "scenes",
    "color": "colors",
    "relation": "relations",
    "binding": "bindings",
    "ocr": "ocr_terms",
}


def _values(source: Mapping[str, Any], field: str) -> set[str]:
    values = source.get(field, [])
    if not isinstance(values, list) or not all(
        isinstance(value, str) and value.strip() for value in values
    ):
        raise ValueError(f"Parser reference field {field!r} must be a string list")
    return {str(value).strip() for value in values}


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _prf(
    true_positive: int, false_positive: int, false_negative: int
) -> dict[str, Any]:
    precision = _safe_ratio(true_positive, true_positive + false_positive)
    recall = _safe_ratio(true_positive, true_positive + false_negative)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_parser_predictions(
    references: Iterable[Mapping[str, Any]],
    predictions: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare parser output with an independently stored calibration reference."""

    reference_by_id = {
        str(row["query_id"]): dict(row)
        for row in references
    }
    prediction_by_id = {
        str(row["query_id"]): dict(row)
        for row in predictions
    }
    if not reference_by_id or set(reference_by_id) != set(prediction_by_id):
        raise ValueError("Parser reference and prediction query IDs must match")

    total_tp = 0
    total_fp = 0
    total_fn = 0
    per_kind: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    directional_total = 0
    directional_correct = 0
    binding_total = 0
    binding_correct = 0
    for kind, reference_field in REFERENCE_FIELDS.items():
        kind_tp = 0
        kind_fp = 0
        kind_fn = 0
        for query_id in sorted(reference_by_id):
            expected = _values(reference_by_id[query_id], reference_field)
            predicted = _values(prediction_by_id[query_id], reference_field)
            kind_tp += len(expected & predicted)
            kind_fp += len(predicted - expected)
            kind_fn += len(expected - predicted)
        per_kind[kind] = _prf(kind_tp, kind_fp, kind_fn)
        total_tp += kind_tp
        total_fp += kind_fp
        total_fn += kind_fn

    for query_id in sorted(reference_by_id):
        reference = reference_by_id[query_id]
        prediction = prediction_by_id[query_id]
        missing: dict[str, list[str]] = {}
        extra: dict[str, list[str]] = {}
        for _, field in REFERENCE_FIELDS.items():
            expected = _values(reference, field)
            predicted = _values(prediction, field)
            if expected - predicted:
                missing[field] = sorted(expected - predicted)
            if predicted - expected:
                extra[field] = sorted(predicted - expected)
        if missing or extra:
            failures.append(
                {
                    "query_id": query_id,
                    "missing": missing,
                    "extra": extra,
                }
            )

        expected_relations = _values(reference, "relations")
        if any(
            build_relation_counterfactual(value) is not None
            for value in expected_relations
        ):
            directional_total += 1
            directional_correct += int(
                expected_relations == _values(prediction, "relations")
            )
        expected_bindings = _values(reference, "bindings")
        if expected_bindings:
            binding_total += 1
            binding_correct += int(
                expected_bindings == _values(prediction, "bindings")
            )

    overall = _prf(total_tp, total_fp, total_fn)
    return {
        "query_count": len(reference_by_id),
        "requirement_micro": overall,
        "per_kind": per_kind,
        "directional_relation_query_accuracy": _safe_ratio(
            directional_correct, directional_total
        ),
        "directional_relation_query_count": directional_total,
        "attribute_binding_query_accuracy": _safe_ratio(
            binding_correct, binding_total
        ),
        "attribute_binding_query_count": binding_total,
        "extra_condition_rate": _safe_ratio(total_fp, total_tp + total_fp),
        "missing_condition_rate": _safe_ratio(total_fn, total_tp + total_fn),
        "exact_query_match_rate": 1.0 - len(failures) / len(reference_by_id),
        "failure_count": len(failures),
        "failures": failures,
    }
