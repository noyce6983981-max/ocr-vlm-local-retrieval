from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from ocr_vlm_retrieval.evaluation.parser_metrics import (
    evaluate_parser_predictions,
)
from scripts.evaluate_v17_parser import parser_predictions


def row(query_id: str) -> dict:
    return {
        "query_id": query_id,
        "query": "what is behind the pole?",
        "split": "calibration",
        "objects": ["object", "the pole"],
        "scenes": [],
        "colors": [],
        "relations": ["object behind the pole"],
        "bindings": [],
        "ocr_terms": [],
    }


def test_parser_metrics_separate_missing_and_extra_requirements() -> None:
    reference = row("q1")
    prediction = {
        **row("q1"),
        "objects": ["object"],
        "relations": ["object in front of the pole"],
    }

    report = evaluate_parser_predictions([reference], [prediction])

    assert report["requirement_micro"]["false_positive"] == 1
    assert report["requirement_micro"]["false_negative"] == 2
    assert report["directional_relation_query_accuracy"] == 0.0
    assert report["missing_condition_rate"] > 0
    assert report["extra_condition_rate"] > 0


def test_parser_metrics_require_identical_query_ids() -> None:
    with pytest.raises(ValueError, match="query IDs"):
        evaluate_parser_predictions([row("q1")], [row("q2")])


def test_parser_predictions_refuse_holdout_rows() -> None:
    reference = row("q1")
    reference["split"] = "holdout"

    with pytest.raises(ValueError, match="calibration-only"):
        parser_predictions([reference], {})
