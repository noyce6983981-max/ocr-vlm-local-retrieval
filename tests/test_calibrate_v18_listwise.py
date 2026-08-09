from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


def _load_module():
    path = PROJECT_ROOT / "scripts/calibrate_v18_listwise.py"
    spec = importlib.util.spec_from_file_location("calibrate_v18_listwise", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_l2_paired_weights_are_not_cross_multiplied() -> None:
    module = _load_module()
    parameters = module.expand_method_parameters(
        "L2",
        {
            "grid": {
                "top_k": [3, 5],
                "retrieval_weight": [0.2, 0.5],
                "verifier_weight": [0.8, 0.5],
                "score_threshold": [0.4, 0.6],
            },
            "paired_grid_fields": [["retrieval_weight", "verifier_weight"]],
        },
    )
    assert len(parameters) == 8
    assert {
        (row["retrieval_weight"], row["verifier_weight"]) for row in parameters
    } == {(0.2, 0.8), (0.5, 0.5)}


def test_locked_v18_grid_has_expected_configuration_count() -> None:
    module = _load_module()
    methods = json.loads(
        (PROJECT_ROOT / "config/studies/v18_methods.json").read_text(
            encoding="utf-8"
        )
    )
    count = sum(
        len(module.expand_method_parameters(method_id, definition))
        for method_id, definition in methods["methods"].items()
    )
    assert count == 265


def test_locked_selector_enforces_baseline_constraints() -> None:
    module = _load_module()
    evaluations = [
        {
            "method_id": "L0",
            "parameters": {"top_k": 3},
            "metrics": {
                "grouped_pair_accuracy": 0.5,
                "hard_negative_no_answer_false_accept_rate": 0.2,
                "positive_top3_conversion": 0.7,
            },
        },
        {
            "method_id": "L3",
            "parameters": {"top_k": 5, "selection_margin_threshold": 0.03},
            "metrics": {
                "grouped_pair_accuracy": 0.9,
                "hard_negative_no_answer_false_accept_rate": 0.3,
                "positive_top3_conversion": 0.8,
            },
        },
        {
            "method_id": "L2",
            "parameters": {"top_k": 3},
            "metrics": {
                "grouped_pair_accuracy": 0.7,
                "hard_negative_no_answer_false_accept_rate": 0.1,
                "positive_top3_conversion": 0.69,
            },
        },
    ]
    selected = module.select_locked_configuration(evaluations)
    assert selected["method_id"] == "L2"
    assert selected["eligible"] is True
