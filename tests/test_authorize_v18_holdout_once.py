from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load(name: str):
    path = Path(__file__).resolve().parents[1] / f"scripts/{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_authorization_accepts_locked_calibration_receipts() -> None:
    module = _load("authorize_v18_holdout_once")
    hashes = {key: "a" * 64 for key in {
        "freeze_receipt", "calibration_scope", "review_report",
        "final_judgments", "verification", "methods", "method_lock",
    }}
    authorization = module.build_authorization(
        freeze_receipt={
            "study_id": "v18",
            "query_set_sha256": "queries",
            "split_counts": {"calibration": 80, "holdout": 80},
            "retrieval_executed": False,
            "holdout_results_opened": False,
        },
        calibration_scope={
            "status": "prepared",
            "exported_split": "calibration",
            "holdout_results_opened": False,
        },
        review_report={
            "status": "v18_calibration_review_complete",
            "conflict_adjudication_required": False,
            "double_reviewed_query_count": 24,
            "reviewer_ids_distinct": True,
        },
        method_lock={
            "status": "v18_method_locked_holdout_not_opened",
            "selected_method_id": "L1",
            "selected_parameters": {"top_k": 3},
            "selected_constraints": {"safe": True},
            "judgments_sha256": hashes["final_judgments"],
            "verification_sha256": hashes["verification"],
            "methods_sha256": hashes["methods"],
            "holdout_results_opened": False,
            "holdout_retrieval_executed": False,
        },
        input_hashes=hashes,
    )
    assert authorization["status"] == "approved_for_one_shot_v18_holdout_opening"
    assert authorization["holdout_results_opened"] is False


def test_authorization_rejects_unlocked_method() -> None:
    module = _load("authorize_v18_holdout_once")
    with pytest.raises(ValueError, match="method lock"):
        module.build_authorization(
            freeze_receipt={
                "split_counts": {"calibration": 80, "holdout": 80},
                "retrieval_executed": False,
                "holdout_results_opened": False,
            },
            calibration_scope={
                "status": "prepared",
                "exported_split": "calibration",
                "holdout_results_opened": False,
            },
            review_report={
                "status": "v18_calibration_review_complete",
                "conflict_adjudication_required": False,
                "double_reviewed_query_count": 24,
                "reviewer_ids_distinct": True,
            },
            method_lock={"status": "open"},
            input_hashes={},
        )
