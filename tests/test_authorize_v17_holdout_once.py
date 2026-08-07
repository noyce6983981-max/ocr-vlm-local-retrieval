from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.authorize_v17_holdout_once import build_authorization
from tests.test_v17_locked_holdout import fixtures


def adjudication_report() -> dict:
    return {
        "status": "holdout_human_adjudication_complete",
        "primary_reviewer_count": 2,
        "independent_adjudicator_count": 1,
        "independent_reviews": True,
        "full_candidate_coverage": True,
        "blinded_to_method_outputs": True,
        "model_assisted_labels_used": False,
        "conflict_adjudication_complete": True,
        "adjudicator_independent_of_primary_reviewers": True,
    }


def test_authorizes_exact_structurally_valid_inputs() -> None:
    lock, verification, judgments, baseline = fixtures()

    result = build_authorization(
        method_lock=lock,
        method_lock_sha256="lock",
        verification=verification,
        judgments=judgments,
        baseline_rows=baseline,
        adjudication_report=adjudication_report(),
        input_hashes={
            "verification": "a" * 64,
            "judgments": "b" * 64,
            "baseline_records": "c" * 64,
        },
    )

    assert result["status"] == "approved_for_one_shot_holdout_evaluation"
    assert result["review_protocol"]["model_assisted_labels_used"] is False
    assert result["structural_preflight"]["metrics_computed"] is False


def test_rejects_model_assisted_holdout_labels() -> None:
    lock, verification, judgments, baseline = fixtures()
    report = copy.deepcopy(adjudication_report())
    report["model_assisted_labels_used"] = True

    with pytest.raises(ValueError, match="model_assisted_labels_used"):
        build_authorization(
            method_lock=lock,
            method_lock_sha256="lock",
            verification=verification,
            judgments=judgments,
            baseline_rows=baseline,
            adjudication_report=report,
            input_hashes={
                "verification": "a" * 64,
                "judgments": "b" * 64,
                "baseline_records": "c" * 64,
            },
        )
