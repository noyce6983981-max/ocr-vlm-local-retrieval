from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts/score_v18_holdout_verification.py"
    )
    spec = importlib.util.spec_from_file_location(
        "score_v18_holdout_verification", path
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_holdout_verifier_rejects_calibration_queries() -> None:
    module = _load_module()
    with pytest.raises(ValueError, match="holdout queries only"):
        module.validate_scope(
            [{"query_id": "q1", "split": "calibration"}],
            [],
            [],
            scope_receipt={},
            pool_receipt={},
            retrieval_receipt={},
            method_lock={},
            scope_receipt_path=Path("scope"),
            pool_path=Path("pool"),
            pool_receipt_path=Path("pool-receipt"),
            retrieval_receipt_path=Path("retrieval"),
            method_lock_path=Path("lock"),
            ranking_path=Path("ranking"),
            expected_count=1,
        )
