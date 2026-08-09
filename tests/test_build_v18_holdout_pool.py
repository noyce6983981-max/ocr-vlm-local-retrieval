from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts/build_v18_holdout_pool.py"
    spec = importlib.util.spec_from_file_location("build_v18_holdout_pool", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_holdout_pool_rejects_calibration_queries() -> None:
    module = _load_module()
    with pytest.raises(ValueError, match="holdout rows only"):
        module.validate_inputs(
            [{"query_id": "q1", "split": "calibration"}],
            scope_receipt={},
            retrieval_receipt={},
            method_lock={},
            scope_receipt_path=Path("scope"),
            method_lock_path=Path("lock"),
            methods_path=Path("methods"),
            expected_count=1,
        )
