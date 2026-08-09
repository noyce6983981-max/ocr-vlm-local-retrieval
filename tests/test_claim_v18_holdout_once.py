from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts/claim_v18_holdout_once.py"
    spec = importlib.util.spec_from_file_location("claim_v18_holdout_once", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_claim_preflight_does_not_need_holdout_rows() -> None:
    module = _load_module()
    claim = module.validate_claim_preflight(
        {
            "status": "approved_for_one_shot_v18_holdout_opening",
            "input_hashes": {"method_lock": "m", "freeze_receipt": "f"},
            "frozen_query_set_sha256": "queries",
        },
        {
            "status": "v18_method_locked_holdout_not_opened",
            "selected_method_id": "L1",
            "selected_parameters": {"top_k": 3},
        },
        {"query_set_sha256": "queries", "holdout_results_opened": False},
        authorization_sha256="a",
        method_lock_sha256="m",
        freeze_receipt_sha256="f",
    )
    assert claim["status"] == "v18_one_shot_holdout_claimed"
    assert claim["holdout_results_opened"] is True
