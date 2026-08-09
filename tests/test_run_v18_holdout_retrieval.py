from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from ocr_vlm_retrieval.studies.query_split import query_set_fingerprint


def _load_module():
    path = PROJECT_ROOT / "scripts/run_v18_holdout_retrieval.py"
    spec = importlib.util.spec_from_file_location("run_v18_holdout_retrieval", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_holdout_scope_requires_claimed_holdout_only() -> None:
    module = _load_module()
    queries = [
        {
            "query_id": "q1",
            "query": "query",
            "split": "holdout",
            "group_id": "g1",
            "query_role": "positive",
            "review_status": "human_query_approved",
        }
    ]
    module.validate_holdout_scope(
        queries,
        {
            "status": "v18_holdout_exported_after_one_shot_claim",
            "exported_split": "holdout",
            "holdout_results_opened": True,
            "retrieval_executed": False,
            "v17_artifacts_modified": False,
            "holdout_query_file_sha256": "queries-file",
            "holdout_query_set_sha256": query_set_fingerprint(queries),
            "method_lock_sha256": "lock",
        },
        {
            "status": "v18_method_locked_holdout_not_opened",
            "methods_sha256": "methods",
        },
        expected_count=1,
        query_file_sha256="queries-file",
        method_lock_sha256="lock",
        methods_sha256="methods",
    )


def test_holdout_scope_rejects_calibration_row() -> None:
    module = _load_module()
    queries = [
        {
            "query_id": "q1",
            "query": "query",
            "split": "calibration",
            "group_id": "g1",
            "query_role": "positive",
            "review_status": "human_query_approved",
        }
    ]
    with pytest.raises(ValueError, match="holdout rows only"):
        module.validate_holdout_scope(
            queries,
            {},
            {},
            expected_count=1,
            query_file_sha256="queries-file",
            method_lock_sha256="lock",
            methods_sha256="methods",
        )
