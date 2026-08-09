from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from ocr_vlm_retrieval.studies.query_split import query_set_fingerprint


def _load_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts/run_v18_calibration_retrieval.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_v18_calibration_retrieval", path
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _queries() -> list[dict[str, object]]:
    return [
        {
            "query_id": "v18_query_001",
            "query": "绿色树木位于岩石山峰前方。",
            "split": "calibration",
            "group_id": "group_001",
            "query_role": "positive",
            "review_status": "human_query_approved",
        },
        {
            "query_id": "v18_query_002",
            "query": "绿色树木位于岩石山峰后方。",
            "split": "calibration",
            "group_id": "group_001",
            "query_role": "single_condition_hard_negative",
            "review_status": "human_query_approved",
        },
    ]


def _receipt(queries: list[dict[str, object]]) -> dict[str, object]:
    return {
        "status": "prepared",
        "exported_split": "calibration",
        "calibration_query_count": 2,
        "holdout_query_count_not_exported": 2,
        "retrieval_executed": False,
        "holdout_results_opened": False,
        "v17_artifacts_modified": False,
        "methods_sha256": "methods",
        "calibration_query_set_sha256": query_set_fingerprint(queries),
    }


def test_v18_retrieval_preflight_accepts_isolated_calibration() -> None:
    module = _load_module()
    queries = _queries()
    module.validate_calibration_scope(
        queries,
        _receipt(queries),
        expected_count=2,
        expected_methods_sha256="methods",
    )


def test_v18_retrieval_preflight_rejects_holdout_row() -> None:
    module = _load_module()
    queries = _queries()
    receipt = _receipt(queries)
    queries[1]["split"] = "holdout"
    with pytest.raises(ValueError, match="calibration rows only"):
        module.validate_calibration_scope(
            queries,
            receipt,
            expected_count=2,
            expected_methods_sha256="methods",
        )


def test_resume_reuses_existing_validated_raw_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    raw_path = tmp_path / "raw.json"
    raw_path.write_text(
        '{"query":"query","attribute_coverage_active":false,'
        '"rankings":{"quality_hybrid":[{"item_id":"a"}]}}',
        encoding="utf-8",
    )

    def fail_run(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("retrieval should not run")

    monkeypatch.setattr(module, "run_live_search", fail_run)
    payload = module.obtain_raw_payload(
        raw_path=raw_path,
        query="query",
        version="v16",
        library_dir=tmp_path,
        policy_path=tmp_path / "policy.json",
        timeout=1,
        reuse_raw=False,
        resume=True,
    )
    assert payload["query"] == "query"


def test_resume_retrieves_missing_raw_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    raw_path = tmp_path / "raw.json"

    def fake_run(*args, **kwargs):  # type: ignore[no-untyped-def]
        assert kwargs["output_path"] == raw_path
        return {
            "query": "query",
            "attribute_coverage_active": True,
            "rankings": {"quality_hybrid": [{"item_id": "a"}]},
        }

    monkeypatch.setattr(module, "run_live_search", fake_run)
    payload = module.obtain_raw_payload(
        raw_path=raw_path,
        query="query",
        version="v17",
        library_dir=tmp_path,
        policy_path=tmp_path / "policy.json",
        timeout=1,
        reuse_raw=False,
        resume=True,
    )
    assert payload["attribute_coverage_active"] is True


def test_v18_retains_legacy_parser_miss_for_natural_compound_query() -> None:
    module = _load_module()
    module.validate_v18_raw_payload(
        {
            "query": "一张表格照片，频数为18，百分比为45%。",
            "attribute_coverage_active": False,
            "rankings": {"quality_hybrid": [{"item_id": "a"}]},
        },
        query="一张表格照片，频数为18，百分比为45%。",
        version="v17",
    )


def test_v18_ranking_merges_confidence_partitions_without_duplicates() -> None:
    module = _load_module()
    payload = {
        "rankings": {
            "quality_hybrid": [
                {"item_id": "a", "score": 0.9},
                {"item_id": "b", "score": 0.8},
            ]
        },
        "low_confidence_rankings": {
            "quality_hybrid": [
                {"item_id": "b", "score": 0.7},
                {"item_id": "c", "score": 0.6},
            ]
        },
    }
    ranking = module.ranking_for_v18(payload, "quality_hybrid")
    assert [row["item_id"] for row in ranking] == ["a", "b", "c"]


def test_empty_entity_result_uses_label_free_study_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    raw_path = tmp_path / "raw.json"
    raw_path.write_text(
        '{"query":"query","attribute_coverage_active":false,'
        '"rankings":{"quality_hybrid":[]}}',
        encoding="utf-8",
    )
    fallback_payload = {
        "query": "query",
        "attribute_coverage_active": False,
        "rankings": {"quality_hybrid": [{"item_id": "a"}]},
        "v18_study_route_fallback": {"labels_read": False},
    }
    monkeypatch.setattr(
        module, "run_study_route_fallback", lambda *args, **kwargs: fallback_payload
    )
    payload = module.obtain_raw_payload(
        raw_path=raw_path,
        query="query",
        version="v16",
        library_dir=tmp_path,
        policy_path=tmp_path / "policy.json",
        timeout=1,
        reuse_raw=False,
        resume=True,
    )
    assert payload["v18_study_route_fallback"]["labels_read"] is False
