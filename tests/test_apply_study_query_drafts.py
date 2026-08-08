from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts/apply_study_query_drafts.py"
    )
    spec = importlib.util.spec_from_file_location("apply_study_query_drafts", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source(authoring_id: str, stratum: str = "spatial_relation") -> dict[str, str]:
    return {
        "authoring_id": authoring_id,
        "stratum": stratum,
        "language_target": "zh",
    }


def _draft(authoring_id: str, kind: str = "relation") -> dict[str, str | int]:
    return {
        "authoring_id": authoring_id,
        "positive_query": "灰色电脑的键盘位于屏幕前方。",
        "hard_negative_query": "灰色电脑的键盘位于屏幕后方。",
        "changed_condition_kind": kind,
        "notes": "只改变前后关系。",
        "draft_author": "codex",
        "draft_version": 1,
    }


def test_draft_merge_preserves_approved_and_keeps_draft_unapproved() -> None:
    module = _load_module()
    approved = {
        "authoring_id": "source_001",
        "review_action": "approve",
        "positive_query": "绿色树木位于岩石山峰前方。",
        "hard_negative_query": "绿色树木位于岩石山峰后方。",
    }
    merged = module.build_draft_submissions(
        [_source("source_001"), _source("source_002")],
        [approved],
        [_draft("source_002")],
        drafted_at="2026-08-08T00:00:00+00:00",
    )
    assert merged[0] == approved
    assert merged[1]["review_action"] == "codex_draft"
    assert merged[1]["single_condition_confirmed"] is False
    assert merged[1]["reviewer_id"] == "codex_draft"


def test_draft_merge_requires_exact_pending_coverage() -> None:
    module = _load_module()
    with pytest.raises(ValueError, match="draft coverage mismatch"):
        module.build_draft_submissions(
            [_source("source_001"), _source("source_002")],
            [],
            [_draft("source_001")],
            drafted_at="2026-08-08T00:00:00+00:00",
        )


def test_draft_merge_rejects_kind_invalid_for_stratum() -> None:
    module = _load_module()
    with pytest.raises(ValueError, match="is invalid"):
        module.build_draft_submissions(
            [_source("source_001")],
            [],
            [_draft("source_001", kind="color")],
            drafted_at="2026-08-08T00:00:00+00:00",
        )
