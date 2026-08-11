"""Regression tests for the separated product and correction sites."""

from __future__ import annotations

import inspect
from pathlib import Path

from streamlit.testing.v1 import AppTest

from app import (
    correction_url,
    render_blind_query_collection,
    render_blind_study_review,
    v19_intent_runtime_settings,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_product_site_does_not_expose_correction_center() -> None:
    app = AppTest.from_file(
        str(PROJECT_ROOT / "app.py"),
        default_timeout=30,
    ).run()
    assert not app.exception
    assert app.radio[0].options == [
        "智能检索",
        "独立盲测",
        "批量入库",
        "资料库",
    ]


def test_internal_correction_site_renders_quality_workflow() -> None:
    app = AppTest.from_file(
        str(PROJECT_ROOT / "correction_app.py"),
        default_timeout=30,
    ).run()
    assert not app.exception
    markup = "\n".join(
        str(element.value) for element in app.markdown
    )
    assert "类别、质量与OCR纠错中心" in markup
    review_metrics = [
        metric
        for metric in app.metric
        if metric.label == "已人工裁决"
    ]
    empty_library_notes = [
        str(element.value) for element in app.info
    ]
    assert review_metrics or any(
        "还没有生成质量路由报告" in value
        for value in empty_library_notes
    )
    if review_metrics:
        assert len(review_metrics) == 1
        assert int(review_metrics[0].value) >= 0
    assert app.sidebar.radio[0].options == [
        "图片分类与OCR纠错",
        "检索真值审核",
        "独立盲测审核",
    ]


def test_search_result_correction_url_targets_internal_site() -> None:
    assert correction_url("user a/b") == (
        "http://127.0.0.1:8502/?item_id=user%20a%2Fb"
    )


def test_blind_collection_code_cannot_execute_retrieval() -> None:
    source = inspect.getsource(render_blind_query_collection)
    assert "execute_live_search" not in source
    assert "execute_next_blind_candidate" not in source
    assert "rankings" not in source
    assert "submit_blind_query" in source


def test_blind_reviewer_hides_system_and_author_signals() -> None:
    source = inspect.getsource(render_blind_study_review)
    assert 'candidate["system_accepted"]' not in source
    assert 'candidate["acceptance_reason"]' not in source
    assert 'candidate["retrieval_route"]' not in source
    assert 'row.get("expected_answerability"' not in source


def test_internal_site_renders_retrieval_truth_workflow() -> None:
    app = AppTest.from_file(
        str(PROJECT_ROOT / "correction_app.py"),
        default_timeout=30,
    ).run()
    app.sidebar.radio[0].set_value("检索真值审核").run()
    assert not app.exception
    subtitles = [str(element.value) for element in app.subheader]
    assert "检索答案纠错" in subtitles
    captions = [str(element.value) for element in app.caption]
    assert any("0 / 100" in value for value in captions)


def test_v19_ui_runtime_is_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("OCR_VLM_ENABLE_V19_INTENT_ROUTING", raising=False)
    monkeypatch.delenv("OCR_VLM_V19_INTENT_ROUTING_URL", raising=False)
    monkeypatch.delenv("OCR_VLM_V19_INTENT_TIMEOUT_SECONDS", raising=False)
    assert v19_intent_runtime_settings() == (
        False,
        "http://127.0.0.1:8765",
        2.0,
    )


def test_v19_ui_runtime_requires_explicit_environment_flag(monkeypatch) -> None:
    monkeypatch.setenv("OCR_VLM_ENABLE_V19_INTENT_ROUTING", "true")
    monkeypatch.setenv(
        "OCR_VLM_V19_INTENT_ROUTING_URL",
        "http://127.0.0.1:8877",
    )
    monkeypatch.setenv("OCR_VLM_V19_INTENT_TIMEOUT_SECONDS", "1.75")
    assert v19_intent_runtime_settings() == (
        True,
        "http://127.0.0.1:8877",
        1.75,
    )
