"""Tests for retrieval explanation helpers."""

from __future__ import annotations

from scripts.retrieval_explain import (
    build_retrieval_explanation,
    dominant_evidence_branch,
    extract_query_terms,
    matched_ocr_terms,
)


def test_extract_query_terms_removes_generic_question_words() -> None:
    terms = extract_query_terms("哪张图片展示了海边日落和水面倒影？")
    assert "海边日落" in terms
    assert "水面倒影" in terms
    assert "图片" not in terms


def test_matched_ocr_terms_returns_literal_evidence() -> None:
    matched = matched_ocr_terms(
        "哪份资料包含课程分数？",
        "课程名称 高等数学 分数 92",
    )
    assert "课程分数" not in matched
    assert "课程" in matched or "分数" in matched


def test_visual_branch_can_be_dominant() -> None:
    assert (
        dominant_evidence_branch(
            {
                "text_score": 0.2,
                "bm25_score": 0.0,
                "visual_score": 0.9,
            }
        )
        == "视觉内容"
    )


def test_reranker_is_reported_as_final_evidence() -> None:
    explanation = build_retrieval_explanation(
        "森林",
        "",
        {
            "text_score": 0.0,
            "bm25_score": 0.0,
            "visual_score": 0.7,
            "reranker_score": 0.82,
        },
    )
    assert explanation["dominant_branch"] == "多模态精排复核"
    assert explanation["ocr_excerpt"] == ""


def test_color_evidence_can_be_dominant() -> None:
    assert (
        dominant_evidence_branch(
            {
                "text_score": 0.1,
                "bm25_score": 0.0,
                "visual_score": 0.6,
                "color_score": 0.92,
            }
        )
        == "颜色特征"
    )


def test_pure_color_route_prefers_color_evidence_label() -> None:
    assert (
        dominant_evidence_branch(
            {
                "retrieval_route": "visual_discovery",
                "color_intent": "blue",
                "pure_color_query": True,
                "text_score": 1.0,
                "visual_score": 0.8,
                "color_score": 0.7,
            }
        )
        == "颜色特征"
    )


def test_compound_color_route_reports_both_constraints() -> None:
    assert (
        dominant_evidence_branch(
            {
                "retrieval_route": "visual_discovery",
                "color_intent": "blue",
                "pure_color_query": False,
            }
        )
        == "视觉内容 + 颜色约束"
    )


def test_unmatched_ocr_does_not_claim_arbitrary_excerpt_as_evidence() -> None:
    explanation = build_retrieval_explanation(
        "森林",
        "水",
        {
            "text_score": 0.99,
            "bm25_score": 0.0,
            "visual_score": 1.0,
        },
    )
    assert explanation["matched_terms"] == []
    assert explanation["ocr_excerpt"] == ""


def test_visual_metadata_route_discloses_metadata_evidence() -> None:
    row = {
        "retrieval_route": "visual_metadata",
        "metadata_score": 0.91,
        "visual_score": 0.45,
    }
    explanation = build_retrieval_explanation("登录界面截图", "", row)
    assert dominant_evidence_branch(row) == "文件名/元数据 + 视觉内容"
    assert explanation["metadata_score"] == 0.91
