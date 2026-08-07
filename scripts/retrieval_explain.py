"""Build concise, auditable explanations for multimodal retrieval results."""

from __future__ import annotations

import re
from typing import Any


MIXED_TOKEN_PATTERN = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff]+|[A-Za-z0-9][A-Za-z0-9._:+/-]*"
)
CHINESE_SPLIT_PATTERN = re.compile(
    r"哪张|哪份|哪些|图片|图像|照片|文档|资料|页面|"
    r"展示|显示|包含|出现|找到|查找|搜索|内容|"
    r"是什么|什么|的|了|和|与|中|里|上"
)
GENERIC_TERMS = {
    "这张",
    "这个",
    "那个",
    "一张",
    "一份",
    "有关",
    "相关",
}


def extract_query_terms(query: str) -> list[str]:
    """Extract useful literal evidence terms from a natural-language query."""
    terms: list[str] = []
    for match in MIXED_TOKEN_PATTERN.finditer(query):
        token = match.group(0)
        if "\u3400" <= token[0] <= "\u9fff":
            pieces = [
                piece
                for piece in CHINESE_SPLIT_PATTERN.split(token)
                if len(piece) >= 2 and piece not in GENERIC_TERMS
            ]
            for piece in pieces:
                terms.append(piece)
                if len(piece) >= 4:
                    terms.extend(
                        piece[index : index + 2]
                        for index in range(len(piece) - 1)
                    )
        elif len(token) >= 2:
            terms.append(token.lower())
    unique: list[str] = []
    seen: set[str] = set()
    for term in terms:
        normalized = term.lower()
        if normalized not in seen:
            seen.add(normalized)
            unique.append(term)
    return unique


def matched_ocr_terms(query: str, ocr_text: str) -> list[str]:
    normalized_text = ocr_text.lower()
    return [
        term
        for term in extract_query_terms(query)
        if term.lower() in normalized_text
    ]


def evidence_excerpt(
    ocr_text: str,
    matched_terms: list[str],
    *,
    max_chars: int = 180,
) -> str:
    normalized = " ".join(ocr_text.split())
    if not normalized or not matched_terms:
        return ""
    positions = [
        normalized.lower().find(term.lower())
        for term in matched_terms
        if normalized.lower().find(term.lower()) >= 0
    ]
    center = min(positions) if positions else 0
    start = max(0, center - max_chars // 4)
    end = min(len(normalized), start + max_chars)
    if end - start < max_chars:
        start = max(0, end - max_chars)
    excerpt = normalized[start:end]
    return (
        ("…" if start else "")
        + excerpt
        + ("…" if end < len(normalized) else "")
    )


def dominant_evidence_branch(result_row: dict[str, Any]) -> str:
    if result_row.get("reranker_score") is not None:
        return "多模态精排复核"
    if result_row.get("retrieval_route") == "visual_discovery":
        if result_row.get("color_intent"):
            return (
                "颜色特征"
                if result_row.get("pure_color_query")
                else "视觉内容 + 颜色约束"
            )
        return "视觉内容"
    if result_row.get("retrieval_route") == "visual_metadata":
        return "文件名/元数据 + 视觉内容"
    scores = {
        "OCR语义": float(result_row.get("text_score", 0.0)),
        "关键词": float(result_row.get("bm25_score", 0.0)),
        "视觉内容": float(result_row.get("visual_score", 0.0)),
        "颜色特征": float(result_row.get("color_score", 0.0)),
    }
    return max(scores, key=scores.get)


def build_retrieval_explanation(
    query: str,
    ocr_text: str,
    result_row: dict[str, Any],
) -> dict[str, Any]:
    matched_terms = matched_ocr_terms(query, ocr_text)
    return {
        "dominant_branch": dominant_evidence_branch(result_row),
        "matched_terms": matched_terms[:8],
        "ocr_excerpt": evidence_excerpt(ocr_text, matched_terms),
        "text_score": float(result_row.get("text_score", 0.0)),
        "bm25_score": float(result_row.get("bm25_score", 0.0)),
        "visual_score": float(result_row.get("visual_score", 0.0)),
        "metadata_score": float(result_row.get("metadata_score", 0.0)),
        "color_score": float(result_row.get("color_score", 0.0)),
        "reranker_score": result_row.get("reranker_score"),
        "exact_topic_match_count": int(
            result_row.get("exact_topic_match_count", 0)
        ),
    }
