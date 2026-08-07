"""Query routing and open-set rejection for multimodal retrieval."""

from __future__ import annotations

from typing import Any


VISUAL_MIN_COSINE = 0.45
TEXT_MIN_COSINE = 0.50
BM25_MIN_RAW_SCORE = 4.0

VISUAL_DESCRIPTION_TERMS = {
    "人物",
    "动物",
    "天空",
    "白云",
    "蓝天",
    "山峰",
    "雪山",
    "湖泊",
    "海边",
    "海面",
    "沙滩",
    "森林",
    "树木",
    "日出",
    "日落",
    "夕阳",
    "倒影",
    "建筑",
    "车辆",
    "帆船",
    "风景",
    "颜色",
    "画面",
}

TEXT_EVIDENCE_TERMS = {
    "写着",
    "包含",
    "标题",
    "文字",
    "关键词",
    "编号",
    "表格",
    "成绩",
    "课程",
    "论文",
    "文档",
    "代码",
    "公式",
    "多少",
}


def infer_query_mode(query: str) -> str:
    """Classify only confident visual-description queries; keep others mixed."""
    normalized = " ".join(query.split())
    if any(term in normalized for term in TEXT_EVIDENCE_TERMS):
        return "mixed"
    visual_hits = sum(
        term in normalized for term in VISUAL_DESCRIPTION_TERMS
    )
    asks_for_image = any(
        term in normalized
        for term in ("哪张图片", "哪幅图片", "哪张照片", "哪幅照片")
    )
    if visual_hits >= 2 or (asks_for_image and visual_hits >= 1):
        return "visual"
    return "mixed"


def _top_two(
    ranking: list[dict[str, Any]],
    field: str,
) -> tuple[float, float]:
    values = sorted(
        (float(row.get(field, -1.0)) for row in ranking),
        reverse=True,
    )
    if not values:
        return -1.0, -1.0
    return values[0], values[1] if len(values) > 1 else -1.0


def evaluate_ranking_acceptance(
    query: str,
    method: str,
    ranking: list[dict[str, Any]],
) -> dict[str, Any]:
    """Reject a ranking when no candidate has sufficient raw evidence."""
    query_mode = infer_query_mode(query)
    if not ranking:
        return {
            "accepted": False,
            "query_mode": query_mode,
            "reason": "资料库为空，没有可比较的数据。",
            "signal_name": "none",
            "signal": 0.0,
            "threshold": 0.0,
        }

    visual_methods = {"visual"}
    if query_mode == "visual":
        visual_methods.update(
            {
                "quality_hybrid",
                "reranker",
                "adaptive",
                "fixed",
                "quality_rrf",
                "rrf",
            }
        )

    if method in visual_methods:
        top, second = _top_two(ranking, "raw_visual_score")
        accepted = top >= VISUAL_MIN_COSINE
        comparison = "达到" if accepted else "低于"
        return {
            "accepted": accepted,
            "query_mode": query_mode,
            "reason": (
                f"最高原始视觉相似度 {top:.3f}，{comparison}校准阈值 "
                f"{VISUAL_MIN_COSINE:.3f}。"
            ),
            "signal_name": "raw_visual_cosine",
            "signal": round(top, 6),
            "threshold": VISUAL_MIN_COSINE,
            "top2_signal": round(second, 6),
            "margin": round(top - second, 6),
        }

    if method == "text":
        top, second = _top_two(ranking, "raw_text_score")
        accepted = top >= TEXT_MIN_COSINE
        comparison = "达到" if accepted else "低于"
        return {
            "accepted": accepted,
            "query_mode": query_mode,
            "reason": (
                f"最高原始文本语义相似度 {top:.3f}，{comparison}校准阈值 "
                f"{TEXT_MIN_COSINE:.3f}。"
            ),
            "signal_name": "raw_text_cosine",
            "signal": round(top, 6),
            "threshold": TEXT_MIN_COSINE,
            "top2_signal": round(second, 6),
            "margin": round(top - second, 6),
        }

    if method == "bm25":
        top, second = _top_two(ranking, "bm25_raw_score")
        accepted = top >= BM25_MIN_RAW_SCORE
        comparison = "达到" if accepted else "低于"
        return {
            "accepted": accepted,
            "query_mode": query_mode,
            "reason": (
                f"最高关键词原始分 {top:.3f}，{comparison}校准阈值 "
                f"{BM25_MIN_RAW_SCORE:.3f}。"
            ),
            "signal_name": "bm25_raw",
            "signal": round(top, 6),
            "threshold": BM25_MIN_RAW_SCORE,
            "top2_signal": round(second, 6),
            "margin": round(top - second, 6),
        }

    top_row = ranking[0]
    raw_text = float(top_row.get("raw_text_score", -1.0))
    raw_visual = float(top_row.get("raw_visual_score", -1.0))
    raw_bm25 = float(top_row.get("bm25_raw_score", 0.0))
    accepted = (
        raw_text >= TEXT_MIN_COSINE
        or raw_visual >= VISUAL_MIN_COSINE
        or raw_bm25 >= BM25_MIN_RAW_SCORE
    )
    return {
        "accepted": accepted,
        "query_mode": query_mode,
        "reason": (
            "首位候选原始证据："
            f"文本 {raw_text:.3f}/{TEXT_MIN_COSINE:.3f}，"
            f"视觉 {raw_visual:.3f}/{VISUAL_MIN_COSINE:.3f}，"
            f"关键词 {raw_bm25:.3f}/{BM25_MIN_RAW_SCORE:.3f}。"
        ),
        "signal_name": "mixed_raw_evidence",
        "signal": round(max(raw_text, raw_visual), 6),
        "threshold": None,
        "raw_text_score": round(raw_text, 6),
        "raw_visual_score": round(raw_visual, 6),
        "bm25_raw_score": round(raw_bm25, 6),
    }


def build_acceptance_decisions(
    query: str,
    rankings: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    return {
        method: evaluate_ranking_acceptance(query, method, ranking)
        for method, ranking in rankings.items()
    }
