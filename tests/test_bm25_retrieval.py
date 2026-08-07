from __future__ import annotations

import pytest

from scripts.bm25_retrieval import (
    bm25_query_weight,
    build_bm25_payload,
    score_bm25,
    tokenize_mixed,
)


def test_tokenize_mixed_supports_chinese_and_latin() -> None:
    tokens = tokenize_mixed("森林 Campus-A1")
    assert {"森", "林", "森林", "campus-a1"} <= set(tokens)


def test_bm25_ranks_exact_ocr_terms_first() -> None:
    payload = build_bm25_payload(
        [
            "森林生态系统研究",
            "实验室深度学习课程",
            "forest ecology report",
        ]
    )
    scores = score_bm25("森林", payload)
    assert scores[0] > scores[1]
    assert scores[0] > scores[2]


def test_bm25_unknown_query_returns_zero_scores() -> None:
    payload = build_bm25_payload(["alpha beta", "gamma delta"])
    assert score_bm25("unseen", payload) == [0.0, 0.0]


def test_short_query_gets_smaller_bm25_hybrid_weight() -> None:
    assert bm25_query_weight("森林") == 0.05
    assert bm25_query_weight("基金账户业务申请表") == 0.3


def test_bm25_rejects_invalid_settings() -> None:
    with pytest.raises(ValueError, match="k1"):
        build_bm25_payload(["text"], k1=0)
