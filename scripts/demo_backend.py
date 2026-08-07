"""Pure score processing helpers for the Streamlit experiment browser."""

from __future__ import annotations

from typing import Any

import numpy as np


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Min-max normalize every query independently."""
    minimum = matrix.min(axis=1, keepdims=True)
    maximum = matrix.max(axis=1, keepdims=True)
    scale = np.maximum(maximum - minimum, 1e-8)
    return (matrix - minimum) / scale


def align_score_matrix(
    archive: dict[str, np.ndarray] | Any,
    query_ids: list[str],
    item_ids: list[str],
) -> np.ndarray:
    """Align a saved score matrix to reviewed query and manifest order."""
    source_queries = archive["query_ids"].tolist()
    source_items = archive["item_ids"].tolist()
    query_order = [source_queries.index(query_id) for query_id in query_ids]
    item_order = [source_items.index(item_id) for item_id in item_ids]
    return archive["scores"][np.ix_(query_order, item_order)].astype(
        np.float32
    )


def build_method_scores(
    text_scores: np.ndarray,
    visual_scores: np.ndarray,
    confidences: np.ndarray,
    fixed_text_weight: float = 0.5,
    adaptive_text_weight_cap: float = 0.6,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Build text, visual, fixed, and OCR-quality-aware score matrices."""
    text = normalize_rows(text_scores)
    visual = normalize_rows(visual_scores)
    fixed = (
        fixed_text_weight * text
        + (1.0 - fixed_text_weight) * visual
    )
    text_weights = np.clip(
        adaptive_text_weight_cap * confidences, 0.0, 1.0
    )
    adaptive = (
        text * text_weights[None, :]
        + visual * (1.0 - text_weights[None, :])
    )
    return {
        "text": text,
        "visual": visual,
        "fixed": fixed,
        "adaptive": adaptive,
    }, text_weights


def reciprocal_rank_contributions(
    scores: np.ndarray,
    *,
    k: int = 60,
    top_n: int = 60,
    positive_only: bool = False,
) -> np.ndarray:
    """Convert one branch's scores into truncated reciprocal-rank values."""
    if scores.ndim != 1:
        raise ValueError("RRF branch scores must be one-dimensional.")
    if k <= 0 or top_n <= 0:
        raise ValueError("RRF k and top_n must be positive.")
    contributions = np.zeros_like(scores, dtype=np.float32)
    order = np.argsort(-scores, kind="stable")[: min(top_n, len(scores))]
    for rank, index in enumerate(order, start=1):
        if positive_only and float(scores[index]) <= 0.0:
            continue
        contributions[index] = 1.0 / (k + rank)
    return contributions


def build_rrf_scores(
    dense_scores: np.ndarray,
    bm25_scores: np.ndarray,
    visual_scores: np.ndarray,
    confidences: np.ndarray,
    *,
    k: int = 60,
    top_n: int = 60,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Build equal and OCR-quality-aware Dense/BM25/Image RRF scores."""
    if not (
        dense_scores.shape
        == bm25_scores.shape
        == visual_scores.shape
        == confidences.shape
    ):
        raise ValueError("RRF inputs must share the same shape.")
    dense = reciprocal_rank_contributions(
        dense_scores, k=k, top_n=top_n
    )
    bm25 = reciprocal_rank_contributions(
        bm25_scores,
        k=k,
        top_n=top_n,
        positive_only=True,
    )
    visual = reciprocal_rank_contributions(
        visual_scores, k=k, top_n=top_n
    )
    quality = np.clip(confidences.astype(np.float32), 0.0, 1.0)
    equal_rrf = dense + bm25 + visual
    quality_rrf = (
        0.5 * quality * dense
        + 0.5 * quality * bm25
        + (1.0 - 0.5 * quality) * visual
    )
    return (
        {"rrf": equal_rrf, "quality_rrf": quality_rrf},
        {
            "dense_rrf": dense,
            "bm25_rrf": bm25,
            "visual_rrf": visual,
        },
    )


def rank_items(scores: np.ndarray, item_ids: list[str]) -> list[str]:
    """Return item IDs in descending score order."""
    order = np.argsort(-scores)
    return [item_ids[index] for index in order]


def retrieval_metrics(
    score_matrix: np.ndarray,
    expected_item_ids: list[str],
    item_ids: list[str],
) -> dict[str, float]:
    """Compute Recall@1, Recall@3, and MRR."""
    ranks: list[int] = []
    for scores, expected_item_id in zip(score_matrix, expected_item_ids):
        order = np.argsort(-scores)
        expected_index = item_ids.index(expected_item_id)
        rank = int(np.where(order == expected_index)[0][0]) + 1
        ranks.append(rank)

    count = len(ranks)
    return {
        "recall_at_1": sum(rank == 1 for rank in ranks) / count,
        "recall_at_3": sum(rank <= 3 for rank in ranks) / count,
        "mrr": sum(1.0 / rank for rank in ranks) / count,
    }
