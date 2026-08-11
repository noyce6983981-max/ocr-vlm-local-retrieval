from __future__ import annotations

from pathlib import Path

import pytest

from scripts.v19_intent_review_app import (
    load_queries,
    load_reviews,
    next_unreviewed_index,
    review_path_for,
    save_review,
)


def test_review_app_loads_exact_balanced_pilot_queries() -> None:
    queries = load_queries()
    assert len(queries) == 72
    assert queries[0]["query_id"] == "v19p_text_001"
    assert queries[0]["query_text"] == "这份方案里核定的项目经费合计是多少元？"


def test_review_save_is_persistent_and_bound_to_query_text(
    tmp_path: Path,
) -> None:
    query = load_queries()[0]
    path = tmp_path / "reviews.csv"
    reviews = save_review(
        path,
        query=query,
        final_route="text_evidence",
        reviewer_id="reviewer_01",
        notes="确认",
    )
    assert reviews[query["query_id"]]["accepted_proposal"] == "true"
    reloaded = load_reviews(path)
    assert reloaded[query["query_id"]]["query_text"] == query["query_text"]
    assert reloaded[query["query_id"]]["notes"] == "确认"

    changed = dict(query)
    changed["query_text"] = "不匹配的查询"
    with pytest.raises(ValueError, match="does not match"):
        save_review(
            path,
            query=changed,
            final_route="mixed",
            reviewer_id="reviewer_01",
            notes="",
        )


def test_review_navigation_prefers_unreviewed_query(tmp_path: Path) -> None:
    queries = load_queries()[:3]
    reviews = {queries[1]["query_id"]: {"query_id": queries[1]["query_id"]}}
    assert next_unreviewed_index(queries, reviews, 0) == 2


def test_reviewer_id_is_path_safe() -> None:
    assert review_path_for("reviewer_01").name == "reviewer_01_pilot_reviews.csv"
    with pytest.raises(ValueError, match="审核者ID"):
        review_path_for("../escape")
