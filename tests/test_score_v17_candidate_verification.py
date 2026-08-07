from __future__ import annotations

import pytest

from scripts.score_v17_candidate_verification import build_verification_tasks


def packets():
    return [
        {
            "query_id": "q1",
            "query": "visible word on the engine",
            "group_id": "g1",
            "candidates": [
                {
                    "item_id": "a",
                    "review_metadata": {"image_path": "images/a.jpg"},
                },
                {
                    "item_id": "b",
                    "review_metadata": {"image_path": "images/b.jpg"},
                },
            ],
        }
    ]


def test_build_verification_tasks_joins_blinded_pool_without_labels() -> None:
    rankings = [
        {
            "query_id": "q1",
            "ranking": [
                {"item_id": "b", "score": 0.8},
                {"item_id": "a", "score": 0.7},
            ],
        }
    ]

    tasks = build_verification_tasks(packets(), rankings, top_k=1)

    assert tasks == [
        {
            "query_id": "q1",
            "query": "visible word on the engine",
            "group_id": "g1",
            "candidates": [
                {
                    "item_id": "b",
                    "image_path": "images/b.jpg",
                    "ranking_score": 0.8,
                }
            ],
        }
    ]
    assert "candidate_relevance" not in str(tasks)


def test_build_verification_tasks_rejects_candidate_outside_pool() -> None:
    rankings = [{"query_id": "q1", "ranking": [{"item_id": "unknown", "score": 0.8}]}]

    with pytest.raises(ValueError, match="outside the blinded review pool"):
        build_verification_tasks(packets(), rankings, top_k=1)


def test_build_verification_tasks_requires_requested_depth() -> None:
    rankings = [{"query_id": "q1", "ranking": [{"item_id": "a", "score": 0.8}]}]

    with pytest.raises(ValueError, match="fewer than 2"):
        build_verification_tasks(packets(), rankings, top_k=2)
