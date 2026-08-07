from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.score_v17_locked_holdout_verification import (
    build_holdout_verification_tasks,
    validate_retrieval_lineage,
)


def test_builds_topk_tasks_from_blinded_review_assets() -> None:
    packets = [
        {
            "query_id": "q1",
            "query": "yellow dolphin over pyramid",
            "group_id": "g1",
            "candidates": [
                {
                    "item_id": "a",
                    "review_asset": {"image_path": "images/a.jpg"},
                },
                {
                    "item_id": "b",
                    "review_asset": {"image_path": "images/b.jpg"},
                },
            ],
        }
    ]
    rankings = [
        {
            "query_id": "q1",
            "ranking": [
                {"item_id": "b", "score": 0.9},
                {"item_id": "a", "score": 0.8},
            ],
        }
    ]

    tasks = build_holdout_verification_tasks(packets, rankings, top_k=2)

    assert [row["item_id"] for row in tasks[0]["candidates"]] == ["b", "a"]
    assert tasks[0]["candidates"][0]["retrieval_rank"] == 1
    assert tasks[0]["candidates"][0]["image_path"] == "images/b.jpg"


def test_rejects_ranked_candidate_outside_blinded_pool() -> None:
    packets = [
        {
            "query_id": "q1",
            "query": "query",
            "group_id": "g1",
            "candidates": [
                {
                    "item_id": "a",
                    "review_asset": {"image_path": "images/a.jpg"},
                }
            ],
        }
    ]
    rankings = [{"query_id": "q1", "ranking": [{"item_id": "b"}]}]

    with pytest.raises(ValueError, match="outside"):
        build_holdout_verification_tasks(packets, rankings, top_k=1)


def test_validates_pre_amendment_retrieval_lock_lineage() -> None:
    receipt = {
        "executed_split": "holdout",
        "judgments_read": False,
        "method_lock_sha256": "prior",
        "run_files": {"v17_quality_hybrid": {"sha256": "ranking"}},
    }

    validate_retrieval_lineage(
        receipt,
        current_lock_sha256="current",
        prior_lock_sha256="prior",
        receipt_sha256="receipt",
        expected_receipt_sha256="receipt",
        ranking_sha256="ranking",
        expected_ranking_sha256="ranking",
    )


def test_rejects_changed_holdout_ranking() -> None:
    receipt = {
        "executed_split": "holdout",
        "judgments_read": False,
        "method_lock_sha256": "prior",
        "run_files": {"v17_quality_hybrid": {"sha256": "ranking"}},
    }

    with pytest.raises(ValueError, match="method lock"):
        validate_retrieval_lineage(
            receipt,
            current_lock_sha256="current",
            prior_lock_sha256="prior",
            receipt_sha256="receipt",
            expected_receipt_sha256="receipt",
            ranking_sha256="ranking",
            expected_ranking_sha256="changed",
        )
