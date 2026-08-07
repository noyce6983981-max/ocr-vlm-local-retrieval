from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.adjudicate_v17_holdout import adjudicate_holdout


def packet(query_id: str) -> dict:
    return {
        "query_id": query_id,
        "pool_sha256": f"pool-{query_id}",
        "study_fingerprint": "study",
        "candidates": [{"item_id": "a"}, {"item_id": "b"}],
    }


def review(
    query_id: str,
    reviewer_id: str,
    relevant: str | None,
    *,
    pool_relevance: str | None = None,
) -> dict:
    decision = pool_relevance or (
        "relevant_candidate_in_pool"
        if relevant is not None
        else "no_relevant_candidate_in_pool"
    )
    return {
        "query_id": query_id,
        "reviewer_id": reviewer_id,
        "task_id": "pooled_relevance",
        "pool_relevance": decision,
        "pool_sha256": f"pool-{query_id}",
        "study_fingerprint": "study",
        "candidate_relevance": {
            "a": relevant == "a",
            "b": relevant == "b",
        },
    }


def fixtures() -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict]]:
    packets = [packet("q1"), packet("q2"), packet("q3")]
    first = [
        review("q1", "reviewer_1", "a"),
        review("q2", "reviewer_1", "a"),
        review("q3", "reviewer_1", None, pool_relevance="uncertain"),
    ]
    second = [
        review("q1", "reviewer_2", "a"),
        review("q2", "reviewer_2", "b"),
        review("q3", "reviewer_2", None, pool_relevance="uncertain"),
    ]
    queue = [packet("q2"), packet("q3")]
    third = [
        review("q2", "reviewer_3", "b"),
        review("q3", "reviewer_3", None),
    ]
    return packets, first, second, queue, third


def test_materializes_consensus_and_blind_third_review() -> None:
    rows, report = adjudicate_holdout(*fixtures())

    assert len(rows) == 3
    assert rows[0]["adjudication_method"] == "independent_primary_consensus"
    assert rows[1]["adjudication_method"] == "independent_third_reviewer_blind"
    assert rows[1]["candidate_relevance"] == {"a": False, "b": True}
    assert rows[2]["pool_relevance"] == "no_relevant_candidate_in_pool"
    assert all(row["adjudicated"] is True for row in rows)
    assert all(row["independent_reviewer_count"] == 2 for row in rows)
    assert report["adjudication_query_count"] == 2
    assert report["model_assisted_labels_used"] is False


def test_rejects_incomplete_third_review() -> None:
    packets, first, second, queue, third = fixtures()
    third.pop()

    with pytest.raises(ValueError, match="exactly the adjudication queue"):
        adjudicate_holdout(packets, first, second, queue, third)


def test_rejects_non_independent_adjudicator() -> None:
    packets, first, second, queue, third = fixtures()
    for row in third:
        row["reviewer_id"] = "reviewer_1"

    with pytest.raises(ValueError, match="independent"):
        adjudicate_holdout(packets, first, second, queue, third)


def test_rejects_queue_that_omits_a_conflict() -> None:
    packets, first, second, queue, third = fixtures()
    queue.pop(0)
    third.pop(0)

    with pytest.raises(ValueError, match="conflict state"):
        adjudicate_holdout(packets, first, second, queue, third)


def test_rejects_candidate_coverage_change() -> None:
    packets, first, second, queue, third = fixtures()
    changed = copy.deepcopy(third)
    changed[0]["candidate_relevance"].pop("a")

    with pytest.raises(ValueError, match="coverage mismatch"):
        adjudicate_holdout(packets, first, second, queue, changed)
