from __future__ import annotations

import pytest

from scripts.adjudicate_v17_calibration import materialize_adjudication


def packet() -> dict[str, object]:
    return {
        "query_id": "q1",
        "query": "red sign",
        "study_fingerprint": "study",
        "candidates": [{"item_id": "a"}, {"item_id": "b"}],
    }


def payload(**decision_overrides: object) -> dict[str, object]:
    decision = {
        "query_id": "q1",
        "answerability": "answerable",
        "relevant_item_ids": ["a"],
        "confidence": "high",
        "rationale": "visible",
    }
    decision.update(decision_overrides)
    return {
        "adjudicator_id": "model-audit",
        "decision_revision": "digest",
        "policy": "exact conjunction",
        "limitations": ["not human gold"],
        "decisions": [decision],
    }


def test_materialize_preserves_human_rows_and_expands_candidate_map() -> None:
    judgments = [
        {
            "query_id": "q1",
            "reviewer_id": "human",
            "reviewer_role": "primary",
            "answerability": "no_answer",
            "candidate_relevance": {"a": False, "b": False},
        }
    ]
    adjudicated, report = materialize_adjudication(
        [packet()], judgments, payload()
    )
    assert judgments[0]["answerability"] == "no_answer"
    assert adjudicated[0]["candidate_relevance"] == {"a": True, "b": False}
    assert adjudicated[0]["reviewer_type"] == "model_assisted"
    assert report["primary_queries_changed"] == 1


def test_materialize_rejects_unknown_candidate() -> None:
    with pytest.raises(ValueError, match="Unknown relevant candidates"):
        materialize_adjudication(
            [packet()], [], payload(relevant_item_ids=["missing"])
        )


def test_materialize_rejects_answerable_without_relevant_item() -> None:
    with pytest.raises(ValueError, match="Answerable decision has no relevant"):
        materialize_adjudication(
            [packet()], [], payload(relevant_item_ids=[])
        )
