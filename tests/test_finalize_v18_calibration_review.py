from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts/finalize_v18_calibration_review.py"
    )
    spec = importlib.util.spec_from_file_location(
        "finalize_v18_calibration_review", path
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _packet(query_id: str, group_id: str) -> dict[str, object]:
    return {
        "query_id": query_id,
        "group_id": group_id,
        "split": "calibration",
        "study_fingerprint": "study",
        "pool_sha256": f"pool-{query_id}",
        "candidates": [{"item_id": "a"}, {"item_id": "b"}],
    }


def _judgment(
    query_id: str, *, reviewer_id: str, reviewer_role: str, relevant: bool = True
) -> dict[str, object]:
    return {
        "query_id": query_id,
        "reviewer_id": reviewer_id,
        "reviewer_role": reviewer_role,
        "study_fingerprint": "study",
        "pool_sha256": f"pool-{query_id}",
        "pool_relevance": (
            "relevant_candidate_in_pool"
            if relevant
            else "no_relevant_candidate_in_pool"
        ),
        "candidate_relevance": {"a": relevant, "b": False},
    }


def test_finalize_exact_consensus_preserves_primary_labels() -> None:
    module = _load_module()
    packets = [_packet("q1", "g1"), _packet("q2", "g1")]
    primary = [
        _judgment(qid, reviewer_id="r1", reviewer_role="primary")
        for qid in ("q1", "q2")
    ]
    secondary = [
        _judgment(qid, reviewer_id="r2", reviewer_role="secondary")
        for qid in ("q1", "q2")
    ]
    finalized, report = module.finalize_review(
        packets,
        primary,
        secondary,
        expected_query_count=2,
        minimum_double_review_fraction=1.0,
    )
    assert len(finalized) == 2
    assert all(row["double_reviewed"] is True for row in finalized)
    assert report["agreement"]["candidate_cohen_kappa"] == 1.0
    assert report["conflict_adjudication_required"] is False


def test_finalize_rejects_secondary_conflict() -> None:
    module = _load_module()
    packets = [_packet("q1", "g1"), _packet("q2", "g1")]
    primary = [
        _judgment(qid, reviewer_id="r1", reviewer_role="primary")
        for qid in ("q1", "q2")
    ]
    secondary = [
        _judgment(
            qid,
            reviewer_id="r2",
            reviewer_role="secondary",
            relevant=qid != "q1",
        )
        for qid in ("q1", "q2")
    ]
    with pytest.raises(ValueError, match="require adjudication"):
        module.finalize_review(
            packets,
            primary,
            secondary,
            expected_query_count=2,
            minimum_double_review_fraction=1.0,
        )


def test_finalize_uses_distinct_adjudicator_for_conflict() -> None:
    module = _load_module()
    packets = [_packet("q1", "g1"), _packet("q2", "g1")]
    primary = [
        _judgment(qid, reviewer_id="r1", reviewer_role="primary")
        for qid in ("q1", "q2")
    ]
    secondary = [
        _judgment(
            qid,
            reviewer_id="r2",
            reviewer_role="secondary",
            relevant=qid != "q1",
        )
        for qid in ("q1", "q2")
    ]
    adjudication = [
        _judgment(
            qid,
            reviewer_id="r3",
            reviewer_role="adjudicator",
            relevant=qid == "q1",
        )
        for qid in ("q1", "q2")
    ]
    finalized, report = module.finalize_review(
        packets,
        primary,
        secondary,
        adjudication,
        expected_query_count=2,
        minimum_double_review_fraction=1.0,
    )
    q1 = next(row for row in finalized if row["query_id"] == "q1")
    assert q1["adjudicated"] is True
    assert q1["reviewer_id"] == "r3"
    assert q1["finalization_method"] == "third_reviewer_adjudication"
    assert report["adjudicated_query_ids"] == ["q1"]


def test_holdout_report_records_opened_results() -> None:
    module = _load_module()
    packets = [
        {**_packet("q1", "g1"), "split": "holdout"},
        {**_packet("q2", "g1"), "split": "holdout"},
    ]
    primary = [
        _judgment(qid, reviewer_id="r1", reviewer_role="primary")
        for qid in ("q1", "q2")
    ]
    secondary = [
        _judgment(qid, reviewer_id="r2", reviewer_role="secondary")
        for qid in ("q1", "q2")
    ]
    _, report = module.finalize_review(
        packets,
        primary,
        secondary,
        expected_query_count=2,
        minimum_double_review_fraction=1.0,
        expected_split="holdout",
    )
    assert report["holdout_results_opened"] is True
    assert report["holdout_retrieval_executed"] is True
