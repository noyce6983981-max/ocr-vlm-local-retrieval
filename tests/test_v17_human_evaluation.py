from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from ocr_vlm_retrieval.evaluation.human_evaluation import (
    agreement_report,
    blind_candidate_pool,
    grouped_paired_bootstrap,
    pool_ranked_runs,
    validate_annotation_coverage,
)
from scripts.build_v17_human_pool import build_pools


def test_pool_is_top20_union_across_independently_versioned_runs() -> None:
    pooled = pool_ranked_runs(
        {
            "v16_global": [
                {"item_id": "shared", "score": 0.9, "title": "Shared"},
                {"item_id": "v16_only", "score": 0.8},
            ],
            "v17_attributes": [
                {"item_id": "v17_only", "score": 0.95},
                {"item_id": "shared", "score": 0.7},
            ],
        },
        top_per_run=20,
    )
    assert {row["item_id"] for row in pooled} == {
        "shared",
        "v16_only",
        "v17_only",
    }
    shared = next(row for row in pooled if row["item_id"] == "shared")
    assert shared["run_ranks"] == {"v16_global": 1, "v17_attributes": 2}
    assert shared["review_metadata"] == {"title": "Shared"}


def test_blind_pool_removes_scores_methods_and_ranks() -> None:
    pooled = pool_ranked_runs(
        {
            "v16": [{"item_id": "a", "score": 0.9}],
            "v17": [{"item_id": "b", "score": 0.8}],
        }
    )
    first = blind_candidate_pool("q1", pooled, study_fingerprint="digest")
    second = blind_candidate_pool("q1", pooled, study_fingerprint="digest")
    assert first == second
    assert {row["item_id"] for row in first} == {"a", "b"}
    serialized = repr(first)
    assert "v16" not in serialized
    assert "v17" not in serialized
    assert "score" not in serialized
    assert "rank" not in serialized


def _judgment(
    query_id: str,
    reviewer_id: str,
    *,
    answerable: bool,
    relevance: dict[str, bool] | None = None,
) -> dict[str, object]:
    return {
        "query_id": query_id,
        "reviewer_id": reviewer_id,
        "answerability": "answerable" if answerable else "no_answer",
        "candidate_relevance": relevance or {"a": False, "b": False},
    }


def test_annotation_coverage_requires_all_holdout_queries_double_reviewed() -> None:
    queries = [
        {"query_id": "cal", "split": "calibration"},
        {"query_id": "hold", "split": "holdout"},
    ]
    judgments = [
        _judgment("cal", "r1", answerable=False),
        _judgment("hold", "r1", answerable=False),
    ]
    coverage = validate_annotation_coverage(queries, judgments)
    assert not coverage["valid"]
    assert any("holdout needs two" in value for value in coverage["failures"])


def test_annotation_coverage_enforces_calibration_overlap() -> None:
    queries = [{"query_id": f"q{index}", "split": "calibration"} for index in range(10)]
    judgments = [_judgment(f"q{index}", "r1", answerable=False) for index in range(10)]
    judgments.extend(
        _judgment(f"q{index}", "r2", answerable=False) for index in range(2)
    )
    coverage = validate_annotation_coverage(queries, judgments)
    assert not coverage["valid"]
    assert coverage["calibration_double_review_fraction"] == 0.2


def test_agreement_reports_raw_kappa_and_conflict_rate() -> None:
    judgments = [
        _judgment("q1", "r1", answerable=True, relevance={"a": True}),
        _judgment("q1", "r2", answerable=True, relevance={"a": True}),
        _judgment("q2", "r1", answerable=False, relevance={"a": False}),
        _judgment("q2", "r2", answerable=True, relevance={"a": True}),
    ]
    report = agreement_report(judgments)
    assert report["double_reviewed_query_count"] == 2
    assert report["answerability_raw_agreement"] == 0.5
    assert report["candidate_raw_agreement"] == 0.5
    assert report["conflict_query_ids"] == ["q2"]
    assert report["adjudication_rate"] == 0.5


def test_grouped_paired_bootstrap_is_deterministic_and_group_aware() -> None:
    rows = [
        {"group_id": "family_a", "v16": 1.0, "v17": 0.0},
        {"group_id": "family_a", "v16": 1.0, "v17": 0.0},
        {"group_id": "family_b", "v16": 0.0, "v17": 0.0},
    ]
    first = grouped_paired_bootstrap(
        rows,
        baseline_field="v16",
        contender_field="v17",
        repetitions=500,
        seed=7,
    )
    second = grouped_paired_bootstrap(
        rows,
        baseline_field="v16",
        contender_field="v17",
        repetitions=500,
        seed=7,
    )
    assert first == second
    assert first["group_count"] == 2
    assert first["paired_difference"] == pytest.approx(-2 / 3)
    assert first["confidence_interval"][1] <= 0.0


def test_build_pools_checks_every_run_covers_the_frozen_queries() -> None:
    queries = [
        {
            "query_id": "q1",
            "query": "yellow dolphin",
            "split": "holdout",
            "group_id": "family1",
        }
    ]
    with pytest.raises(ValueError, match="query mismatch"):
        build_pools(
            queries,
            {"v16": {}},
            top_per_run=20,
        )


def test_capped_pool_changes_fingerprint_and_keeps_uncapped_count() -> None:
    queries = [
        {
            "query_id": "q1",
            "query": "yellow dolphin",
            "split": "calibration",
            "group_id": "family1",
        }
    ]
    runs = {
        "v16": {
            "q1": [
                {"item_id": "shared", "score": 0.9},
                {"item_id": "v16_only", "score": 0.8},
            ]
        },
        "v17": {
            "q1": [
                {"item_id": "shared", "score": 0.95},
                {"item_id": "v17_only", "score": 0.85},
            ]
        },
    }
    full_audit, _, full_fingerprint = build_pools(queries, runs, top_per_run=20)
    capped_audit, capped_review, capped_fingerprint = build_pools(
        queries, runs, top_per_run=20, pool_size=2
    )
    assert full_audit[0]["candidate_count"] == 3
    assert capped_audit[0]["candidate_count"] == 2
    assert capped_audit[0]["uncapped_candidate_count"] == 3
    assert capped_review[0]["candidate_count"] == 2
    assert capped_fingerprint != full_fingerprint


def test_capped_pool_guarantees_each_runs_top_depth() -> None:
    queries = [
        {
            "query_id": "q1",
            "query": "yellow dolphin",
            "split": "calibration",
            "group_id": "family1",
        }
    ]
    runs = {
        "v16": {"q1": [{"item_id": "a1"}, {"item_id": "a2"}]},
        "v17": {"q1": [{"item_id": "b1"}, {"item_id": "b2"}]},
    }
    audit, _, _ = build_pools(
        queries,
        runs,
        top_per_run=20,
        pool_size=2,
        guaranteed_depth=1,
    )
    assert {row["item_id"] for row in audit[0]["candidates"]} == {"a1", "b1"}


def test_duplicate_reviewer_judgment_is_rejected() -> None:
    duplicate = [
        _judgment("q1", "r1", answerable=False),
        _judgment("q1", "r1", answerable=False),
    ]
    with pytest.raises(ValueError, match="Duplicate judgment"):
        agreement_report(duplicate)
