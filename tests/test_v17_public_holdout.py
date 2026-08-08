from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ocr_vlm_retrieval.evaluation.protocol_lock import file_sha256
from ocr_vlm_retrieval.evaluation.public_holdout import (
    PUBLIC_RECORD_FIELDS,
    analyze_public_paired_records,
    build_public_paired_records,
    summarize_verification_runtime,
    validate_public_paired_records,
)
from scripts.verify_v17_public_holdout import verify_bundle


def decision(
    query_id: str,
    group_id: str,
    pool_relevance: str,
    *,
    relevant_in_top3: bool,
    v16_accepted: bool,
    v16_selected_relevant: bool,
    v16_correct: bool,
    v17_accepted: bool,
    v17_selected_rank: int | None,
    v17_selected_relevant: bool,
    v17_correct: bool,
) -> dict:
    return {
        "query_id": query_id,
        "group_id": group_id,
        "pool_relevance": pool_relevance,
        "relevant_in_top_k": relevant_in_top3,
        "v16_accepted": v16_accepted,
        "v16_selected_relevant": v16_selected_relevant,
        "v16_pool_conditioned_correct": float(v16_correct),
        "accepted": v17_accepted,
        "selected_rank": v17_selected_rank,
        "selected_relevant": v17_selected_relevant,
        "v17_pool_conditioned_correct": float(v17_correct),
    }


def public_input_fixtures() -> tuple[dict, list[dict], list[dict]]:
    relevant = "relevant_candidate_in_pool"
    no_relevant = "no_relevant_candidate_in_pool"
    final_report = {
        "status": "v17_holdout_evaluated_once",
        "method_tuned_on_holdout": False,
        "query_count": 4,
        "decisions": [
            decision(
                "raw_q1",
                "source_secret_a",
                relevant,
                relevant_in_top3=True,
                v16_accepted=False,
                v16_selected_relevant=False,
                v16_correct=False,
                v17_accepted=True,
                v17_selected_rank=2,
                v17_selected_relevant=True,
                v17_correct=True,
            ),
            decision(
                "raw_q2",
                "source_secret_a",
                relevant,
                relevant_in_top3=True,
                v16_accepted=True,
                v16_selected_relevant=True,
                v16_correct=True,
                v17_accepted=True,
                v17_selected_rank=1,
                v17_selected_relevant=False,
                v17_correct=False,
            ),
            decision(
                "raw_q3",
                "source_secret_b",
                no_relevant,
                relevant_in_top3=False,
                v16_accepted=True,
                v16_selected_relevant=False,
                v16_correct=False,
                v17_accepted=False,
                v17_selected_rank=None,
                v17_selected_relevant=False,
                v17_correct=True,
            ),
            decision(
                "raw_q4",
                "source_secret_c",
                no_relevant,
                relevant_in_top3=False,
                v16_accepted=False,
                v16_selected_relevant=False,
                v16_correct=True,
                v17_accepted=False,
                v17_selected_rank=None,
                v17_selected_relevant=False,
                v17_correct=True,
            ),
        ],
    }
    rankings = []
    for query_id in ("raw_q1", "raw_q2", "raw_q3", "raw_q4"):
        second_item = "right" if query_id == "raw_q2" else "wrong_2"
        rankings.append(
            {
                "query_id": query_id,
                "ranking": [
                    {"item_id": f"{query_id}_wrong_1"},
                    {"item_id": f"{query_id}_{second_item}"},
                    {"item_id": f"{query_id}_wrong_3"},
                ],
            }
        )
    judgments = []
    for query_id in ("raw_q1", "raw_q2", "raw_q3", "raw_q4"):
        is_relevant = query_id in {"raw_q1", "raw_q2"}
        candidate_relevance = {
            f"{query_id}_wrong_1": False,
            f"{query_id}_wrong_2": False,
            f"{query_id}_wrong_3": False,
            f"{query_id}_right": is_relevant,
        }
        judgments.append(
            {
                "query_id": query_id,
                "task_id": "pooled_relevance",
                "pool_relevance": relevant if is_relevant else no_relevant,
                "candidate_relevance": candidate_relevance,
            }
        )
    return final_report, rankings, judgments


def test_public_records_remove_raw_identity_and_recompute_funnel() -> None:
    final_report, rankings, judgments = public_input_fixtures()

    rows = build_public_paired_records(
        final_report=final_report,
        v16_rankings=rankings,
        judgments=judgments,
    )
    serialized = json.dumps(rows)
    assert len(rows) == 4
    assert all(set(row) == PUBLIC_RECORD_FIELDS for row in rows)
    assert "raw_q" not in serialized
    assert "source_secret" not in serialized
    assert "item_id" not in serialized

    analysis = analyze_public_paired_records(
        rows, bootstrap_repetitions=25, seed=17
    )
    assert analysis["metrics"]["v16_retrieval_recall_at_3"] == 0.5
    assert analysis["metrics"]["v17_retrieval_recall_at_3"] == 1.0
    assert analysis["metrics"]["v17_top3_reachable_conversion_rate"] == 0.5
    assert analysis["metrics"]["v17_accepted_selected_relevant_precision"] == 0.5
    assert analysis["metrics"]["v16_pool_conditioned_end_to_end_accuracy"] == 0.5
    assert analysis["metrics"]["v17_pool_conditioned_end_to_end_accuracy"] == 0.75
    assert analysis["funnel"]["v17_reachable_but_not_selected"] == 1
    assert analysis["funnel"]["reachable_wrong_candidate_accepted"] == 1


def test_public_record_validation_rejects_extra_identity_field() -> None:
    final_report, rankings, judgments = public_input_fixtures()
    rows = build_public_paired_records(
        final_report=final_report,
        v16_rankings=rankings,
        judgments=judgments,
    )
    rows[0]["source_relpath"] = "private/source"

    with pytest.raises(ValueError, match="privacy allowlist"):
        validate_public_paired_records(rows)


def build_fixture_rows() -> list[dict]:
    final_report, rankings, judgments = public_input_fixtures()
    return build_public_paired_records(
        final_report=final_report,
        v16_rankings=rankings,
        judgments=judgments,
    )


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("status", "sealed V17"),
        ("tuned", "method_tuned"),
        ("query_sets", "identical queries"),
        ("query_count", "query_count"),
        ("group", "group_id"),
        ("short_ranking", "fewer than 3"),
        ("bad_item", "invalid item"),
        ("pool_mismatch", "Pool relevance differs"),
        ("rank_type", "selected_rank"),
        ("bool_type", "must be boolean"),
    ],
)
def test_public_record_builder_rejects_malformed_private_inputs(
    case: str, message: str
) -> None:
    final_report, rankings, judgments = public_input_fixtures()
    if case == "status":
        final_report["status"] = "draft"
    elif case == "tuned":
        final_report["method_tuned_on_holdout"] = True
    elif case == "query_sets":
        rankings.pop()
    elif case == "query_count":
        final_report["query_count"] = 5
    elif case == "group":
        final_report["decisions"][0]["group_id"] = ""
    elif case == "short_ranking":
        rankings[0]["ranking"] = rankings[0]["ranking"][:2]
    elif case == "bad_item":
        rankings[0]["ranking"][0]["item_id"] = ""
    elif case == "pool_mismatch":
        final_report["decisions"][0]["pool_relevance"] = (
            "no_relevant_candidate_in_pool"
        )
    elif case == "rank_type":
        final_report["decisions"][0]["selected_rank"] = "2"
    else:
        final_report["decisions"][0]["v16_accepted"] = 1

    with pytest.raises(ValueError, match=message):
        build_public_paired_records(
            final_report=final_report,
            v16_rankings=rankings,
            judgments=judgments,
        )


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("empty", "At least one"),
        ("alias", "remapped"),
        ("duplicate", "Duplicate"),
        ("bool", "must be boolean"),
        ("pool", "Invalid public"),
        ("rank", "integer in"),
        ("acceptance", "inconsistent"),
    ],
)
def test_public_record_validation_rejects_malformed_rows(
    case: str, message: str
) -> None:
    rows = build_fixture_rows()
    if case == "empty":
        rows = []
    elif case == "alias":
        rows[0]["query_id"] = "raw_query"
    elif case == "duplicate":
        rows[1]["query_id"] = rows[0]["query_id"]
    elif case == "bool":
        rows[0]["v16_accepted"] = 1
    elif case == "pool":
        rows[0]["pool_relevance"] = "corpus_no_answer"
    elif case == "rank":
        rows[0]["v17_selected_rank"] = 4
    else:
        rows[0]["v17_accepted"] = False

    with pytest.raises(ValueError, match=message):
        validate_public_paired_records(rows)


def test_public_analysis_requires_both_pool_strata() -> None:
    rows = [
        row
        for row in build_fixture_rows()
        if row["pool_relevance"] == "relevant_candidate_in_pool"
    ]

    with pytest.raises(ValueError, match="relevant and no-relevant"):
        analyze_public_paired_records(rows, bootstrap_repetitions=2)


def test_runtime_summary_is_descriptive_and_uses_recorded_timings() -> None:
    summary = summarize_verification_runtime(
        {
            "status": "complete",
            "judgments_read": False,
            "completed_query_count": 4,
            "load_seconds": 6.0,
            "elapsed_seconds": 8.0,
            "peak_reserved_gib": 4.0,
            "results": [
                {"elapsed_seconds": 1.0},
                {"elapsed_seconds": 2.0},
                {"elapsed_seconds": 3.0},
                {"elapsed_seconds": 4.0},
            ],
        }
    )

    assert summary["per_query_elapsed_seconds"]["mean"] == 2.5
    assert summary["per_query_elapsed_seconds"]["p50"] == 2.5
    assert summary["per_query_elapsed_seconds"]["p95"] == pytest.approx(3.85)
    assert "not production" in summary["scope"]


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("status", "complete"),
        ("judgments", "judgments_read"),
        ("rows", "no result"),
        ("elapsed", "positive"),
        ("count", "differs"),
    ],
)
def test_runtime_summary_rejects_malformed_artifacts(
    case: str, message: str
) -> None:
    artifact = {
        "status": "complete",
        "judgments_read": False,
        "completed_query_count": 1,
        "load_seconds": 1.0,
        "elapsed_seconds": 1.0,
        "peak_reserved_gib": 1.0,
        "results": [{"elapsed_seconds": 1.0}],
    }
    if case == "status":
        artifact["status"] = "partial"
    elif case == "judgments":
        artifact["judgments_read"] = True
    elif case == "rows":
        artifact["results"] = []
    elif case == "elapsed":
        artifact["results"] = [{"elapsed_seconds": 0.0}]
    else:
        artifact["completed_query_count"] = 2

    with pytest.raises(ValueError, match=message):
        summarize_verification_runtime(artifact)


def test_published_v17_audit_is_deidentified_and_reproducible() -> None:
    root = PROJECT_ROOT / "data/evaluation/v17"
    paired_path = root / "public_holdout_paired_records.jsonl"
    audit_path = root / "public_holdout_audit.json"
    summary_path = root / "final_holdout_summary.json"
    amendment_path = root / "amendments/008_post_evaluation_reporting_addendum.json"
    status = json.loads((root / "study_status.json").read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))

    result = verify_bundle(
        paired_path=paired_path,
        audit_path=audit_path,
        summary_path=summary_path,
    )

    assert result["valid"] is True
    assert result["query_count"] == 40
    assert audit["method_changed"] is False
    assert audit["model_rerun"] is False
    assert audit["holdout_rerun"] is False
    assert audit["analysis"]["funnel"] == {
        "relevant_candidate_in_pool": 32,
        "v17_relevant_in_top3": 26,
        "v17_selected_relevant": 14,
        "v17_top3_recall_miss": 6,
        "v17_reachable_but_not_selected": 12,
        "reachable_all_candidates_below_threshold": 10,
        "reachable_wrong_candidate_accepted": 2,
        "no_relevant_candidate_in_pool": 8,
        "v17_correct_rejection": 5,
        "v17_false_accept": 3,
    }
    reporting = status["holdout"]["reporting_addendum"]
    assert file_sha256(paired_path) == reporting["paired_records_sha256"]
    assert file_sha256(audit_path) == reporting["audit_sha256"]
    assert amendment["hypothesis_disposition"]["H1"]["status"] == (
        "not_tested_as_preregistered"
    )
    assert amendment["hypothesis_disposition"]["H2"]["status"] == (
        "supported_under_canonical_recall_at_3_endpoint"
    )
    assert amendment["hypothesis_disposition"]["H3"]["status"] == (
        "not_tested_on_independent_human_holdout"
    )
