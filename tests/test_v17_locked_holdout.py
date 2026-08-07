from __future__ import annotations

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
from scripts.evaluate_v17_locked_holdout import evaluate_locked_holdout
from scripts.run_v17_holdout_once import validate_authorization


def candidate(item_id: str, score: float) -> dict:
    return {
        "item_id": item_id,
        "full_query_score": score,
        "requirement_scores": {"object": score},
    }


def fixtures() -> tuple[dict, dict, list[dict], list[dict]]:
    lock = {
        "status": "method_locked_holdout_sealed",
        "parser_policy_sha256": "policy",
        "candidate_ranking_policy_sha256": "ranking-policy",
        "holdout_inputs": {
            "v17_candidate_ranking_sha256": "ranking",
            "v16_baseline_ranking_sha256": "baseline-ranking",
        },
        "inference": {"top_k": 3},
        "aggregation": {
            "method": "full_query",
            "threshold": 0.51,
            "selection_policy": "highest_retrieval_rank_among_passed",
            "contrastive_relations": False,
            "relation_margin_threshold": 0.0,
        },
        "holdout": {"query_count": 2},
    }
    verification = {
        "status": "complete",
        "scope": "v17_holdout_ranked_candidates_only",
        "judgments_read": False,
        "top_k": 3,
        "policy_sha256": "policy",
        "ranking_policy_sha256": "ranking-policy",
        "ranking_sha256": "ranking",
        "results": [
            {
                "query_id": "q1",
                "group_id": "g1",
                "candidates": [
                    candidate("q1_wrong", 0.3),
                    candidate("q1_right", 0.8),
                    candidate("q1_tail", 0.1),
                ],
            },
            {
                "query_id": "q2",
                "group_id": "g2",
                "candidates": [
                    candidate("q2_a", 0.2),
                    candidate("q2_b", 0.1),
                    candidate("q2_c", 0.05),
                ],
            },
        ],
    }
    judgments = [
        {
            "query_id": "q1",
            "task_id": "pooled_relevance",
            "pool_relevance": "relevant_candidate_in_pool",
            "candidate_relevance": {
                "q1_wrong": False,
                "q1_right": True,
                "q1_tail": False,
            },
            "independent_reviewer_count": 2,
            "adjudicated": True,
        },
        {
            "query_id": "q2",
            "task_id": "pooled_relevance",
            "pool_relevance": "no_relevant_candidate_in_pool",
            "candidate_relevance": {
                "q2_a": False,
                "q2_b": False,
                "q2_c": False,
            },
            "independent_reviewer_count": 2,
            "adjudicated": True,
        },
    ]
    baseline = [
        {
            "query_id": "q1",
            "group_id": "g1",
            "scope": "v17_holdout_v16_locked_baseline",
            "judgments_read": False,
            "method": "quality_hybrid",
            "ranking_sha256": "baseline-ranking",
            "v16_accepted": False,
            "selected_item_id": None,
            "selected_rank": None,
        },
        {
            "query_id": "q2",
            "group_id": "g2",
            "scope": "v17_holdout_v16_locked_baseline",
            "judgments_read": False,
            "method": "quality_hybrid",
            "ranking_sha256": "baseline-ranking",
            "v16_accepted": True,
            "selected_item_id": "q2_a",
            "selected_rank": 1,
        },
    ]
    return lock, verification, judgments, baseline


def test_locked_holdout_applies_fixed_top3_rank_first_gate() -> None:
    lock, verification, judgments, baseline = fixtures()

    report = evaluate_locked_holdout(
        method_lock=lock,
        verification=verification,
        judgments=judgments,
        baseline_rows=baseline,
        bootstrap_repetitions=20,
        seed=17,
    )

    assert report["status"] == "v17_holdout_evaluated_once"
    assert report["method_tuned_on_holdout"] is False
    assert report["metrics"]["pool_conditioned_end_to_end_accuracy"] == 1.0
    assert report["decisions"][0]["selected_rank"] == 2
    assert report["decisions"][1]["accepted"] is False


@pytest.mark.parametrize(
    ("location", "key", "value", "message"),
    [
        (
            "verification",
            "scope",
            "v17_calibration_ranked_candidates_only",
            "holdout-only",
        ),
        ("verification", "judgments_read", True, "judgments_read=false"),
        ("verification", "policy_sha256", "changed", "policy hash"),
        ("lock", "status", "candidate", "final locked state"),
    ],
)
def test_locked_holdout_rejects_protocol_mismatch(
    location: str,
    key: str,
    value: object,
    message: str,
) -> None:
    lock, verification, judgments, baseline = fixtures()
    target = lock if location == "lock" else verification
    target[key] = value

    with pytest.raises(ValueError, match=message):
        evaluate_locked_holdout(
            method_lock=lock,
            verification=verification,
            judgments=judgments,
            baseline_rows=baseline,
            bootstrap_repetitions=2,
            seed=17,
        )


def test_locked_holdout_requires_complete_independent_adjudication() -> None:
    lock, verification, judgments, baseline = fixtures()
    judgments[0]["independent_reviewer_count"] = 1

    with pytest.raises(ValueError, match="two independent reviewers"):
        evaluate_locked_holdout(
            method_lock=lock,
            verification=verification,
            judgments=judgments,
            baseline_rows=baseline,
            bootstrap_repetitions=2,
            seed=17,
        )


def test_authorization_binds_independent_review_and_exact_inputs(
    tmp_path: Path,
) -> None:
    input_paths = {}
    for key in ("verification", "judgments", "baseline_records"):
        path = tmp_path / f"{key}.json"
        path.write_text(key, encoding="utf-8")
        input_paths[key] = path
    authorization = {
        "schema_version": 1,
        "status": "approved_for_one_shot_holdout_evaluation",
        "method_lock_sha256": "lock-hash",
        "review_protocol": {
            "independent_reviewer_count": 2,
            "independent_reviews": True,
            "full_candidate_coverage": True,
            "blinded_to_method_outputs": True,
            "model_assisted_labels_used": False,
            "conflict_adjudication_complete": True,
            "adjudicator_independent_of_primary_reviewers": True,
        },
        "input_hashes": {
            key: file_sha256(path) for key, path in input_paths.items()
        },
    }

    assert not validate_authorization(
        authorization,
        method_lock_sha256="lock-hash",
        input_paths=input_paths,
    )
    authorization["review_protocol"]["model_assisted_labels_used"] = True
    errors = validate_authorization(
        authorization,
        method_lock_sha256="lock-hash",
        input_paths=input_paths,
    )
    assert any("model_assisted_labels_used" in error for error in errors)
