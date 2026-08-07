from __future__ import annotations

import pytest

from scripts.calibrate_v17_candidate_gate import (
    calibrate,
    merge_judgment_extensions,
)


def verification():
    rows = []
    for query_id, score in (("q1", 0.8), ("q2", 0.7), ("q3", 0.2), ("q4", 0.6)):
        rows.append(
            {
                "query_id": query_id,
                "group_id": query_id,
                "candidates": [
                    {
                        "item_id": f"item_{query_id}",
                        "full_query_score": score,
                        "requirement_scores": {"r1": score},
                    }
                ],
            }
        )
    rows.append(
        {
            "query_id": "q5",
            "group_id": "q5",
            "candidates": [
                {
                    "item_id": "item_q5",
                    "full_query_score": 0.9,
                    "requirement_scores": {"r1": 0.9},
                }
            ],
        }
    )
    return {
        "status": "complete",
        "judgments_read": False,
        "model": "test-model",
        "policy_sha256": "policy",
        "ranking_sha256": "ranking",
        "results": rows,
    }


def judgments():
    return [
        {
            "query_id": "q1",
            "task_id": "pooled_relevance",
            "pool_relevance": "relevant_candidate_in_pool",
            "candidate_relevance": {"item_q1": True},
        },
        {
            "query_id": "q2",
            "task_id": "pooled_relevance",
            "pool_relevance": "no_relevant_candidate_in_pool",
            "candidate_relevance": {"item_q2": False},
        },
        {
            "query_id": "q3",
            "task_id": "pooled_relevance",
            "pool_relevance": "no_relevant_candidate_in_pool",
            "candidate_relevance": {"item_q3": False},
        },
        {
            "query_id": "q4",
            "task_id": "pooled_relevance",
            "pool_relevance": "relevant_candidate_in_pool",
            "candidate_relevance": {"item_q4": True},
        },
        {
            "query_id": "q5",
            "task_id": "pooled_relevance",
            "pool_relevance": "excluded",
            "candidate_relevance": {"item_q5": False},
        },
    ]


def baseline_rows():
    return [
        {
            "query_id": query_id,
            "v16_accepted": True,
            "v16_correct": float(query_id in {"q1", "q4"}),
        }
        for query_id in ("q1", "q2", "q3", "q4", "q5")
    ]


def test_calibrate_selects_non_degenerate_geometric_gate() -> None:
    report, decisions, gate = calibrate(
        verification=verification(),
        judgments=judgments(),
        baseline_rows=baseline_rows(),
        bootstrap_repetitions=25,
        seed=17,
    )

    assert gate["method"] == "attribute_geometric"
    assert gate["threshold"] == 0.21
    assert report["scope"]["excluded_query_ids"] == ["q5"]
    assert not report["selected_operating_point"]["degenerate_reject_all"]
    assert (
        report["selected_operating_point"][
            "pool_conditioned_false_accept_rate"
        ]
        == 0.5
    )
    assert len(decisions) == 4
    assert (
        report["paired_group_bootstrap"][
            "pool_conditioned_false_accept_v17_minus_v16"
        ]["repetitions"]
        == 25
    )


def test_calibrate_rejects_verifier_that_read_judgments() -> None:
    payload = verification()
    payload["judgments_read"] = True

    with pytest.raises(ValueError, match="judgments_read=false"):
        calibrate(
            verification=payload,
            judgments=judgments(),
            baseline_rows=baseline_rows(),
            bootstrap_repetitions=2,
        )


def test_calibrate_requires_complete_requested_topk_verification() -> None:
    payload = verification()

    with pytest.raises(ValueError, match="only supports K=1"):
        calibrate(
            verification=payload,
            judgments=judgments(),
            baseline_rows=baseline_rows(),
            bootstrap_repetitions=2,
            top_k_values=(1, 3),
            require_requested_top_k=True,
        )


def test_judgment_extension_adds_labels_without_mutating_base() -> None:
    base = judgments()
    merged = merge_judgment_extensions(
        base,
        [
            {
                "query_id": "q1",
                "split": "calibration",
                "candidate_relevance": {"new_item": False},
            }
        ],
    )

    assert merged[0]["candidate_relevance"]["new_item"] is False
    assert "new_item" not in base[0]["candidate_relevance"]
