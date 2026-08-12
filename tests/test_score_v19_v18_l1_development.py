from __future__ import annotations

from scripts.score_v19_v18_l1_development import l1_result, summarize


def test_l1_result_selects_highest_verifier_score_not_retrieval_rank() -> None:
    task = {
        "query_id": "q",
        "query_role": "answerable_positive",
        "content_stratum": "visual_compositional",
        "gold_answerable": True,
        "gold_relevant_item_ids": ["second"],
        "candidates": [
            {"item_id": "first", "retrieval_rank": 1},
            {"item_id": "second", "retrieval_rank": 2},
        ],
    }
    result = l1_result(task, [0.6, 0.7], threshold=0.63)
    assert result["accepted"] is True
    assert result["selected_item_id"] == "second"
    assert result["selected_retrieval_rank"] == 2


def test_l1_summary_uses_all_queries_for_e2e_accuracy() -> None:
    results = [
        {
            "gold_answerable": True,
            "gold_relevant_item_ids": ["a"],
            "accepted": True,
            "selected_item_id": "a",
            "candidate_item_ids": ["a"],
        },
        {
            "gold_answerable": False,
            "gold_relevant_item_ids": [],
            "accepted": False,
            "selected_item_id": None,
            "candidate_item_ids": [],
        },
    ]
    summary = summarize(results)
    assert summary["end_to_end_accuracy"] == 1.0
    assert summary["empty_candidate_count"] == 1


def test_l1_partial_summary_handles_one_class_during_atomic_progress() -> None:
    summary = summarize(
        [
            {
                "gold_answerable": True,
                "gold_relevant_item_ids": ["a"],
                "accepted": False,
                "selected_item_id": None,
                "candidate_item_ids": ["a"],
            }
        ]
    )
    assert summary["negative_count"] == 0
    assert summary["negative_correct_reject_rate"] == 0.0
    assert summary["false_accept_rate"] == 0.0
