from __future__ import annotations

from scripts.score_v19_colqwen2_verifier_development import (
    choose_threshold,
    grouped_cross_validated_summary,
    score_cache_from_payload,
)


def _tasks_and_scores():
    tasks = []
    scores = []
    for family, positive_score, negative_score in [
        ("a", 0.9, 0.4),
        ("b", 0.8, 0.3),
    ]:
        for suffix, answerable, score in [
            ("q1", True, positive_score),
            ("q3", False, negative_score),
        ]:
            query_id = f"family_{family}_{suffix}"
            tasks.append(
                {
                    "query_id": query_id,
                    "gold_answerable": answerable,
                    "gold_relevant_item_ids": ["gold"] if answerable else [],
                    "candidates": [{"item_id": "gold", "retrieval_rank": 1}],
                }
            )
            scores.append(
                {
                    "query_id": query_id,
                    "candidate_item_ids": ["gold"],
                    "scores": [score],
                }
            )
    return tasks, scores


def test_score_cache_validates_and_indexes_query_candidate_pairs() -> None:
    cache = score_cache_from_payload(
        {
            "results": [
                {
                    "query_id": "q",
                    "candidate_item_ids": ["a", "b"],
                    "scores": [0.1, 0.2],
                }
            ]
        }
    )
    assert cache == {("q", "a"): 0.1, ("q", "b"): 0.2}


def test_threshold_selection_respects_false_accept_constraint() -> None:
    tasks, scores = _tasks_and_scores()
    threshold, summary = choose_threshold(
        tasks, scores, max_false_accept_rate=0.0
    )
    assert 0.4 < threshold <= 0.8
    assert summary["end_to_end_accuracy"] == 1.0
    assert summary["false_accept_rate"] == 0.0


def test_grouped_cross_validation_holds_out_whole_query_families() -> None:
    tasks, scores = _tasks_and_scores()
    summary = grouped_cross_validated_summary(
        tasks, scores, max_false_accept_rate=0.0
    )
    assert summary["family_count"] == 2
    assert summary["protocol"] == "leave_one_query_family_out_threshold_selection"
    assert summary["end_to_end_accuracy"] == 0.75
