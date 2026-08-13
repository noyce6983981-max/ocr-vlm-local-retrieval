from __future__ import annotations

from scripts.run_v19_retrieval_holdout_once import (
    eligible_assignments,
    hard_negative_far,
)


def test_holdout_runtime_eligibility_ignores_benchmark_metadata() -> None:
    query = "找编号为17、日期为1997年7月31日的进展报告。"
    assignments = [
        {"query_id": "a", "query": query, "content_stratum": "ocr_literal_lookup"},
        {"query_id": "b", "query": query, "content_stratum": "pure_visual"},
    ]
    assert [row["query_id"] for row in eligible_assignments(assignments)] == [
        "a",
        "b",
    ]


def test_holdout_runtime_does_not_activate_for_arbitrary_layout_words() -> None:
    assignments = [
        {
            "query_id": "layout",
            "query": "找同时包含 Phone、Fax 和 Email 字段的双语表单。",
            "content_stratum": "ocr_literal_lookup",
        }
    ]
    assert eligible_assignments(assignments) == []


def test_hard_negative_far_uses_only_single_condition_negatives() -> None:
    rows = [
        {"query_role": "single_condition_hard_negative", "accepted": True},
        {"query_role": "single_condition_hard_negative", "accepted": False},
        {"query_role": "unanswerable_neighbor", "accepted": True},
    ]
    assert hard_negative_far(rows) == 0.5
