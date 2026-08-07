from __future__ import annotations

import json

from scripts.build_blind_model_review_bundle import safe_review_row


def test_safe_review_row_excludes_ranker_and_author_signals() -> None:
    row = safe_review_row(
        {
            "query_id": "q1",
            "query": "查找年度报告",
            "expected_answerability": "no_answer_probe",
        },
        {
            "candidate_item_ids": ["item_a"],
            "system_accepted": True,
            "acceptance_reason": "hidden",
            "candidate_results": [
                {"item_id": "item_a", "ranks": {"text": 1}, "scores": {}}
            ],
        },
        {
            "item_a": {
                "item_id": "item_a",
                "source_path": "images/a.jpg",
                "category": "hidden",
            }
        },
        lambda item_id: "ANNUAL REPORT 2025",
    )
    assert row == {
        "query_id": "q1",
        "query": "查找年度报告",
        "candidates": [
            {
                "item_id": "item_a",
                "source_path": "images/a.jpg",
                "ocr_text": "ANNUAL REPORT 2025",
            }
        ],
    }
    serialized = json.dumps(row)
    for forbidden in (
        "expected_answerability",
        "system_accepted",
        "acceptance_reason",
        "ranks",
        "scores",
        "pool_sources",
        "category",
    ):
        assert forbidden not in serialized
