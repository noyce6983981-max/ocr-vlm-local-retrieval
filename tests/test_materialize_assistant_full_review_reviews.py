"""Tests for assistant full-review model override materialization."""

from __future__ import annotations

from scripts.build_assistant_full_review_queues import EXPECTED_COUNTS
from scripts.materialize_assistant_full_review_reviews import materialize


def test_materialize_preserves_all_splits_and_categories() -> None:
    rows = []
    sequence = 0
    for split, count in EXPECTED_COUNTS.items():
        for _ in range(count):
            sequence += 1
            rows.append(
                {
                    "blind_id": f"blind_{sequence}",
                    "item_id": f"item_{sequence}",
                    "split": split,
                    "assistant_category": "general_text_document",
                    "assistant_notes": "reviewed",
                    "status": "completed",
                    "reviewed_at": "2026-07-30T00:00:00Z",
                }
            )

    output = materialize(rows)

    assert len(output) == sum(EXPECTED_COUNTS.values())
    assert {row["decision"] for row in output} == {"reclassified"}
    assert {
        row["revised_category"] for row in output
    } == {"general_text_document"}
