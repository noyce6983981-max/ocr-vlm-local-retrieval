from __future__ import annotations

from datetime import datetime, timezone

import pytest

from scripts.quality_review import (
    read_quality_reviews,
    save_quality_review,
    validate_quality_review,
)


def test_validate_reclassification_requires_known_category() -> None:
    with pytest.raises(ValueError, match="valid revised category"):
        validate_quality_review(
            item_id="item_1",
            decision="reclassified",
            revised_category="",
            known_item_ids={"item_1"},
        )


def test_validate_rejects_unknown_item() -> None:
    with pytest.raises(ValueError, match="Unknown item"):
        validate_quality_review(
            item_id="item_2",
            decision="accepted",
            revised_category="",
            known_item_ids={"item_1"},
        )


def test_save_quality_review_upserts_atomically(tmp_path) -> None:
    path = tmp_path / "reviews.csv"
    save_quality_review(
        path,
        {
            "item_id": "item_1",
            "decision": "accepted",
            "revised_category": "",
            "quality_tags": ["clear"],
            "human_notes": "first",
        },
        now=datetime(2026, 7, 29, tzinfo=timezone.utc),
    )
    save_quality_review(
        path,
        {
            "item_id": "item_1",
            "decision": "ocr_retry",
            "revised_category": "",
            "quality_tags": ["blur", "low_resolution"],
            "human_notes": "updated",
        },
        now=datetime(2026, 7, 30, tzinfo=timezone.utc),
    )

    rows = read_quality_reviews(path)
    assert list(rows) == ["item_1"]
    assert rows["item_1"]["decision"] == "ocr_retry"
    assert rows["item_1"]["quality_tags"] == "blur;low_resolution"
    assert rows["item_1"]["human_notes"] == "updated"
