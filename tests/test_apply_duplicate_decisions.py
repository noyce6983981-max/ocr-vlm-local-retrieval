"""Tests for recoverable near-duplicate quarantine decisions."""

from __future__ import annotations

import pytest

from scripts.apply_duplicate_decisions import apply_duplicate_decisions


def sample_manifest() -> list[dict[str, object]]:
    return [
        {
            "item_id": "keep",
            "perceptual_group": "group_001",
        },
        {
            "item_id": "remove",
            "perceptual_group": "group_001",
        },
        {
            "item_id": "different",
            "perceptual_group": "group_002",
        },
    ]


def test_duplicate_variant_is_quarantined_recoverably() -> None:
    updated, report = apply_duplicate_decisions(
        sample_manifest(),
        [
            {
                "perceptual_group": "group_001",
                "keep_item_id": "keep",
                "remove_item_id": "remove",
                "reason": "same base image",
            }
        ],
        applied_at="2026-07-29T00:00:00+00:00",
    )
    by_id = {row["item_id"]: row for row in updated}
    assert by_id["keep"]["dedup_role"] == "representative"
    assert by_id["remove"]["search_enabled"] is False
    assert by_id["remove"]["duplicate_of"] == "keep"
    assert by_id["different"].get("search_enabled", True) is True
    assert report["active_after"] == 2
    assert report["newly_quarantined"] == 1


def test_mismatched_group_is_rejected() -> None:
    with pytest.raises(ValueError, match="group mismatch"):
        apply_duplicate_decisions(
            sample_manifest(),
            [
                {
                    "perceptual_group": "group_001",
                    "keep_item_id": "keep",
                    "remove_item_id": "different",
                    "reason": "incorrect decision",
                }
            ],
            applied_at="2026-07-29T00:00:00+00:00",
        )
