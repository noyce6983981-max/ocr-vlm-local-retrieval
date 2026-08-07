from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from scripts.apply_quality_reviews import (
    apply_quality_reviews,
    publish_quality_reviews,
)


def test_apply_quality_reviews_reclassifies_and_disables() -> None:
    manifest = [
        {"item_id": "a", "category": "natural_no_text"},
        {"item_id": "b", "category": "degraded_document"},
        {"item_id": "c", "category": "clear_document"},
    ]
    reviews = {
        "a": {
            "decision": "reclassified",
            "revised_category": "scene_text",
            "quality_tags": "blur;tilt",
            "human_notes": "visible sign",
        },
        "b": {
            "decision": "ocr_retry",
            "revised_category": "",
            "human_notes": "",
        },
    }
    updated, report = apply_quality_reviews(
        manifest, reviews, applied_at="2026-07-29T00:00:00+00:00"
    )

    assert updated[0]["category"] == "scene_text"
    assert updated[0]["original_category"] == "natural_no_text"
    assert updated[0]["quality_tags"] == ["blur", "tilt"]
    assert updated[0]["taxonomy_review_status"] == "reviewed"
    assert updated[0]["search_enabled"] is True
    assert updated[1]["search_enabled"] is False
    assert "search_enabled" not in updated[2]
    assert report["active_pages"] == 2
    assert report["inactive_pages"] == 1


def test_apply_quality_reviews_rejects_unknown_item() -> None:
    with pytest.raises(ValueError, match="unknown item"):
        apply_quality_reviews(
            [{"item_id": "a"}],
            {"missing": {"decision": "accepted"}},
            applied_at="now",
        )


def test_publish_quality_reviews_creates_recoverable_backup(
    tmp_path,
) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    original = {"item_id": "a", "category": "clear_document"}
    manifest_path.write_text(
        json.dumps(original) + "\n", encoding="utf-8"
    )
    report = publish_quality_reviews(
        manifest_path,
        {
            "a": {
                "decision": "quarantined",
                "revised_category": "",
                "human_notes": "privacy",
            }
        },
        now=datetime(2026, 7, 29, tzinfo=timezone.utc),
    )

    updated = json.loads(manifest_path.read_text(encoding="utf-8"))
    backup = manifest_path.parent / "manifest_backups" / (
        "manifest_20260729_000000_000000.jsonl"
    )
    assert updated["search_enabled"] is False
    assert json.loads(backup.read_text(encoding="utf-8")) == original
    assert report["backup_path"] == str(backup)
