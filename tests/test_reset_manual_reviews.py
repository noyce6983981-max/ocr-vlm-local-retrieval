from __future__ import annotations

from scripts.reset_manual_reviews import reset_manifest_rows


def test_reset_removes_human_state_and_recovers_form_category() -> None:
    rows, refined = reset_manifest_rows(
        [
            {
                "item_id": "form",
                "category": "general_text_document",
                "taxonomy_v1_category": "clear_document",
                "taxonomy_review_status": "reviewed",
                "source_tags": ["form_layout"],
                "quality_review_applied_at": "yesterday",
                "quality_review_notes": "old memory",
            }
        ]
    )
    assert refined == 1
    assert rows[0]["category"] == "table_form_ticket"
    assert rows[0]["taxonomy_review_status"] == "pending"
    assert "quality_review_applied_at" not in rows[0]
    assert "quality_review_notes" not in rows[0]


def test_reset_preserves_unambiguous_source_category() -> None:
    rows, refined = reset_manifest_rows(
        [
            {
                "item_id": "scene",
                "category": "scene_text",
                "taxonomy_v1_category": "scene_text",
                "taxonomy_review_status": "reviewed",
                "source_tags": ["real_scene_text"],
            }
        ]
    )
    assert refined == 0
    assert rows[0]["category"] == "scene_text"
    assert rows[0]["taxonomy_review_status"] == "preserved"
