"""Tests for the v2 content-category and quality-tag taxonomy."""

from __future__ import annotations

from scripts.migrate_taxonomy_v2 import migrate_rows
from scripts.taxonomy import (
    migrate_manifest_row,
    parse_quality_tags,
    source_tag_category_suggestion,
)


def test_clear_document_becomes_pending_general_text() -> None:
    migrated = migrate_manifest_row(
        {
            "item_id": "a",
            "category": "clear_document",
            "quality_tags": ["real_scan", "clear_scan"],
        }
    )
    assert migrated["category"] == "general_text_document"
    assert migrated["quality_tags"] == ["clear"]
    assert migrated["source_tags"] == ["real_scan"]
    assert migrated["taxonomy_review_status"] == "pending"


def test_degraded_document_keeps_quality_as_separate_axis() -> None:
    migrated = migrate_manifest_row(
        {
            "item_id": "b",
            "category": "degraded_document",
            "quality_tags": ["low_resolution", "scan_noise"],
        }
    )
    assert migrated["category"] == "general_text_document"
    assert migrated["quality_tags"] == [
        "low_resolution",
        "scan_noise",
    ]
    assert migrated["taxonomy_review_status"] == "pending"


def test_human_reclassification_is_preserved() -> None:
    migrated = migrate_manifest_row(
        {"item_id": "c", "category": "clear_document"},
        {
            "decision": "reclassified",
            "revised_category": "table_form_ticket",
        },
    )
    assert migrated["category"] == "table_form_ticket"
    assert migrated["taxonomy_review_status"] == "reviewed"


def test_migration_keeps_search_state_and_row_order() -> None:
    rows = [
        {
            "item_id": "a",
            "category": "degraded_document",
            "search_enabled": False,
        },
        {
            "item_id": "b",
            "category": "scene_text",
            "search_enabled": True,
        },
    ]
    migrated = migrate_rows(rows, {})
    assert [row["item_id"] for row in migrated] == ["a", "b"]
    assert migrated[0]["search_enabled"] is False
    assert migrated[1]["category"] == "scene_text"
    assert migrated[1]["taxonomy_review_status"] == "preserved"


def test_clear_is_removed_when_issue_tag_is_selected() -> None:
    assert parse_quality_tags(["clear", "blur"]) == ["blur"]


def test_form_provenance_has_high_precision_category_suggestion() -> None:
    assert (
        source_tag_category_suggestion({"source_tags": ["form_layout"]})
        == "table_form_ticket"
    )
    assert (
        source_tag_category_suggestion({"source_tags": ["news_article"]})
        is None
    )
    assert (
        source_tag_category_suggestion(
            {"source_tags": [], "public_source_name": "FUNSD"}
        )
        == "table_form_ticket"
    )
