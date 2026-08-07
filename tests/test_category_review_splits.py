"""Tests for frozen category-review split construction."""

from __future__ import annotations

from scripts.build_category_review_splits import (
    build_split_rows,
    parse_bool,
    review_required_ids,
    validate_split_rows,
)


def sample_manifest() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    categories = (
        "general_text_document",
        "complex_academic",
        "table_form_ticket",
    )
    for index in range(30):
        rows.append(
            {
                "item_id": f"item_{index:02d}",
                "category": categories[index % len(categories)],
                "taxonomy_review_status": (
                    "pending" if index < 24 else "preserved"
                ),
                "source_group_id": f"source_{index:02d}",
                "perceptual_group": (
                    "near_duplicate_pair"
                    if index in {0, 1}
                    else ""
                ),
                "hard_negative_group": (
                    "hard_pair" if index in {2, 3} else ""
                ),
            }
        )
    return rows


def test_pending_and_quality_routed_pages_require_review() -> None:
    manifest = sample_manifest()
    quality_rows = [
        {"item_id": "item_25", "quality_route": "ocr_retry"},
        {"item_id": "item_26", "quality_route": "pass"},
    ]
    required = review_required_ids(manifest, quality_rows)
    assert "item_00" in required
    assert "item_25" in required
    assert "item_26" not in required


def test_split_is_deterministic_and_group_isolated() -> None:
    manifest = sample_manifest()
    first = build_split_rows(manifest, [], seed=9)
    second = build_split_rows(manifest, [], seed=9)
    assert first == second
    summary = validate_split_rows(first)
    assert summary["review_required_pages"] == 24
    assert summary["group_leakage_count"] == 0

    by_id = {row["item_id"]: row for row in first}
    assert by_id["item_00"]["split"] == by_id["item_01"]["split"]
    assert by_id["item_02"]["split"] == by_id["item_03"]["split"]
    assert all(
        row["split"] == "train"
        for row in first
        if not parse_bool(row["review_required"])
    )


def test_split_keeps_all_three_review_partitions() -> None:
    rows = build_split_rows(sample_manifest(), [], seed=42)
    required_splits = {
        row["split"]
        for row in rows
        if parse_bool(row["review_required"])
    }
    assert required_splits == {"train", "validation", "test"}
