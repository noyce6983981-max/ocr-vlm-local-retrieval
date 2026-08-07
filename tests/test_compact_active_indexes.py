"""Tests for compacting indexes to active item IDs."""

from __future__ import annotations

from scripts.compact_active_library_indexes import (
    filter_metadata_indices,
)


def test_filter_metadata_indices_preserves_order() -> None:
    metadata = [
        {"item_id": "first"},
        {"item_id": "disabled"},
        {"item_id": "second"},
    ]
    rows, indices = filter_metadata_indices(
        metadata,
        {"first", "second"},
    )
    assert [row["item_id"] for row in rows] == ["first", "second"]
    assert indices.tolist() == [0, 2]


def test_filter_metadata_indices_supports_multiple_chunks() -> None:
    metadata = [
        {"item_id": "document"},
        {"item_id": "document"},
        {"item_id": "other"},
    ]
    rows, indices = filter_metadata_indices(metadata, {"document"})
    assert len(rows) == 2
    assert indices.tolist() == [0, 1]
