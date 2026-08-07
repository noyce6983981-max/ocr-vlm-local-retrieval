"""Unit tests for feedback category model helpers."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from scripts.train_feedback_category_model import (
    build_feature_matrix,
    effective_category,
    is_category_review,
    l2_normalize,
    read_review_splits,
    resolve_row_splits,
)
from scripts.assistant_category_proposals import (
    is_active_assistant_proposal,
    read_assistant_proposals,
    resolve_assistant_proposal,
    write_assistant_proposals,
)


def test_human_reclassification_overrides_weak_label() -> None:
    row = {"category": "complex_academic"}
    review = {
        "decision": "reclassified",
        "revised_category": "table_form_ticket",
    }
    assert effective_category(row, review) == "table_form_ticket"
    assert effective_category(row, {"decision": "accepted"}) == (
        "complex_academic"
    )


def test_assistant_proposal_is_lower_priority_than_human_truth() -> None:
    row = {"category": "ppt_poster_slide"}
    proposal = {
        "status": "pending",
        "proposed_category": "general_text_document",
    }
    assert effective_category(row, None, proposal) == (
        "general_text_document"
    )
    assert effective_category(
        row,
        {
            "decision": "reclassified",
            "revised_category": "table_form_ticket",
        },
        proposal,
    ) == "table_form_ticket"
    assert effective_category(
        row,
        {"decision": "accepted"},
        proposal,
    ) == "ppt_poster_slide"


def test_assistant_proposal_round_trip_and_resolution(
    tmp_path: Path,
) -> None:
    path = tmp_path / "assistant.csv"
    write_assistant_proposals(
        path,
        {
            "page": {
                "item_id": "page",
                "proposed_category": "general_text_document",
                "assistant_notes": "looks like a press release",
                "proposal_round": "4",
                "status": "pending",
            }
        },
    )
    proposal = read_assistant_proposals(path)["page"]
    assert is_active_assistant_proposal(proposal)
    resolve_assistant_proposal(
        path,
        item_id="page",
        human_category="table_form_ticket",
    )
    resolved = read_assistant_proposals(path)["page"]
    assert resolved["status"] == "revised"
    assert not is_active_assistant_proposal(resolved)


def test_l2_normalize_handles_zero_vector() -> None:
    matrix = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    normalized = l2_normalize(matrix)
    np.testing.assert_allclose(normalized[0], [0.6, 0.8])
    np.testing.assert_allclose(normalized[1], [0.0, 0.0])


def test_only_usable_category_decisions_enter_training() -> None:
    assert is_category_review({"decision": "accepted"})
    assert is_category_review({"decision": "reclassified"})
    assert not is_category_review({"decision": "ocr_retry"})
    assert not is_category_review({"decision": "quarantined"})
    assert not is_category_review(None)


def test_read_review_splits_rejects_duplicate_ids(
    tmp_path: Path,
) -> None:
    path = tmp_path / "splits.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["item_id", "split"]
        )
        writer.writeheader()
        writer.writerows(
            [
                {"item_id": "same", "split": "train"},
                {"item_id": "same", "split": "test"},
            ]
        )
    try:
        read_review_splits(path)
    except ValueError as error:
        assert "Duplicate item ID" in str(error)
    else:
        raise AssertionError("Expected duplicate split ID to fail.")


def test_frozen_holdout_rows_are_excluded_from_fit_mask() -> None:
    rows = [
        {"item_id": "train_page"},
        {"item_id": "validation_page"},
        {"item_id": "test_page"},
    ]
    splits = resolve_row_splits(
        rows,
        {
            "train_page": "train",
            "validation_page": "validation",
            "test_page": "test",
        },
    )
    assert list(splits == "train") == [True, False, False]


def test_feature_matrix_excludes_disabled_pages() -> None:
    manifest = [
        {"item_id": "active", "category": "general_text_document"},
        {
            "item_id": "disabled",
            "category": "general_text_document",
            "search_enabled": False,
        },
    ]
    visual = {
        "active": np.array([1.0, 0.0], dtype=np.float32),
        "disabled": np.array([0.0, 1.0], dtype=np.float32),
    }
    text = {
        "active": np.array([0.0, 1.0], dtype=np.float32),
        "disabled": np.array([1.0, 0.0], dtype=np.float32),
    }
    rows, matrix = build_feature_matrix(manifest, visual, text, {})
    assert [row["item_id"] for row in rows] == ["active"]
    assert matrix.shape == (1, 8)


def test_page_without_ocr_text_uses_zero_text_features() -> None:
    manifest = [
        {"item_id": "image_only", "category": "natural_no_text"},
    ]
    visual = {
        "image_only": np.array([1.0, 0.0], dtype=np.float32),
    }
    text = {
        "another_page": np.array([0.0, 1.0], dtype=np.float32),
    }
    rows, matrix = build_feature_matrix(manifest, visual, text, {})
    assert rows[0]["item_id"] == "image_only"
    np.testing.assert_allclose(matrix[0, 2:4], [0.0, 0.0])
