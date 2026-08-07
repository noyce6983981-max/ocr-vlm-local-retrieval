from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from ocr_vlm_retrieval.evaluation.query_collection import (
    build_textvqa_compositional_proposals,
    freeze_reviewed_queries,
    near_neighbor_transform,
    normalize_query,
    proposal_summary,
    validate_query_rows,
)
from ocr_vlm_retrieval.gating.attribute_coverage import load_attribute_policy

POLICY = load_attribute_policy(PROJECT_ROOT / "config/v17_attribute_coverage.json")


def test_near_neighbor_transform_changes_only_one_color_condition() -> None:
    transformed = near_neighbor_transform("What number is on the red sign?")
    assert transformed["query"] == "What number is on the blue sign?"
    assert transformed["transformation_type"] == "color_substitution"


def test_near_neighbor_relation_transform_uses_word_boundaries() -> None:
    transformed = near_neighbor_transform("What number is on the phone?")
    assert transformed["query"] == "What number is under the phone?"
    assert transformed["source_value"] == "on"


def test_group_leakage_is_rejected() -> None:
    rows = [
        {"query_id": "q1", "query": "one", "split": "calibration", "group_id": "g"},
        {"query_id": "q2", "query": "two", "split": "holdout", "group_id": "g"},
    ]
    with pytest.raises(ValueError, match="cross splits"):
        validate_query_rows(rows, expected_count=2)


def test_freeze_requires_complete_human_review_and_preserves_split() -> None:
    proposals = [
        {
            "query_id": "q1",
            "query": "What number is on the red sign?",
            "split": "calibration",
            "group_id": "g1",
            "route": "compositional_visual",
            "query_family": "color+relation",
            "query_origin": "human_authored_textvqa_source_question",
            "transformation_type": "none",
        }
    ]
    frozen = freeze_reviewed_queries(
        proposals,
        [
            {
                "query_id": "q1",
                "review_action": "approve",
                "reviewer_id": "human_a",
            }
        ],
        POLICY,
        expected_count=1,
    )
    assert frozen[0]["review_status"] == "human_query_approved"
    assert frozen[0]["split"] == "calibration"
    assert frozen[0]["transformation_type"] == "none"


def test_freeze_rejects_incomplete_or_noncompositional_rewrite() -> None:
    proposal = {
        "query_id": "q1",
        "query": "What number is on the red sign?",
        "split": "calibration",
        "group_id": "g1",
        "route": "compositional_visual",
        "query_family": "color+relation",
        "query_origin": "human_authored_textvqa_source_question",
    }
    with pytest.raises(ValueError, match="Review coverage mismatch"):
        freeze_reviewed_queries([proposal], [], POLICY, expected_count=1)
    with pytest.raises(ValueError, match="no longer compositional"):
        freeze_reviewed_queries(
            [proposal],
            [
                {
                    "query_id": "q1",
                    "review_action": "rewrite",
                    "reviewed_query": "forest",
                    "reviewer_id": "human_a",
                }
            ],
            POLICY,
            expected_count=1,
        )


def test_normalization_is_case_and_whitespace_insensitive() -> None:
    assert normalize_query("  What   Is This ") == "what is this"


def test_full_proposal_to_human_query_freeze_lifecycle() -> None:
    manifest = [
        {
            "item_id": f"item_{index:03d}",
            "public_source_name": "TextVQA real scene images",
            "public_source_file": f"image_{index:03d}.jpg",
            "source_group_id": f"source_{index:03d}",
            "hard_negative_group": "",
            "license": "CC BY 4.0",
            "source_tags": [
                "real_scene_text",
                "textvqa",
                f"question=What number {index} is on the red sign?",
            ],
        }
        for index in range(56)
    ]
    proposals = build_textvqa_compositional_proposals(
        manifest,
        POLICY,
        answerable_seed_count=56,
        total_count=80,
    )
    summary = proposal_summary(proposals)
    assert summary["query_count"] == 80
    assert summary["split_counts"] == {"calibration": 40, "holdout": 40}
    assert summary["origin_counts"] == {
        "human_authored_textvqa_source_question": 56,
        "near_neighbor_transformation_proposal": 24,
    }
    assert summary["group_count"] == 56

    reviews = [
        {
            "query_id": row["query_id"],
            "review_action": "approve",
            "reviewer_id": "human_query_reviewer",
        }
        for row in proposals
    ]
    frozen = freeze_reviewed_queries(
        proposals,
        reviews,
        POLICY,
        expected_count=80,
    )
    assert len(frozen) == 80
    assert {row["review_status"] for row in frozen} == {
        "human_query_approved"
    }
