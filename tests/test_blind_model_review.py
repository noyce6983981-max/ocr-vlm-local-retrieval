from __future__ import annotations

import json

import pytest

from scripts.blind_model_review import (
    consensus_model_reviews,
    partition_model_reviews,
    validate_model_review_batch,
)


def safe_bundle():
    return {
        "q1": {
            "query_id": "q1",
            "query": "年度报告",
            "candidates": [
                {"item_id": "a", "source_path": "a.jpg", "ocr_text": ""}
            ],
        }
    }


def write_batch(path, review):
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "reviewer": "memoryless_test",
                "bundle_sha256": "bundle-hash",
                "reviews": [review],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_model_review_validation_and_partition(tmp_path) -> None:
    path = tmp_path / "batch.json"
    write_batch(
        path,
        {
            "query_id": "q1",
            "decision": "answerable",
            "relevant_item_ids": ["a"],
            "confidence": 0.96,
            "notes_zh": "标题明确写有年度报告。",
            "uncertain_item_ids": [],
        },
    )
    rows = validate_model_review_batch(
        path,
        safe_bundle=safe_bundle(),
        expected_query_ids={"q1"},
        expected_bundle_sha256="bundle-hash",
    )
    auto, second = partition_model_reviews(rows)
    assert len(auto) == 1
    assert not second


def test_uncertain_or_low_confidence_review_requires_second_pass(tmp_path) -> None:
    path = tmp_path / "batch.json"
    write_batch(
        path,
        {
            "query_id": "q1",
            "decision": "ambiguous",
            "relevant_item_ids": [],
            "confidence": 0.70,
            "notes_zh": "图片模糊，无法确认。",
            "uncertain_item_ids": ["a"],
        },
    )
    rows = validate_model_review_batch(
        path,
        safe_bundle=safe_bundle(),
        expected_query_ids={"q1"},
        expected_bundle_sha256="bundle-hash",
    )
    auto, second = partition_model_reviews(rows)
    assert not auto
    assert len(second) == 1


def test_model_review_rejects_hidden_or_unknown_candidate(tmp_path) -> None:
    path = tmp_path / "batch.json"
    write_batch(
        path,
        {
            "query_id": "q1",
            "decision": "answerable",
            "relevant_item_ids": ["not_in_pool"],
            "confidence": 0.99,
            "notes_zh": "错误候选。",
            "uncertain_item_ids": [],
        },
    )
    with pytest.raises(ValueError, match="unknown candidate"):
        validate_model_review_batch(
            path,
            safe_bundle=safe_bundle(),
            expected_query_ids={"q1"},
            expected_bundle_sha256="bundle-hash",
        )


def test_consensus_accepts_exact_clean_majority() -> None:
    base = {
        "query_id": "q1",
        "decision": "answerable",
        "relevant_item_ids": ["a"],
        "confidence": 0.95,
        "notes_zh": "明确相关",
        "uncertain_item_ids": [],
    }
    first = [{**base, "reviewer": "a"}]
    second = [{**base, "reviewer": "b", "confidence": 0.91}]
    third = [
        {
            **base,
            "reviewer": "c",
            "decision": "no_answer",
            "relevant_item_ids": [],
        }
    ]
    accepted, unresolved = consensus_model_reviews([first, second, third])
    assert len(accepted) == 1
    assert accepted[0]["relevant_item_ids"] == ["a"]
    assert not unresolved


def test_consensus_leaves_uncertain_majority_unresolved() -> None:
    rows = [
        [
            {
                "query_id": "q1",
                "decision": "answerable",
                "relevant_item_ids": ["a"],
                "confidence": 0.95,
                "notes_zh": "可能相关",
                "uncertain_item_ids": ["b"],
                "reviewer": reviewer,
            }
        ]
        for reviewer in ("a", "b", "c")
    ]
    accepted, unresolved = consensus_model_reviews(rows)
    assert not accepted
    assert len(unresolved) == 1
