from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from ocr_vlm_retrieval.studies.query_split import (
    STRATA,
    build_authoring_queue,
    freeze_authored_queries,
    query_set_fingerprint,
    source_identity_keys,
    validate_authored_pair,
)


def _source(index: int, category: str, *, has_text: bool = False) -> dict[str, object]:
    return {
        "item_id": f"item_{index:03d}",
        "source_path": f"images/item_{index:03d}.jpg",
        "source_group_id": f"source_{index:03d}",
        "hard_negative_group": "",
        "perceptual_group": f"perceptual_{index:03d}",
        "privacy_review_required": False,
        "dedup_role": "representative",
        "taxonomy_v1_category": category,
        "has_text_expected": has_text,
        "public_source_name": f"dataset_{index % 5}",
    }


def _manifest() -> list[dict[str, object]]:
    rows = [_source(index, "clear_document", has_text=True) for index in range(20)]
    rows.extend(_source(index, "natural_no_text") for index in range(20, 80))
    return rows


def _ocr() -> dict[str, dict[str, object]]:
    return {
        f"item_{index:03d}": {
            "status": "cached",
            "character_count": 30 if index < 20 else 0,
            "text_box_count": 3 if index < 20 else 0,
            "ocr_excerpt": f"TEXT {index}",
        }
        for index in range(80)
    }


def test_authoring_queue_is_deterministic_balanced_and_disjoint() -> None:
    first = build_authoring_queue(_manifest(), _ocr())
    second = build_authoring_queue(reversed(_manifest()), _ocr())
    assert first == second
    assert len(first) == 80
    assert Counter(row["stratum"] for row in first) == Counter(
        {stratum: 20 for stratum in STRATA}
    )
    assert Counter((row["split"], row["language_target"]) for row in first) == {
        ("calibration", "zh"): 20,
        ("calibration", "en"): 20,
        ("holdout", "zh"): 20,
        ("holdout", "en"): 20,
    }
    assert len({row["source_item_id"] for row in first}) == 80


def test_predecessor_identity_is_excluded() -> None:
    rows = _manifest() + [_source(80, "natural_no_text")]
    excluded = source_identity_keys(rows[20])
    queue = build_authoring_queue(rows, _ocr(), excluded_identity_keys=excluded)
    assert "item_020" not in {row["source_item_id"] for row in queue}


def test_authoring_queue_can_apply_chinese_only_protocol_amendment() -> None:
    queue = build_authoring_queue(_manifest(), _ocr(), languages=("zh",))
    assert len(queue) == 80
    assert {row["language_target"] for row in queue} == {"zh"}
    assert Counter((row["stratum"], row["split"]) for row in queue) == {
        (stratum, split): 10
        for stratum in STRATA
        for split in ("calibration", "holdout")
    }


def _query_pair(row: dict[str, object], index: int) -> tuple[str, str, str]:
    language = str(row["language_target"])
    stratum = str(row["stratum"])
    if stratum == "spatial_relation":
        if language == "zh":
            return (
                f"请找红色汽车位于标志左边的图片编号{index}",
                f"请找红色汽车位于标志右边的图片编号{index}",
                "relation",
            )
        return (
            f"Find red car number {index} to the left of the sign",
            f"Find red car number {index} to the right of the sign",
            "relation",
        )
    if stratum == "ocr_visual_condition":
        if language == "zh":
            return (
                f"请找红色标牌上写着编号A{index}的图片",
                f"请找红色标牌上写着编号B{index}的图片",
                "ocr_text",
            )
        return (
            f"Find the red sign showing code A{index}",
            f"Find the red sign showing code B{index}",
            "ocr_text",
        )
    if language == "zh":
        return (
            f"请找蓝色汽车旁边有标志的场景编号{index}",
            f"请找黄色汽车旁边有标志的场景编号{index}",
            "color",
        )
    return (
        f"Find scene number {index} with a blue car beside a sign",
        f"Find scene number {index} with a yellow car beside a sign",
        "color",
    )


def test_approved_pairs_expand_to_160_group_locked_queries() -> None:
    queue = build_authoring_queue(_manifest(), _ocr())
    submissions = []
    for index, row in enumerate(queue):
        positive, negative, changed = _query_pair(row, index)
        submissions.append(
            {
                "authoring_id": row["authoring_id"],
                "review_action": "approve",
                "positive_query": positive,
                "hard_negative_query": negative,
                "changed_condition_kind": changed,
                "single_condition_confirmed": True,
                "reviewer_id": "human_query_author",
            }
        )
    frozen = freeze_authored_queries(
        queue, submissions, study_id="v18-condition-aware-listwise-160"
    )
    assert len(frozen) == 160
    assert Counter(row["split"] for row in frozen) == {
        "calibration": 80,
        "holdout": 80,
    }
    assert Counter(row["query_role"] for row in frozen) == {
        "positive": 80,
        "single_condition_hard_negative": 80,
    }
    assert len(query_set_fingerprint(frozen)) == 64


def test_pair_validation_rejects_language_and_confirmation_errors() -> None:
    with pytest.raises(ValueError, match="Chinese-target"):
        validate_authored_pair(
            "Find a red car near a sign",
            "Find a blue car near a sign",
            language_target="zh",
            changed_condition_kind="color",
            single_condition_confirmed=True,
        )
    with pytest.raises(ValueError, match="confirm"):
        validate_authored_pair(
            "请找红色汽车旁边的标志",
            "请找蓝色汽车旁边的标志",
            language_target="zh",
            changed_condition_kind="color",
            single_condition_confirmed=False,
        )
