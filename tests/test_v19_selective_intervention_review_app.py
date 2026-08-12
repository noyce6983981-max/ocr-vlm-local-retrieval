from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.v19_selective_intervention_review_app import (
    ROLES,
    completion_path_for,
    family_snapshot_sha256,
    load_reviews,
    mark_family_seen,
    progress_path_for,
    review_path_for,
    save_completion,
    save_review,
)


def _family(index: int = 1) -> dict[str, object]:
    return {
        "family_id": f"family_{index}",
        "split": "development",
        "content_stratum": "pure_visual",
        "target": {"item_id": f"target_{index}"},
        "neighbor": {"item_id": f"neighbor_{index}"},
        "query_drafts": {role: f"这是第{index}组的{role}测试查询" for role in ROLES},
        "changed_condition_kind": "color",
    }


def test_review_paths_reject_unsafe_reviewer_id() -> None:
    with pytest.raises(ValueError, match="审核者ID"):
        review_path_for("../unsafe")
    assert completion_path_for("reviewer_01").name == "reviewer_01_completion.json"
    assert progress_path_for("reviewer_01").name == (
        "reviewer_01_browse_progress.json"
    )


def test_review_save_is_atomic_and_snapshot_bound(tmp_path: Path) -> None:
    family = _family()
    path = tmp_path / "review.jsonl"
    reviews = save_review(
        path,
        family=family,
        query_texts=family["query_drafts"],  # type: ignore[arg-type]
        changed_condition_kind="color",
        pair_decision="accept",
        notes="checked",
        reviewer_id="reviewer_01",
    )
    assert reviews["family_1"]["family_snapshot_sha256"] == (
        family_snapshot_sha256(family)
    )
    assert load_reviews(path)["family_1"]["notes"] == "checked"


def test_completion_requires_every_family_seen(tmp_path: Path) -> None:
    families = [_family(1), _family(2)]
    path = tmp_path / "completion.json"
    with pytest.raises(ValueError, match="完整浏览"):
        save_completion(
            path,
            reviewer_id="reviewer_01",
            families=families,
            seen_family_ids={"family_1"},
        )
    save_completion(
        path,
        reviewer_id="reviewer_01",
        families=families,
        seen_family_ids={"family_1", "family_2"},
    )
    assert json.loads(path.read_text(encoding="utf-8"))["eligible_for_freeze"] is False


def test_browse_progress_survives_process_restart(tmp_path: Path) -> None:
    families = [_family(1), _family(2)]
    path = tmp_path / "progress.json"
    first = mark_family_seen(
        path,
        reviewer_id="reviewer_01",
        families=families,
        family_id="family_1",
    )
    second = mark_family_seen(
        path,
        reviewer_id="reviewer_01",
        families=families,
        family_id="family_2",
    )
    assert first == {"family_1"}
    assert second == {"family_1", "family_2"}
    assert json.loads(path.read_text(encoding="utf-8"))["seen_family_count"] == 2
