from __future__ import annotations

from pathlib import Path

import pytest

from scripts.v19_formal_family_review_app import (
    family_snapshot_sha256,
    load_families,
    load_reviews,
    next_unreviewed_index,
    review_path_for,
    save_review,
)


def test_formal_review_app_loads_sixty_exact_families() -> None:
    families = load_families()
    assert len(families) == 60
    assert sum(len(row["queries"]) for row in families) == 240
    assert len({row["family_id"] for row in families}) == 60


def test_family_review_persists_exact_snapshot(tmp_path: Path) -> None:
    family = load_families()[0]
    path = tmp_path / "reviews.csv"
    reviews = save_review(
        path,
        family=family,
        final_route=str(family["proposed_route"]),
        reviewer_id="reviewer_01",
        notes="确认",
    )
    saved = reviews[str(family["family_id"])]
    assert saved["family_snapshot_sha256"] == family_snapshot_sha256(family)
    assert saved["accepted_proposal"] == "true"
    assert load_reviews(path)[str(family["family_id"])]["notes"] == "确认"


def test_family_review_detects_changed_queries(tmp_path: Path) -> None:
    family = load_families()[0]
    path = tmp_path / "reviews.csv"
    save_review(
        path,
        family=family,
        final_route=str(family["proposed_route"]),
        reviewer_id="reviewer_01",
        notes="",
    )
    changed = dict(family)
    changed["queries"] = [*family["queries"][:-1], "被替换的查询"]
    with pytest.raises(ValueError, match="content changed"):
        save_review(
            path,
            family=changed,
            final_route=str(family["proposed_route"]),
            reviewer_id="reviewer_01",
            notes="",
        )


def test_family_review_navigation_and_reviewer_id(tmp_path: Path) -> None:
    families = load_families()
    reviews = {str(families[1]["family_id"]): {"family_id": "saved"}}
    assert next_unreviewed_index(families, reviews, 0) == 2
    assert review_path_for("reviewer_02").name == "reviewer_02_family_reviews.csv"
    with pytest.raises(ValueError):
        review_path_for("../escape")
