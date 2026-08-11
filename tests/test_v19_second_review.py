from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from scripts.prepare_v19_second_review import build_manifest
from scripts.v19_formal_family_review_app import load_families
from scripts.v19_second_family_review_app import (
    load_reviews,
    review_path_for,
    save_review,
)


def test_second_review_selection_is_balanced_and_label_blind() -> None:
    families = load_families()
    manifest = build_manifest(families, "seed")
    selected = manifest["families"]
    assert len(selected) == 18
    assert Counter(row["split"] for row in selected) == {
        "calibration": 9,
        "holdout": 9,
    }
    assert all("proposed_route" not in row for row in selected)
    originals = {row["family_id"]: row for row in families}
    route_counts = Counter(
        originals[row["family_id"]]["proposed_route"] for row in selected
    )
    assert route_counts == {
        route: 3
        for route in {
            "text_evidence",
            "visual_discovery",
            "visual_metadata",
            "entity_exact",
            "topic_discovery",
            "mixed",
        }
    }


def test_second_review_requires_independent_id_and_persists(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="不同"):
        review_path_for("reviewer_01")
    family = build_manifest(load_families(), "seed")["families"][0]
    path = tmp_path / "second.csv"
    reviews = save_review(
        path,
        {},
        family=family,
        final_route="mixed",
        reviewer_id="reviewer_02",
        notes="独立判断",
    )
    assert reviews[family["family_id"]]["final_route"] == "mixed"
    assert load_reviews(path)[family["family_id"]]["notes"] == "独立判断"
