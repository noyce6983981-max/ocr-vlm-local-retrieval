from __future__ import annotations

from collections import Counter

import pytest

from scripts.finalize_v19_formal_primary_review import (
    freeze_family_reviews,
    materialize_reviewed_queries,
    read_csv,
)
from scripts.v19_formal_family_review_app import (
    family_snapshot_sha256,
    load_families,
)


def test_primary_review_default_accept_materializes_balanced_splits() -> None:
    families = load_families()
    first = families[0]
    explicit = {
        str(first["family_id"]): {
            "family_id": str(first["family_id"]),
            "family_snapshot_sha256": family_snapshot_sha256(first),
            "split": str(first["split"]),
            "proposed_route": str(first["proposed_route"]),
            "final_route": str(first["proposed_route"]),
            "reviewer_id": "reviewer_01",
            "reviewed_at_unix": "1.0",
            "accepted_proposal": "true",
            "notes": "",
        }
    }
    frozen = freeze_family_reviews(
        families, explicit, reviewer_id="reviewer_01", frozen_at_unix=2.0
    )
    assert len(frozen) == 60
    assert Counter(row["acceptance_mode"] for row in frozen) == {
        "explicit_saved": 1,
        "default_accept_after_full_browse": 59,
    }

    calibration = materialize_reviewed_queries(
        read_csv(
            families_path().parent / "calibration_queries_draft.csv"
        ),
        families,
        frozen,
        split="calibration",
    )
    holdout = materialize_reviewed_queries(
        read_csv(families_path().parent / "holdout_queries_draft.csv"),
        families,
        frozen,
        split="holdout",
    )
    assert len(calibration) == len(holdout) == 120
    assert Counter(row["gold_route"] for row in calibration) == {
        route: 20
        for route in {
            "text_evidence",
            "visual_discovery",
            "visual_metadata",
            "entity_exact",
            "topic_discovery",
            "mixed",
        }
    }
    assert {row["status"] for row in holdout} == {
        "primary_reviewed_formal_holdout"
    }


def families_path():
    from scripts.v19_formal_family_review_app import FAMILY_PATH

    return FAMILY_PATH


def test_primary_review_rejects_snapshot_mismatch() -> None:
    families = load_families()
    first = families[0]
    explicit = {
        str(first["family_id"]): {
            "family_snapshot_sha256": "bad",
            "split": str(first["split"]),
            "proposed_route": str(first["proposed_route"]),
            "final_route": str(first["proposed_route"]),
        }
    }
    with pytest.raises(ValueError, match="snapshot mismatch"):
        freeze_family_reviews(
            families, explicit, reviewer_id="reviewer_01", frozen_at_unix=2.0
        )
