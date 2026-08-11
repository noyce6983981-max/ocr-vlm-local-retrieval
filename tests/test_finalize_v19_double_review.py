from __future__ import annotations

from scripts.finalize_v19_double_review import (
    cohen_kappa,
    finalize_query_status,
    validate_double_review,
)
from scripts.finalize_v19_formal_primary_review import read_csv
from scripts.prepare_v19_second_review import build_manifest
from scripts.v19_formal_family_review_app import load_families


def test_double_review_perfect_agreement_and_final_status() -> None:
    families = load_families()
    manifest = build_manifest(families, "seed")
    primary = []
    secondary = {}
    originals = {row["family_id"]: row for row in families}
    for family in manifest["families"]:
        route = originals[family["family_id"]]["proposed_route"]
        primary.append(
            {
                "family_id": family["family_id"],
                "final_route": route,
                "reviewer_id": "reviewer_01",
            }
        )
        secondary[family["family_id"]] = {
            "family_id": family["family_id"],
            "family_snapshot_sha256": family["family_snapshot_sha256"],
            "split": family["split"],
            "final_route": route,
            "reviewer_id": "reviewer_02",
        }
    disagreements, summary = validate_double_review(
        manifest, primary, secondary
    )
    assert disagreements == []
    assert summary["raw_agreement"] == 1.0
    assert summary["cohen_kappa"] == 1.0
    calibration = finalize_query_status(
        read_csv(
            families_path().parent / "calibration_queries_reviewed.csv"
        ),
        split="calibration",
    )
    assert {row["status"] for row in calibration} == {
        "adjudicated_formal_calibration_30pct_double_review"
    }


def families_path():
    from scripts.v19_formal_family_review_app import FAMILY_PATH

    return FAMILY_PATH


def test_cohen_kappa_handles_balanced_disagreement() -> None:
    assert cohen_kappa(["mixed", "text_evidence"], ["text_evidence", "mixed"]) < 0
