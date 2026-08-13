from __future__ import annotations

import hashlib

import pytest

from scripts.compile_v19_selective_intervention_review import compile_review
from scripts.v19_selective_intervention_review_app import (
    ROLES,
    family_snapshot_sha256,
)


def _family(index: int) -> dict[str, object]:
    return {
        "family_id": f"family_{index}",
        "split": "development",
        "content_stratum": "pure_visual",
        "target": {"item_id": f"target_{index}"},
        "neighbor": {"item_id": f"neighbor_{index}"},
        "query_drafts": {role: f"第{index}组{role}查询文本" for role in ROLES},
        "changed_condition_kind": "color",
    }


def _completion(families: list[dict[str, object]]) -> dict[str, object]:
    material = "\n".join(family_snapshot_sha256(row) for row in families)
    return {
        "reviewer_id": "reviewer_01",
        "seen_family_ids": [str(row["family_id"]) for row in families],
        "family_bundle_sha256": hashlib.sha256(material.encode("utf-8")).hexdigest(),
    }


def test_compile_accepts_untouched_and_corrected_families() -> None:
    families = [_family(1), _family(2)]
    corrected = {role: f"修正后的第2组{role}查询文本" for role in ROLES}
    reviews = {
        "family_2": {
            "family_snapshot_sha256": family_snapshot_sha256(families[1]),
            "pair_decision": "accept",
            "query_texts": corrected,
            "changed_condition_kind": "relation",
        }
    }
    rows = compile_review(
        families=families,
        reviews=reviews,
        completion=_completion(families),
        reviewer_id="reviewer_01",
    )
    assert len(rows) == 8
    assert sum(row["query_review_status"] == "human_corrected" for row in rows) == 4
    assert all(row["status"] == "reviewed_not_frozen" for row in rows)


def test_compile_blocks_rejected_source_pair() -> None:
    families = [_family(1)]
    reviews = {
        "family_1": {
            "family_snapshot_sha256": family_snapshot_sha256(families[0]),
            "pair_decision": "replace_pair",
            "query_texts": families[0]["query_drafts"],
            "changed_condition_kind": "color",
        }
    }
    with pytest.raises(ValueError, match="require replacement"):
        compile_review(
            families=families,
            reviews=reviews,
            completion=_completion(families),
            reviewer_id="reviewer_01",
        )
