from __future__ import annotations

from scripts.prepare_v19_selective_intervention_review import (
    STRATUM_CATEGORIES,
    build_source_families,
)


def test_source_family_queue_is_disjoint_and_balanced() -> None:
    manifest = []
    ocr_counts = {}
    categories = {
        "ocr_literal_lookup": "general_text_document",
        "pure_visual": "natural_no_text",
        "layout_table": "table_form_ticket",
        "entity_identifier": "general_text_document",
        "visual_compositional": "natural_no_text",
        "text_visual_compositional": "scene_text",
        "topic_discovery": "complex_academic",
        "relationship_scene": "scene_text",
    }
    for group_index, (_stratum, category) in enumerate(
        categories.items(), start=1
    ):
        for neighbor_index in range(2):
            item_id = f"item_{group_index}_{neighbor_index}"
            manifest.append(
                {
                    "item_id": item_id,
                    "source_path": f"images/{item_id}.jpg",
                    "category": category,
                    "hard_negative_group": f"group_{group_index}",
                    "privacy_review_required": False,
                    "dedup_role": "representative",
                }
            )
            ocr_counts[item_id] = 200

    rows = build_source_families(
        manifest,
        stratum_counts={name: 1 for name in STRATUM_CATEGORIES},
        excluded_item_ids=set(),
        ocr_character_counts=ocr_counts,
        seed="test-seed",
    )

    assert len(rows) == 8
    assert {row["content_stratum"] for row in rows} == set(STRATUM_CATEGORIES)
    assert {row["review_status"] for row in rows} == {
        "pending_human_review_not_frozen"
    }
    source_ids = {
        row[role]["item_id"] for row in rows for role in ("target", "neighbor")
    }
    assert len(source_ids) == 16


def test_source_family_queue_excludes_entire_predecessor_group() -> None:
    manifest = []
    ocr_counts = {}
    for group_index in range(9):
        for neighbor_index in range(2):
            item_id = f"item_{group_index}_{neighbor_index}"
            manifest.append(
                {
                    "item_id": item_id,
                    "source_path": f"images/{item_id}.jpg",
                    "category": "general_text_document",
                    "hard_negative_group": f"group_{group_index}",
                    "privacy_review_required": False,
                    "dedup_role": "representative",
                }
            )
            ocr_counts[item_id] = 200

    counts = {name: 0 for name in STRATUM_CATEGORIES}
    counts["ocr_literal_lookup"] = 1
    rows = build_source_families(
        manifest,
        stratum_counts=counts,
        excluded_item_ids={"item_0_0"},
        ocr_character_counts=ocr_counts,
        seed="test-seed",
    )

    selected = {rows[0][role]["item_id"] for role in ("target", "neighbor")}
    assert selected.isdisjoint({"item_0_0", "item_0_1"})
