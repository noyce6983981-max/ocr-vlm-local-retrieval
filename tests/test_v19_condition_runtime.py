from __future__ import annotations

import json

from ocr_vlm_retrieval.runtime.v19_condition_runtime import (
    alias_fallback_payload,
    build_condition_ranking,
    ordered_candidate_ids,
)
from scripts.v19_live_search import _ocr_lines


def test_ordered_candidate_ids_promotes_verified_candidate() -> None:
    assert ordered_candidate_ids(["a", "b", "c"], "b") == ["b", "a", "c"]


def test_build_condition_ranking_preserves_original_rank() -> None:
    ranking = build_condition_ranking(
        {"a": {"item_id": "a"}, "b": {"item_id": "b"}},
        ["a", "b"],
        {"a": 2.0, "b": 1.5},
        [
            {
                "item_id": "a",
                "retrieval_rank": 1,
                "constraint_count": 2,
                "matched_constraint_count": 1,
                "all_constraints_matched": False,
            },
            {
                "item_id": "b",
                "retrieval_rank": 2,
                "constraint_count": 2,
                "matched_constraint_count": 2,
                "all_constraints_matched": True,
            },
        ],
        selected_item_id="b",
    )
    assert [row["item_id"] for row in ranking] == ["b", "a"]
    assert ranking[0]["v19_original_retrieval_rank"] == 2


def test_alias_fallback_payload_keeps_provenance() -> None:
    payload = {
        "rankings": {"quality_hybrid": [{"item_id": "a"}]},
        "acceptance": {
            "quality_hybrid": {"accepted": True, "reason": "旧链路通过。"}
        },
    }
    result = alias_fallback_payload(payload, reason="查询没有完整必要条件")
    assert result["rankings"]["v19_condition"] == [{"item_id": "a"}]
    assert result["v19_condition_runtime"]["intervened"] is False
    assert "没有完整必要条件" in result["acceptance"]["v19_condition"]["reason"]


def test_ocr_lines_prefers_human_override(tmp_path) -> None:
    base_dir = tmp_path / "ocr/json"
    override_dir = tmp_path / "ocr/overrides"
    base_dir.mkdir(parents=True)
    override_dir.mkdir(parents=True)
    (base_dir / "page.json").write_text(
        json.dumps({"rec_texts": ["base"], "rec_scores": [0.99]}),
        encoding="utf-8",
    )
    (override_dir / "page.json").write_text(
        json.dumps({"rec_texts": ["人工纠错"]}, ensure_ascii=False),
        encoding="utf-8",
    )

    assert _ocr_lines(tmp_path, "page", minimum_confidence=0.35) == ["人工纠错"]
