from __future__ import annotations

from ocr_vlm_retrieval.gating.ocr_literals import extract_ocr_literal_groups
from ocr_vlm_retrieval.gating.literal_evidence import (
    literal_override_is_eligible,
    select_literal_candidate,
)
from scripts.evaluate_v19_ocr_literal_override import paired_family_bootstrap


def test_literal_override_selects_candidate_covering_all_constraints() -> None:
    decision = select_literal_candidate(
        "Covington & Burling，日期为1999年12月9日",
        ["near", "gold"],
        {
            "near": ["COVINGTON & BURLING", "December 8, 1999"],
            "gold": ["COVINGTON & BURLING", "December 9, 1999"],
        },
        fuzzy_threshold=0.88,
    )
    assert decision["accepted"] is True
    assert decision["selected_item_id"] == "gold"


def test_literal_override_rejects_partial_near_neighbor() -> None:
    decision = select_literal_candidate(
        "Covington & Burling，日期为1999年12月8日",
        ["near"],
        {"near": ["COVINGTON & BURLING", "December 9, 1999"]},
        fuzzy_threshold=0.88,
    )
    assert decision["accepted"] is False
    assert decision["selected_item_id"] is None


def test_topic_override_requires_a_named_literal() -> None:
    named = extract_ocr_literal_groups("浏览1959年联邦贸易委员会的资料。")
    year_only = extract_ocr_literal_groups("找2001年的内部邮件。")
    assert literal_override_is_eligible("topic_discovery", named) is True
    assert literal_override_is_eligible("topic_discovery", year_only) is False


def test_literal_override_does_not_leak_to_visual_routes() -> None:
    groups = extract_ocr_literal_groups("找带有 KOOL Naturals 标志的绿色汽车。")
    assert literal_override_is_eligible("visual_compositional", groups) is False


def test_entity_override_accepts_only_exact_chinese_entities() -> None:
    chinese = extract_ocr_literal_groups("哪一页记录了患者赵海鹏？")
    english = extract_ocr_literal_groups("查找姓名为 Ronald S. Milstein 的记录。")
    assert literal_override_is_eligible("entity_identifier", chinese) is True
    assert literal_override_is_eligible("entity_identifier", english) is False


def test_paired_bootstrap_preserves_query_families() -> None:
    baseline = [
        {
            "query_id": f"family_{family}_q{query}",
            "gold_answerable": True,
            "gold_relevant_item_ids": ["gold"],
            "selected_item_id": None,
            "accepted": False,
        }
        for family in range(2)
        for query in range(1, 5)
    ]
    candidate = [
        {**row, "selected_item_id": "gold", "accepted": True}
        for row in baseline
    ]
    result = paired_family_bootstrap(baseline, candidate, samples=100, seed=7)
    assert result["family_count"] == 2
    assert result["observed_e2e_delta"] == 1.0
    assert result["confidence_interval_95"] == [1.0, 1.0]
