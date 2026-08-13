from __future__ import annotations

from ocr_vlm_retrieval.gating.literal_evidence_v19_2 import (
    is_explicit_topic_discovery,
    select_v19_2_literal_candidate,
    v19_2_override_eligibility,
)
from ocr_vlm_retrieval.gating.ocr_literals_v19_2 import (
    complete_explicit_evidence_item_ids,
    extract_v19_2_literal_groups,
)


def test_topic_discovery_aliases_are_query_text_only() -> None:
    assert is_explicit_topic_discovery("有哪些文档讨论 KOOL Naturals 的宣传方案？")
    assert is_explicit_topic_discovery(
        "有哪些材料涉及 Federal Trade Commission 的烟草研究咨询？"
    )
    assert not is_explicit_topic_discovery("KOOL Naturals")


def test_named_topic_alias_activates_literal_contract() -> None:
    decision = v19_2_override_eligibility(
        "有哪些文档讨论 KOOL Naturals 的宣传方案？"
    )
    assert decision["eligible"] is True
    assert decision["reason"] == "explicit_named_discovery_contract_v19_2"


def test_arbitrary_layout_query_does_not_activate() -> None:
    decision = v19_2_override_eligibility("有哪些文档同时含有表格和图片？")
    assert decision["eligible"] is False


def test_explicit_conjunction_extracts_every_quoted_condition() -> None:
    query = "哪份资料必须包含“Alpha Project”，并同时包含“Budget 2026”？"
    groups = extract_v19_2_literal_groups(query)
    explicit = [group.label for group in groups if group.source == "explicit_required"]
    assert explicit == ["Alpha Project", "Budget 2026"]
    assert len(groups) == 2


def test_explicit_conjunction_requires_both_conditions_on_one_candidate() -> None:
    query = "哪份资料必须包含“Alpha Project”，并同时包含“Budget 2026”？"
    decision = select_v19_2_literal_candidate(
        query,
        ["partial", "complete"],
        {
            "partial": ["Alpha Project"],
            "complete": ["Alpha Project", "Budget 2026"],
        },
        fuzzy_threshold=0.88,
    )
    assert decision["selected_item_id"] == "complete"


def test_explicit_conjunction_ignores_identifier_markers_inside_quotes() -> None:
    decision = v19_2_override_eligibility(
        "哪份资料必须包含“Original invoice number”，并同时包含“Virginia 23219”？"
    )
    assert decision["eligible"] is True
    assert decision["completeness"]["complete"] is True


def test_explicit_conjunction_does_not_fuzzy_accept_near_neighbor() -> None:
    query = "哪份资料必须包含“DIRECT ACCOUNTS”，并同时包含“OUTSIDE REGION”？"
    decision = select_v19_2_literal_candidate(
        query,
        ["near", "exact"],
        {
            "near": ["DIRECT ACCOUNT", "OUTSIDE REGIONS"],
            "exact": ["DIRECT ACCOUNTS", "OUTSIDE REGION"],
        },
        fuzzy_threshold=0.84,
    )
    assert decision["selected_item_id"] == "exact"


def test_natural_conjunction_variants_preserve_two_condition_contract() -> None:
    queries = (
        "帮我找同时写有“Alpha Project”和“Budget 2026”的那一页。",
        "请定位一份资料，其中“Alpha Project”与“Budget 2026”必须在同页出现。",
        "页面需要同时出现“Alpha Project”；另一个不可缺少的内容是“Budget 2026”。",
        "我只要同时能看到“Alpha Project”以及“Budget 2026”的资料，缺一项都不要。",
    )
    for query in queries:
        groups = extract_v19_2_literal_groups(query)
        assert [group.label for group in groups] == ["Alpha Project", "Budget 2026"]
        assert v19_2_override_eligibility(query)["eligible"] is True


def test_multiple_quote_styles_preserve_query_order() -> None:
    queries = (
        "同页出现「Alpha Project」和「Budget 2026」。",
        "同时核验【Alpha Project】以及【Budget 2026】。",
    )
    for query in queries:
        assert [group.label for group in extract_v19_2_literal_groups(query)] == [
            "Alpha Project",
            "Budget 2026",
        ]


def test_complete_explicit_evidence_promotes_only_same_page_match() -> None:
    groups = extract_v19_2_literal_groups(
        "找同时写有“Alpha Project”和“Budget 2026”的页面。"
    )
    matches = complete_explicit_evidence_item_ids(
        groups,
        {
            "partial_a": ["Alpha Project"],
            "partial_b": ["Budget 2026"],
            "complete": ["Alpha Project", "Budget 2026"],
        },
    )
    assert matches == ["complete"]


def test_complete_explicit_evidence_rejects_empty_normalized_condition() -> None:
    groups = extract_v19_2_literal_groups('必须包含“++”并同时包含“Invoice 42”')

    assert complete_explicit_evidence_item_ids(
        groups, {"page": ["Invoice 42"]}
    ) == []
