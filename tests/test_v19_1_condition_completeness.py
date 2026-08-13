from __future__ import annotations

from ocr_vlm_retrieval.gating.literal_evidence_v19_1 import (
    select_v19_1_literal_candidate,
    v19_1_override_eligibility,
)
from ocr_vlm_retrieval.gating.ocr_literals_v19_1 import (
    extract_v19_1_literal_groups,
    parsed_ocr_dates,
)


def test_structured_date_parser_handles_ocr_punctuation_variants() -> None:
    dates = parsed_ocr_dates(["12-9.99", "DEC-9 1999", "July 31. 1997", "2020.6.15"])
    assert (1999, 12, 9) in dates
    assert (1997, 7, 31) in dates
    assert (2020, 6, 15) in dates


def test_structured_date_comparison_rejects_neighbor_day() -> None:
    decision = select_v19_1_literal_candidate(
        "找日期为1999年12月8日的 Covington & Burling 传真。",
        ["near"],
        {"near": ["COVINGTON & BURLING", "DEC-9 1999"]},
        fuzzy_threshold=0.88,
    )
    assert decision["eligible"] is True
    assert decision["accepted"] is False


def test_generic_chinese_organization_is_a_mandatory_condition() -> None:
    query = "找杭州今元标矩科技有限公司、联系电话末位为9的纳税人登记表。"
    groups = extract_v19_1_literal_groups(query)
    assert [(group.label, group.source) for group in groups] == [
        ("phone_number_ends_with_9", "phone_suffix"),
        ("杭州今元标矩科技有限公司", "chinese_entity"),
    ]


def test_phone_match_cannot_override_an_organization_mismatch() -> None:
    query = "找杭州今元标矩科技有限公司、联系电话末位为9的纳税人登记表。"
    decision = select_v19_1_literal_candidate(
        query,
        ["near"],
        {"near": ["杭州另一家科技有限公司", "联系电话 123456789"]},
        fuzzy_threshold=0.88,
    )
    assert decision["eligible"] is True
    assert decision["accepted"] is False


def test_identifier_requires_an_additional_condition() -> None:
    identifier_only = v19_1_override_eligibility("找编号17的报告。")
    with_name = v19_1_override_eligibility(
        "找编号17、标题为 Retail Excel Progress Report 的报告。"
    )
    assert identifier_only["eligible"] is False
    assert with_name["eligible"] is True


def test_identifier_and_document_topic_are_distinct_conditions() -> None:
    decision = v19_1_override_eligibility(
        "查找编号为 MC545715483586 的人体器官捐献登记表。"
    )
    assert decision["eligible"] is True
    assert [
        (group["label"], group["source"]) for group in decision["constraint_groups"]
    ] == [
        ("MC545715483586", "structured_identifier"),
        ("人体器官捐献登记表", "document_topic"),
    ]


def test_year_prefix_does_not_pollute_translated_organization() -> None:
    decision = v19_1_override_eligibility(
        "浏览1959年联邦贸易委员会与烟草研究机构通信的资料。"
    )
    labels = [group["label"] for group in decision["constraint_groups"]]
    assert "联邦贸易委员会" in labels
    assert "1959年联邦贸易委员会" not in labels


def test_incomplete_identifier_marker_forces_fallback() -> None:
    decision = v19_1_override_eligibility("找编号为的公司登记表。")
    assert decision["eligible"] is False
    assert decision["reason"] == "necessary_condition_extraction_incomplete"
    assert (
        "structured_identifier" in decision["completeness"]["missing_condition_sources"]
    )


def test_english_no_does_not_match_inside_reynolds() -> None:
    groups = extract_v19_1_literal_groups("浏览 R. J. Reynolds 的产品报告。")
    assert all(group.source != "structured_identifier" for group in groups)


def test_phone_suffix_allows_space_before_digit() -> None:
    groups = extract_v19_1_literal_groups("查找联系电话末位为 6 的应聘登记表。")
    assert ("phone_number_ends_with_6", "phone_suffix") in {
        (group.label, group.source) for group in groups
    }


def test_non_person_noun_is_not_extracted_after_patient_marker() -> None:
    groups = extract_v19_1_literal_groups("哪些资料讨论患者表现？")
    assert all(group.source != "chinese_name" for group in groups)


def test_structured_identifier_matches_inside_one_ocr_line() -> None:
    decision = select_v19_1_literal_candidate(
        "哪一页写有 PARK OUTDOOR ADVERTISING 和编号 1-016-U？",
        ["target"],
        {
            "target": [
                "PARK OUTDOCR ADVERTISING",
                "INVOICE NUMBER 1-016-U",
            ]
        },
        fuzzy_threshold=0.88,
    )
    assert decision["accepted"] is True


def test_short_latin_token_change_is_not_fuzzy_accepted() -> None:
    decision = select_v19_1_literal_candidate(
        "浏览 cyclic GMP response element 与 Chromogranin A 的资料。",
        ["near"],
        {
            "near": [
                "A functional cyclic AMP response element",
                "Chromogranin A",
            ]
        },
        fuzzy_threshold=0.88,
    )
    assert decision["accepted"] is False


def test_long_latin_neighbor_word_is_not_fuzzy_accepted() -> None:
    decision = select_v19_1_literal_candidate(
        "浏览 Hypoxia in Cultured Human Epithelium 的资料。",
        ["near"],
        {"near": ["Hypoxia in Cultured Human Endothelium"]},
        fuzzy_threshold=0.88,
    )
    assert decision["accepted"] is False


def test_flexible_year_and_named_field_are_mandatory() -> None:
    groups = extract_v19_1_literal_groups(
        "浏览 1974 年、验收部位为石灰石浆液回流管路的报告。"
    )
    assert ("1974年", "structured_year") in {
        (group.label, group.source) for group in groups
    }
    assert ("石灰石浆液回流管路", "named_field") in {
        (group.label, group.source) for group in groups
    }


def test_document_topic_is_pruned_when_two_stronger_conditions_exist() -> None:
    groups = extract_v19_1_literal_groups(
        "查找镇江市人才开发有限责任公司、编号为723514777的登记申请表。"
    )
    assert all(group.source != "document_topic" for group in groups)


def test_named_field_stops_at_chinese_possessive_marker() -> None:
    groups = extract_v19_1_literal_groups(
        "浏览填写人为 R. G. Ryan 的 NEWPORT LIGHTS HEAVY UP 进度报告。"
    )
    assert ("R. G. Ryan", "named_field") in {
        (group.label, group.source) for group in groups
    }


def test_five_character_name_neighbor_requires_exact_token() -> None:
    decision = select_v19_1_literal_candidate(
        "浏览 R. G. Bryan 填写的报告。",
        ["near"],
        {"near": ["FROM: R. G. Ryan"]},
        fuzzy_threshold=0.88,
    )
    assert decision["accepted"] is False


def test_hyphenated_term_preserves_attribute_binding() -> None:
    negative = select_v19_1_literal_candidate(
        "浏览 human adult beta-globin genes 的论文。",
        ["near"],
        {"near": ["human adult alpha-globin genes", "beta experiment"]},
        fuzzy_threshold=0.88,
    )
    positive = select_v19_1_literal_candidate(
        "浏览 human adult alpha-globin genes 的论文。",
        ["target"],
        {"target": ["human adult α-globin genes"]},
        fuzzy_threshold=0.88,
    )
    assert negative["accepted"] is False
    assert positive["accepted"] is True
