from __future__ import annotations

from ocr_vlm_retrieval.gating.ocr_literals import (
    evaluate_ocr_literal_groups,
    extract_ocr_literal_groups,
)


def test_extracts_organization_and_date_as_separate_constraints() -> None:
    groups = extract_ocr_literal_groups(
        "找出发件机构为 Covington & Burling、日期为1999年12月9日的传真页。"
    )
    assert [group.label for group in groups] == [
        "1999年12月9日",
        "Covington & Burling",
    ]
    evidence = evaluate_ocr_literal_groups(
        groups,
        ["COVINGTO N& B URLING", "Date: December 9, 1999"],
    )
    assert evidence["all_constraints_matched"] is True


def test_changed_date_fails_while_shared_organization_matches() -> None:
    groups = extract_ocr_literal_groups(
        "找出发件机构为 Covington & Burling、日期为1999年12月8日的传真页。"
    )
    evidence = evaluate_ocr_literal_groups(
        groups,
        ["COVINGTON & BURLING", "Date: December 9, 1999"],
    )
    assert evidence["matched_constraint_count"] == 1
    assert evidence["all_constraints_matched"] is False


def test_translates_high_precision_ocr_clues() -> None:
    groups = extract_ocr_literal_groups("要求用蓝色墨水和大写字母填写、必填项带星号")
    labels = {group.label for group in groups}
    assert {"蓝色墨水", "大写字母", "必填项"}.issubset(labels)
    assert len(groups) == 3
    evidence = evaluate_ocr_literal_groups(
        groups,
        ["Please use black ink and BLOCK CAPITALS. All fields are mandatory."],
    )
    assert evidence["all_constraints_matched"] is False


def test_passive_and_active_smoking_are_not_conflated() -> None:
    active = extract_ocr_literal_groups("Barnes 与 Bero 对主动吸烟综述")
    evidence = evaluate_ocr_literal_groups(
        active,
        ["Barnes and Bero review articles that exculpate passive smoking"],
    )
    assert evidence["all_constraints_matched"] is False


def test_phone_suffix_is_checked_against_phone_length_numbers() -> None:
    groups = extract_ocr_literal_groups("查找卫念念的登记表，联系电话末位为9")
    assert groups[-1].source == "phone_suffix"
    mismatch = evaluate_ocr_literal_groups(
        groups,
        ["姓名 卫念念", "联系电话 15918555738", "出生日期 1999"],
    )
    assert mismatch["all_constraints_matched"] is False
    match = evaluate_ocr_literal_groups(
        groups,
        ["姓名 卫念念", "联系电话 15918555739"],
    )
    assert match["all_constraints_matched"] is True


def test_extracts_chinese_patient_and_organization_entities() -> None:
    patient = extract_ocr_literal_groups("哪一页记录了患者赵海鹏？")
    organization = extract_ocr_literal_groups("哪些页面出现了实体“汇丰晋信”？")
    assert [group.label for group in patient] == ["赵海鹏"]
    assert [group.label for group in organization] == ["汇丰晋信"]


def test_extracts_alphanumeric_year_and_amount_constraints() -> None:
    groups = extract_ocr_literal_groups(
        "找1959年与5G和 $25,000 REWARD 有关的页面"
    )
    labels = [group.label for group in groups]
    assert "1959年" in labels
    assert "5G" in labels
    assert "$25,000 REWARD" in labels
