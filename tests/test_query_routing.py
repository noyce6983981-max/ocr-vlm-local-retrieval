from __future__ import annotations

import csv
from pathlib import Path

from scripts.query_routing import (
    classify_query_form,
    detect_color_intent,
    extract_strict_entity_term,
    expand_visual_query,
    extract_temporal_personal_terms,
    extract_topic_evidence_terms,
    extract_quoted_terms,
    infer_retrieval_route,
    is_pure_color_query,
    is_keyword_discovery_query,
    is_visual_discovery_query,
)


def test_explicit_scene_text_uses_text_evidence() -> None:
    assert (
        infer_retrieval_route(
            "哪张场景照片中出现“惠图仿石艺术漆”和“品质浇筑美好”？"
        )
        == "text_evidence"
    )


def test_natural_image_title_uses_visual_metadata() -> None:
    assert (
        infer_retrieval_route(
            "查找一张展示“A Corner of Hongqiao Park”的自然图像。"
        )
        == "visual_metadata"
    )


def test_general_semantic_query_is_topic_discovery() -> None:
    assert (
        infer_retrieval_route("找一下介绍深度学习方法的那一页")
        == "topic_discovery"
    )


def test_short_subjective_query_uses_visual_discovery() -> None:
    assert is_visual_discovery_query("美丽") is True
    assert is_visual_discovery_query("森林") is True
    assert infer_retrieval_route("美丽") == "visual_discovery"
    assert infer_retrieval_route("森林") == "visual_discovery"


def test_visual_topics_are_not_misclassified_as_person_names() -> None:
    for query in ("山水", "山林", "山峰", "山水画"):
        assert extract_strict_entity_term(query) is None
        assert is_visual_discovery_query(query) is True
        assert infer_retrieval_route(query) == "visual_discovery"


def test_short_person_name_requires_text_evidence() -> None:
    assert extract_strict_entity_term("方岩松") == "方岩松"
    assert extract_strict_entity_term("查找方岩松的资料") == "方岩松"
    assert is_visual_discovery_query("方岩松") is False
    assert infer_retrieval_route("方岩松") == "entity_exact"


def test_unknown_short_term_is_not_automatically_visual() -> None:
    assert is_visual_discovery_query("量子力学") is False
    assert is_keyword_discovery_query("量子力学") is True
    assert infer_retrieval_route("量子力学") == "topic_discovery"


def test_person_name_is_not_keyword_discovery() -> None:
    assert extract_strict_entity_term("盛和") == "盛和"
    assert is_keyword_discovery_query("盛和") is False


def test_lexical_analysis_prevents_surname_prefix_false_positives() -> None:
    for query in ("方程式", "林业", "马达", "高等数学"):
        assert extract_strict_entity_term(query) is None
        assert classify_query_form(query) == "topic_discovery"


def test_query_form_matrix() -> None:
    expected = {
        "山水": "visual_discovery",
        "蓝天白云": "visual_discovery",
        "海边日落": "visual_discovery",
        "盛和": "entity_exact",
        "方岩松": "entity_exact",
        "张三": "entity_exact",
        "量子力学": "topic_discovery",
        "深度学习": "topic_discovery",
        "神经网络": "topic_discovery",
        "哪份文档包含课程成绩？": "text_evidence",
        "如何设计一个检索系统？": "mixed",
    }
    assert {
        query: classify_query_form(query) for query in expected
    } == expected


def test_versioned_query_intent_regression_set() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "data/evaluation/search_intent_regression_v12.csv"
    )
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) >= 30
    errors = {
        row["query"]: (
            row["expected_route"],
            classify_query_form(row["query"]),
        )
        for row in rows
        if classify_query_form(row["query"])
        != row["expected_route"]
    }
    assert errors == {}


def test_fact_question_is_not_keyword_discovery() -> None:
    query = "哪份文档介绍量子力学？"
    assert is_keyword_discovery_query(query) is False


def test_explicit_fact_query_is_not_visual_discovery() -> None:
    assert (
        is_visual_discovery_query("哪份文档提到了火星探测器能源消耗？")
        is False
    )


def test_explicit_person_context_is_exact_even_for_unknown_name() -> None:
    assert extract_strict_entity_term("找一个名为方岩松的人") == "方岩松"
    assert classify_query_form("姓名是方岩松") == "entity_exact"


def test_visual_form_words_route_unseen_picture_queries() -> None:
    assert classify_query_form("找一张红色跑车照片") == "visual_discovery"


def test_technical_context_overrides_visual_substrings() -> None:
    for query in ("自然语言处理", "建筑学", "动物学", "车辆工程"):
        assert classify_query_form(query) == "topic_discovery"


def test_color_and_broad_scene_queries_use_visual_evidence() -> None:
    assert detect_color_intent("蓝") == "blue"
    assert detect_color_intent("找蓝色汽车") == "blue"
    assert is_pure_color_query("找一张蓝色图片") is True
    assert is_pure_color_query("找蓝色汽车") is False
    assert classify_query_form("蓝") == "visual_discovery"
    assert classify_query_form("天地") == "visual_discovery"
    assert "蓝色" in expand_visual_query("蓝")
    assert "蓝色汽车" in expand_visual_query("蓝色汽车")
    assert "天空与大地" in expand_visual_query("天地")


def test_color_characters_do_not_match_unrelated_words() -> None:
    assert detect_color_intent("蓝牙") is None
    assert detect_color_intent("白血病") is None
    assert classify_query_form("蓝牙") == "topic_discovery"
    assert classify_query_form("白血病") == "topic_discovery"


def test_visual_query_expansion_adds_concrete_semantics() -> None:
    assert "自然风景" in expand_visual_query("美丽")
    assert "绿色森林" in expand_visual_query("森林")
    explicit = "哪份文档提到了火星探测器能源消耗？"
    assert expand_visual_query(explicit) == explicit


def test_visual_structure_is_not_misrouted_as_plain_topic_or_text() -> None:
    expected = {
        "找带印章的正式文件": "visual_metadata",
        "查找有柱状图的汇报幻灯片": "visual_metadata",
        "找有三栏排版的幻灯片": "visual_metadata",
        "找带搜索框的手机界面": "visual_metadata",
        "找包含错误提示弹窗的截图": "visual_metadata",
        "找包含流程图的PPT页面": "visual_metadata",
        "找包含英文广告牌的街景照片": "mixed",
        "查找会议室白板上写着明天计划的照片": "mixed",
        "找车库里蓝色自行车的维修记录": "mixed",
    }
    assert {
        query: classify_query_form(query) for query in expected
    } == expected


def test_quoted_scene_text_stays_on_text_evidence_route() -> None:
    assert (
        classify_query_form(
            "哪张场景照片中出现“惠图仿石艺术漆”和“品质浇筑美好”？"
        )
        == "text_evidence"
    )


def test_topic_evidence_terms_remove_request_boilerplate() -> None:
    assert extract_topic_evidence_terms("查找标题里有年度报告的文档") == [
        "年度报告"
    ]
    reference_terms = extract_topic_evidence_terms(
        "找带参考文献列表的学术资料"
    )
    assert "参考文献列表" in reference_terms
    assert "参考文献" in reference_terms


def test_relative_time_constraints_are_explicit() -> None:
    assert extract_temporal_personal_terms("找我去年在海边拍的日落照片")
    assert extract_temporal_personal_terms("找昨晚收到的通知截图") == [
        "昨晚"
    ]
    assert extract_temporal_personal_terms("找蓝天下的建筑照片") == []


def test_quoted_terms_capture_literal_ocr_or_metadata_evidence() -> None:
    assert extract_quoted_terms("查找展示“A Corner of Hongqiao Park”的图片") == [
        "A Corner of Hongqiao Park"
    ]
    assert extract_quoted_terms('查找包含"Primary Date"的文档') == [
        "Primary Date"
    ]
