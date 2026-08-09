from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from ocr_vlm_retrieval.gating.condition_decomposition import (
    decompose_condition_query,
)


def _policy() -> dict[str, object]:
    path = Path(__file__).resolve().parents[1] / "config/v17_attribute_coverage.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_ocr_compound_query_is_split_into_independent_conditions() -> None:
    plan = decompose_condition_query(
        "一张居民家庭收入统计题的表格照片，1000至1200元分组的频数为18，百分比为45%。",
        _policy(),
    )
    assert plan.compositional is True
    assert [row.kind for row in plan.requirements] == ["object", "ocr", "ocr"]
    assert [row.value for row in plan.requirements[1:]] == [
        "1000至1200元分组的频数为18",
        "百分比为45%",
    ]


def test_color_bindings_are_not_collapsed_into_one_global_condition() -> None:
    plan = decompose_condition_query(
        "一只玩具蝴蝶有橙红色翅膀和蓝绿色身体，翅膀上同时有黄色与紫色斑块。",
        _policy(),
    )
    values = [row.value for row in plan.requirements]
    assert plan.compositional is True
    assert any("橙红色翅膀" in value for value in values)
    assert any("蓝绿色身体" in value for value in values)
    assert any("黄色" in value for value in values)
    assert any("紫色斑块" in value for value in values)


def test_spatial_relation_is_a_mandatory_requirement() -> None:
    plan = decompose_condition_query(
        "一片绿色农田位于高大山脉的前方，村庄分布在农田与山脉之间。",
        _policy(),
    )
    relation_values = [
        row.value for row in plan.requirements if row.kind == "relation"
    ]
    assert plan.compositional is True
    assert any("前方" in value for value in relation_values)
    assert any("之间" in value for value in relation_values)


def test_parser_api_does_not_accept_pair_or_label_metadata() -> None:
    parameters = decompose_condition_query.__annotations__
    assert set(parameters) == {"query", "policy", "return"}


def test_question_mark_inside_quoted_ocr_text_is_not_split() -> None:
    plan = decompose_condition_query(
        "米黄色页面顶部写着“What is an Endowed Chair?”，"
        "下方是带扶手的黑白古典椅子插图。",
        _policy(),
    )
    values = [row.value for row in plan.requirements]
    assert "米黄色页面顶部写着“What is an Endowed Chair?”" in values
    assert "黑白古典椅子插图" in values
    assert "”" not in values


def test_side_specific_colors_keep_their_bindings() -> None:
    plan = decompose_condition_query(
        "白色背景上的卡通雨伞，伞面左侧为红色、右侧为黄色，并有蓝色边框。",
        _policy(),
    )
    values = [row.value for row in plan.requirements]
    assert "伞面左侧红色" in values
    assert "右侧黄色" in values


def test_shared_postfix_color_keeps_both_bound_objects() -> None:
    plan = decompose_condition_query(
        "草地上的羊驼身体覆盖灰色卷毛，脸和长颈为白色。",
        _policy(),
    )
    values = [row.value for row in plan.requirements]
    assert "灰色卷毛" in values
    assert "脸和长颈白色" in values
