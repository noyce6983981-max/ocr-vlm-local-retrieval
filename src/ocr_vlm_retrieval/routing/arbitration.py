"""Deterministic safety arbitration for local-LLM route proposals."""

from __future__ import annotations

from ocr_vlm_retrieval.routing.schema import Route

_EXPLICIT_LITERAL_MARKERS = (
    "写着",
    "写有",
    "提到",
    "文字",
    "标题",
    "标签",
    "号码",
    "编号",
    "型号",
    "姓名",
    "国籍",
    "学校",
    "日期",
    "金额",
    "总额",
    "标价",
    "表格",
    "单元格",
    "正文",
    "通知",
    "发票",
    "小票",
)


def has_explicit_literal_requirement(query: str) -> bool:
    """Return whether candidate-page text is an explicit necessary condition."""

    return any(marker in query for marker in _EXPLICIT_LITERAL_MARKERS) or any(
        marker in query for marker in "“”「」"
    )


def guard_llm_route(
    query: str,
    *,
    rule_route: Route,
    llm_route: Route,
) -> tuple[Route, str | None]:
    """Prevent narrow LLM routes from discarding required retrieval branches."""

    lacks_literal_requirement = not has_explicit_literal_requirement(query)
    if lacks_literal_requirement and (
        (
            rule_route == "visual_discovery"
            and llm_route in {"text_evidence", "mixed"}
        )
        or (rule_route == "mixed" and llm_route == "text_evidence")
    ):
        return rule_route, "preserve_visual_route_without_literal_requirement"

    compact = "".join(query.split()).strip("，。！？?：:；;、\"'“”‘’")
    if llm_route == "entity_exact" and len(compact) > 16:
        return rule_route, "reject_entity_route_for_long_compound_query"
    return llm_route, None
