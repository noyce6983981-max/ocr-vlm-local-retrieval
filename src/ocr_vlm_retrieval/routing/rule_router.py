"""Compatibility adapter for the existing deterministic rule router."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from types import ModuleType

from ocr_vlm_retrieval.routing.schema import Route, validate_route

RuleClassifier = Callable[[str], str]

_LITERAL_EVIDENCE_MARKERS = (
    "写着",
    "写有",
    "提到",
    "显示",
    "标题",
    "说明",
    "标签",
    "号码",
    "型号",
    "标注",
    "文字",
    "正文",
    "包含",
    "字样",
)
_VISUAL_APPEARANCE_MARKERS = (
    "照片",
    "画面",
    "红色",
    "蓝色",
    "绿色",
    "白色",
    "黄色",
    "金色",
    "穿",
    "外墙",
    "徽章",
    "印章",
    "标志",
    "球衣",
    "瓶子",
    "路牌",
    "胸牌",
)
_LAYOUT_STRUCTURE_MARKERS = (
    "左边",
    "右边",
    "左侧",
    "右侧",
    "左上角",
    "右上角",
    "下方",
    "顶部",
    "底部",
    "申请表",
    "清单",
    "界面",
    "截图",
    "柱状图",
    "折线图",
    "时间轴",
    "地图",
    "节点",
    "证书",
)
_TEXT_LOCATOR_ONLY_MARKERS = ("标题写着", "说明文字")
_VISUAL_REQUEST_MARKERS = (
    "照片",
    "图片",
    "画面",
    "图中",
    "哪张图",
    "一张图",
    "的图",
)
_LOOKUP_MARKERS = ("查", "找", "定位", "检索", "有没有", "在哪")
_EXACT_IDENTIFIER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])[A-Za-z]{1,8}[-_]?[0-9]{1,8}(?![A-Za-z0-9])"
)


@dataclass(frozen=True, slots=True)
class RuleDecision:
    """Auditable output of deterministic routing."""

    route: Route
    ambiguous: bool
    reason_codes: tuple[str, ...]


def default_ambiguity_reasons(query: str, route: Route) -> tuple[str, ...]:
    """Return conservative reasons for invoking the pilot LLM router."""

    del query
    if route == "mixed":
        return ("rule_route_mixed",)
    return ()


def _looks_like_missed_person_lookup(query: str, module: ModuleType) -> bool:
    normalized = "".join(query.split()).strip("，。！？?：:；;、\"'“”‘’")
    if not normalized:
        return False
    person_prefixes = (
        "查",
        "查一下",
        "查到",
        "找",
        "定位",
        "检索",
        "有没有",
        "看看",
    )
    person_suffixes = (
        "吗",
        "出现",
        "在哪",
        "对应",
        "的记录",
        "有没有",
        "被提到",
    )
    if len(normalized) > 4 and not any(
        marker in normalized
        for marker in (*person_prefixes, *person_suffixes, "名单")
    ):
        return False

    single_surnames: frozenset[str] = frozenset(
        getattr(module, "COMMON_SINGLE_SURNAMES", frozenset())
    )
    compound_surnames: tuple[str, ...] = tuple(
        getattr(module, "COMMON_COMPOUND_SURNAMES", ())
    )
    lexical_decision = getattr(module, "lexical_person_name_decision", None)
    non_person_suffixes: tuple[str, ...] = tuple(
        getattr(module, "NON_PERSON_TERM_SUFFIXES", ())
    )
    if not callable(lexical_decision):
        return False

    candidates: list[tuple[str, str, str]] = []
    for index, character in enumerate(normalized):
        compound = next(
            (
                surname
                for surname in compound_surnames
                if normalized.startswith(surname, index)
            ),
            None,
        )
        surname_length = len(compound) if compound else 1
        if compound is None and character not in single_surnames:
            continue
        for given_name_length in (1, 2):
            end = index + surname_length + given_name_length
            if end <= len(normalized):
                candidates.append(
                    (normalized[index:end], normalized[:index], normalized[end:])
                )

    for candidate, before, after in candidates:
        prefix_matches = not before or any(
            before.endswith(prefix) for prefix in person_prefixes
        )
        suffix_matches = not after or any(
            after.startswith(suffix) for suffix in person_suffixes
        )
        if not prefix_matches or not suffix_matches:
            continue
        if candidate.endswith(non_person_suffixes):
            continue
        decision = lexical_decision(candidate)
        if decision is True:
            return True
        if decision is None:
            return True
    return False


def _has_explicit_compound_evidence(query: str) -> bool:
    """Detect queries that explicitly require at least two evidence families."""

    literal = any(marker in query for marker in _LITERAL_EVIDENCE_MARKERS)
    literal = literal or any(marker in query for marker in "“”「」")
    literal = literal or any(
        character.isascii() and character.isalnum() for character in query
    )
    visual = any(marker in query for marker in _VISUAL_APPEARANCE_MARKERS)
    layout = any(marker in query for marker in _LAYOUT_STRUCTURE_MARKERS)
    if (
        literal
        and layout
        and not visual
        and any(marker in query for marker in _TEXT_LOCATOR_ONLY_MARKERS)
    ):
        return False
    return sum((literal, visual, layout)) >= 2


def _looks_like_pure_visual_scene_query(query: str, route: Route) -> bool:
    """Recover natural image requests that the legacy router labels mixed."""

    if route != "mixed":
        return False
    literal = any(marker in query for marker in _LITERAL_EVIDENCE_MARKERS)
    literal = literal or any(marker in query for marker in "“”「」")
    literal = literal or any(
        character.isascii() and character.isalnum() for character in query
    )
    layout = any(marker in query for marker in _LAYOUT_STRUCTURE_MARKERS)
    visual_request = any(marker in query for marker in _VISUAL_REQUEST_MARKERS)
    return visual_request and not literal and not layout


def _looks_like_short_exact_identifier_lookup(query: str, route: Route) -> bool:
    """Recover short product/code lookups without capturing compound scenes."""

    compact = "".join(query.split()).strip("，。！？?：:；;、\"'“”‘’")
    if route != "mixed" or len(compact) > 18:
        return False
    if not _EXACT_IDENTIFIER_PATTERN.search(compact):
        return False
    if any(marker in query for marker in _VISUAL_APPEARANCE_MARKERS):
        return False
    if any(marker in query for marker in _LAYOUT_STRUCTURE_MARKERS):
        return False
    return any(marker in query for marker in _LOOKUP_MARKERS)


def legacy_ambiguity_reasons(
    module: ModuleType,
) -> Callable[[str, Route], tuple[str, ...]]:
    """Build a conservative call gate using the frozen legacy vocabulary."""

    def detect(query: str, route: Route) -> tuple[str, ...]:
        explicit_compound = _has_explicit_compound_evidence(query)
        reasons: list[str] = []
        if route == "mixed" and not explicit_compound:
            reasons.append("rule_route_mixed")
        elif route != "mixed" and explicit_compound:
            reasons.append("explicit_compound_evidence_missed_by_rule")
        eligible_route = route in {"text_evidence", "topic_discovery", "mixed"}
        if eligible_route and _looks_like_missed_person_lookup(query, module):
            reasons.append("probable_person_lookup_missed_by_rule")
        return tuple(reasons)

    return detect


class RuleRouter:
    """Wrap a classifier without changing the frozen legacy implementation."""

    def __init__(
        self,
        classifier: RuleClassifier,
        ambiguity_detector: Callable[[str, Route], tuple[str, ...]] = (
            default_ambiguity_reasons
        ),
    ) -> None:
        self._classifier = classifier
        self._ambiguity_detector = ambiguity_detector

    @classmethod
    def legacy(cls) -> RuleRouter:
        """Load ``scripts.query_routing`` lazily as a transition adapter."""

        module = import_module("scripts.query_routing")
        classifier = getattr(module, "classify_query_form", None)
        if not callable(classifier):
            raise RuntimeError("Legacy query classifier is unavailable")
        return cls(classifier, legacy_ambiguity_reasons(module))

    @classmethod
    def v19_calibrated(cls) -> RuleRouter:
        """Add two calibration-derived high-confidence recoveries to legacy."""

        module = import_module("scripts.query_routing")
        classifier = getattr(module, "classify_query_form", None)
        if not callable(classifier):
            raise RuntimeError("Legacy query classifier is unavailable")

        def calibrated_classifier(query: str) -> str:
            route = validate_route(classifier(query))
            compact = "".join(query.split()).strip(
                "，。！？?：:；;、\"'“”‘’"
            )
            if (
                route in {"text_evidence", "topic_discovery", "mixed"}
                and len(compact) <= 18
                and _looks_like_missed_person_lookup(query, module)
            ):
                return "entity_exact"
            if _looks_like_short_exact_identifier_lookup(query, route):
                return "entity_exact"
            if _looks_like_pure_visual_scene_query(query, route):
                return "visual_discovery"
            return route

        return cls(calibrated_classifier, legacy_ambiguity_reasons(module))

    def route(self, query: str) -> RuleDecision:
        """Classify one non-empty query and annotate ambiguity."""

        normalized = " ".join(query.split())
        if not normalized:
            raise ValueError("query must not be empty")
        route = validate_route(self._classifier(normalized))
        reasons = self._ambiguity_detector(normalized, route)
        return RuleDecision(
            route=route,
            ambiguous=bool(reasons),
            reason_codes=reasons,
        )
