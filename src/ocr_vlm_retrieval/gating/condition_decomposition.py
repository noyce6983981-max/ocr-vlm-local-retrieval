"""Deterministic clause-level decomposition for natural compound queries."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable, Mapping
from typing import Any

from ocr_vlm_retrieval.gating.attribute_coverage import (
    AttributePlan,
    AttributeRequirement,
)

PARSER_VERSION = 18
CLAUSE_SEPARATOR = re.compile(r"[，,；;。]+")
CONJUNCTION = re.compile(r"(?:以及|并且|和|与|及)")
OCR_SIGNAL = re.compile(
    r"(?:\d|%|％|写着|写有|显示|填写|标注|印有|文字|字母|号码|编号|"
    r"姓名|国籍|日期|频数|百分比|元|cm|mm|kg)",
    re.IGNORECASE,
)
QUOTED_TERM = re.compile(r"[“\"《「『]([^”\"》」』]+)[”\"》」』]")
CHINESE_COLOR_PHRASE = re.compile(
    r"(?:浅|深|淡|亮|暗)?[红橙黄绿青蓝紫粉灰黑白棕褐金银]{1,3}色"
)
CHINESE_COLOR_SHORTHAND = re.compile(
    r"[红橙黄绿青蓝紫粉灰黑白棕褐金银]{2,3}(?=[\u3400-\u4dbf\u4e00-\u9fff])"
)
V18_RELATION_MARKERS = (
    "位于",
    "前方",
    "后方",
    "上方",
    "下方",
    "左侧",
    "右侧",
    "之间",
    "身后",
    "周围",
    "分布在",
    "分布着",
    "连接",
    "穿过",
    "沿着",
    "航行",
    "飞着",
    "延伸",
    "覆盖",
    "排列",
    "竖起",
    "站在",
    "坐在",
    "悬在",
    "矗立在",
    "放在",
    "装在",
)
COLOR_NOUN_BOUNDARY = re.compile(
    r"(?:位于|处于|站在|坐在|悬在|矗立在|分布在|连接|穿过|沿着|"
    r"航行|飞着|延伸|覆盖|排列|竖起|拥有|带有|同时有|放在|装在|"
    r"上|下|中|呈|为|是|有)"
)


def _normalize(query: str) -> str:
    text = unicodedata.normalize("NFKC", str(query)).strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" ，,；;。！？!?")


def _unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        cleaned = value.strip(" ，,；;。！？!?")
        if cleaned and cleaned not in result:
            result.append(cleaned)
    return result


def _color_aliases(policy: Mapping[str, Any]) -> tuple[str, ...]:
    aliases = {
        str(alias)
        for values in policy.get("color_aliases", {}).values()
        for alias in values
        if str(alias).strip()
    }
    return tuple(sorted(aliases, key=len, reverse=True))


def _contains_color(text: str, aliases: Iterable[str]) -> bool:
    return bool(
        CHINESE_COLOR_PHRASE.search(text) or CHINESE_COLOR_SHORTHAND.search(text)
    ) or any(alias in text for alias in aliases)


def _relation_markers(policy: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                *(str(value) for value in policy.get("relation_markers", []) if value),
                *V18_RELATION_MARKERS,
            },
            key=len,
            reverse=True,
        )
    )


def _color_binding_values(text: str, aliases: Iterable[str]) -> list[str]:
    values: list[str] = []
    occupied: list[tuple[int, int]] = []
    color_spans: list[tuple[int, int, str]] = []
    chinese_matches = [
        *CHINESE_COLOR_PHRASE.finditer(text),
        *CHINESE_COLOR_SHORTHAND.finditer(text),
    ]
    ordered_matches = sorted(
        chinese_matches,
        key=lambda row: (row.start(), -len(row.group())),
    )
    for match in ordered_matches:
        if any(
            match.start() >= left and match.end() <= right for left, right in occupied
        ):
            continue
        color_spans.append((match.start(), match.end(), match.group()))
        occupied.append((match.start(), match.end()))
    for alias in aliases:
        start = 0
        while True:
            position = text.find(alias, start)
            if position < 0:
                break
            end = position + len(alias)
            if not any(position >= left and end <= right for left, right in occupied):
                color_spans.append((position, end, alias))
                occupied.append((position, end))
            start = position + len(alias)
    for _position, end, color in sorted(color_spans):
        suffix = text[end:]
        suffix = COLOR_NOUN_BOUNDARY.split(suffix, maxsplit=1)[0]
        suffix = re.split(r"[的、和与及\s]", suffix, maxsplit=1)[0]
        suffix = suffix[:8]
        if suffix:
            value = f"{color}{suffix}"
        else:
            prefix = re.split(r"[、,，;；]", text[:_position])[-1]
            prefix = re.sub(r"(?:为|呈|是|有)$", "", prefix).strip()
            prefix = prefix[-10:]
            value = f"{prefix}{color}" if prefix else color
        if value not in values:
            values.append(value)
    return values


def _trailing_noun_phrase(text: str, aliases: Iterable[str]) -> str:
    """Return a short noun suffix after the final explicit color alias."""

    positions = [
        (text.rfind(alias), alias) for alias in aliases if text.rfind(alias) >= 0
    ]
    if not positions:
        return ""
    position, alias = max(positions)
    suffix = text[position + len(alias) :].lstrip("的")
    suffix = re.split(
        r"(?:上|下|中|里|旁|前|后|内|外|处|呈|为|是)", suffix, maxsplit=1
    )[0]
    return suffix[:8]


def _atomic_segments(query: str, policy: Mapping[str, Any]) -> list[str]:
    aliases = _color_aliases(policy)
    relation_markers = _relation_markers(policy)
    atoms: list[str] = []
    for clause in CLAUSE_SEPARATOR.split(query):
        clause = clause.strip()
        if not clause:
            continue
        has_between_relation = "之间" in clause or any(
            marker in clause and marker in {"between", "among"}
            for marker in relation_markers
        )
        shared_postfix_color = bool(
            re.search(
                r"(?:和|与|及).*(?:为|呈|是)(?:浅|深|淡|亮|暗)?"
                r"[红橙黄绿青蓝紫粉灰黑白棕褐金银]{1,3}色$",
                clause,
            )
        )
        pieces = (
            [clause]
            if has_between_relation or shared_postfix_color
            else CONJUNCTION.split(clause)
        )
        pieces = [piece.strip() for piece in pieces if piece.strip()]
        for index, piece in enumerate(pieces):
            if not _contains_color(piece, aliases):
                atoms.append(piece)
                continue
            own_suffix = _trailing_noun_phrase(piece, aliases)
            suffix = own_suffix
            if not suffix and index + 1 < len(pieces):
                suffix = _trailing_noun_phrase(pieces[index + 1], aliases)
            if suffix and not own_suffix:
                atoms.append(
                    f"{piece}{suffix}" if not piece.endswith(suffix) else piece
                )
            else:
                atoms.append(piece)
    return _unique(atoms)


def _rows_for_segment(
    segment: str,
    *,
    policy: Mapping[str, Any],
    color_aliases: Iterable[str],
) -> list[tuple[str, str]]:
    if OCR_SIGNAL.search(segment) or QUOTED_TERM.search(segment):
        return [("ocr", segment)]
    rows: list[tuple[str, str]] = []
    color_values = _color_binding_values(segment, color_aliases)
    rows.extend(("binding", value) for value in color_values)
    has_relation = any(marker in segment for marker in _relation_markers(policy))
    side_color_assignment = len(color_values) >= 2 and any(
        marker in segment for marker in ("左侧", "右侧", "左边", "右边")
    )
    if has_relation and not side_color_assignment:
        relation_value = segment
        for alias in color_aliases:
            relation_value = relation_value.replace(alias, "")
        rows.append(("relation", relation_value))
    rows.extend(
        ("scene", str(scene))
        for scene in policy.get("scene_terms", [])
        if str(scene) and str(scene) in segment
    )
    return rows or [("object", segment)]


def _prompt(kind: str, value: str) -> str:
    if kind == "ocr":
        return f"候选页面中的可见文字或数字完整支持这一条件：{value}。"
    if kind == "relation":
        return f"候选图像清楚显示这一动作或空间关系：{value}。"
    if kind == "binding":
        return f"候选图像中的同一对象完整满足这一属性绑定：{value}。"
    if kind == "scene":
        return f"候选图像的整体场景清楚满足：{value}。"
    return f"候选图像中清楚可见并可辨认这一对象条件：{value}。"


def _requirement(
    kind: str,
    value: str,
    policy: Mapping[str, Any],
    ordinal: int,
) -> AttributeRequirement:
    definition = policy["thresholds"][kind]
    suffix = hashlib.sha256(f"{kind}\n{value}\n{ordinal}".encode()).hexdigest()[:8]
    return AttributeRequirement(
        requirement_id=f"condition_{kind}_{ordinal}_{suffix}",
        kind=kind,
        value=value,
        prompt=_prompt(kind, value),
        threshold=float(definition["minimum"]),
        full_score=float(definition["full_score"]),
        weight=float(definition.get("weight", 1.0)),
        mandatory=bool(definition.get("mandatory", True)),
    )


def _fingerprint(query: str, requirements: Iterable[AttributeRequirement]) -> str:
    material = {
        "query": query,
        "parser_version": PARSER_VERSION,
        "requirements": [row.to_dict() for row in requirements],
    }
    return hashlib.sha256(
        json.dumps(
            material,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def decompose_condition_query(
    query: str,
    policy: Mapping[str, Any],
) -> AttributePlan:
    """Split one natural query without using pair labels or study metadata."""

    normalized = _normalize(query)
    aliases = _color_aliases(policy)
    rows = [
        row
        for segment in _atomic_segments(normalized, policy)
        for row in _rows_for_segment(
            segment,
            policy=policy,
            color_aliases=aliases,
        )
    ]
    max_requirements = int(policy.get("max_requirements", 8))
    requirements = tuple(
        _requirement(kind, value, policy, ordinal)
        for ordinal, (kind, value) in enumerate(rows[:max_requirements], start=1)
    )
    mandatory_count = sum(row.mandatory for row in requirements)
    return AttributePlan(
        query=normalized,
        requirements=requirements,
        compositional=mandatory_count >= 2,
        parser_version=PARSER_VERSION,
        fingerprint=_fingerprint(normalized, requirements),
    )
