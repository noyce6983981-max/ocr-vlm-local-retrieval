"""Attribute-level evidence for compositional visual retrieval.

The V16 product gate uses one global image/query similarity for compound visual
queries.  A high-scoring attribute can therefore hide a missing mandatory
attribute.  This module keeps the decomposition and aggregation independent of
model code so that V17 can compare global similarity against conservative
attribute coverage without changing the frozen V16 policy.

This is *attribute-level multi-query interaction*, not patch/token-level visual
late interaction.  Each visual requirement is encoded separately, while the
existing document image embedding remains precomputed as one vector.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

VISUAL_REQUIREMENT_KINDS = frozenset({"object", "scene", "relation", "binding"})


@dataclass(frozen=True)
class AttributeRequirement:
    """One independently testable condition in a visual query."""

    requirement_id: str
    kind: str
    value: str
    prompt: str
    threshold: float
    full_score: float
    weight: float = 1.0
    mandatory: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AttributePlan:
    """Deterministic query decomposition used for scoring and auditing."""

    query: str
    requirements: tuple[AttributeRequirement, ...]
    compositional: bool
    parser_version: int
    fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "requirements": [row.to_dict() for row in self.requirements],
            "compositional": self.compositional,
            "parser_version": self.parser_version,
            "fingerprint": self.fingerprint,
        }


def load_attribute_policy(path: Path) -> dict[str, Any]:
    """Load and minimally validate one V17 attribute policy."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Attribute policy must be a JSON object.")
    if int(payload.get("version", 0)) < 1:
        raise ValueError("Attribute policy version must be positive.")
    thresholds = payload.get("thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError("Attribute policy requires threshold definitions.")
    for kind in ("object", "scene", "relation", "binding", "color", "ocr"):
        definition = thresholds.get(kind)
        if not isinstance(definition, dict):
            raise ValueError(f"Missing threshold definition for {kind!r}.")
        threshold = float(definition.get("minimum", -1.0))
        full_score = float(definition.get("full_score", -1.0))
        if full_score <= threshold:
            raise ValueError(f"full_score must exceed minimum for {kind!r}.")
    return payload


def _normalize_query(query: str) -> str:
    normalized = unicodedata.normalize("NFKC", query).strip().casefold()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(
        r"(?<=[\u3400-\u4dbf\u4e00-\u9fff]) "
        r"(?=[\u3400-\u4dbf\u4e00-\u9fff])",
        "",
        normalized,
    )
    return normalized.strip("，。！？?；;：:,.! ")


def _canonicalize_english_question(query: str) -> str:
    """Turn common visual-QA answer slots into observable image evidence.

    Question prefixes such as ``what letter is`` describe the answer type; they
    are not themselves objects visible in an image. Keeping the prefix in an
    object requirement produced prompts such as "an image containing *what
    kind of zone is*". These deterministic rewrites retain the requested
    evidence and spatial relation while removing that grammatical artifact.
    """

    if not re.search(r"[a-z]", query):
        return query
    text = re.sub(r"^what's\b", "what is", query).strip()

    word_with_color = re.fullmatch(r"what word is written in (\w+) (on .+)", text)
    if word_with_color:
        color, relation = word_with_color.groups()
        return f"{color} visible word {relation}"

    does_say = re.fullmatch(r"what does (.+?) say", text)
    if does_say:
        subject = does_say.group(1).strip()
        matched_relation = _relation_match(
            subject,
            (
                "on the back of",
                "on the front of",
                "on the left",
                "on the right",
                "in front of",
                "under",
                "below",
                "above",
                "behind",
                "on",
            ),
        )
        if matched_relation:
            _, start, _ = matched_relation
            left = subject[:start].strip()
            target = "logo text" if "logo" in left else "visible text"
            return f"{target} {subject[start:]}"
        return f"visible text on {subject}"

    rewrites: tuple[tuple[str, str], ...] = (
        (r"^what is printed (on .+)$", r"printed text \1"),
        (r"^what is written (.+)$", r"visible text \1"),
        (r"^what greeting is written (.+)$", r"greeting text \1"),
        (r"^what letter is (.+)$", r"visible letter \1"),
        (r"^what four numbers are (.+)$", r"four-digit number \1"),
        (r"^what kind of zone is (.+)$", r"zone \1"),
        (r"^what type of parking is (.+)$", r"parking sign \1"),
        (r"^what is the price shown (.+)$", r"price text \1"),
        (r"^what name is displayed (.+)$", r"visible name \1"),
        (r"^what number is (.+)$", r"visible number \1"),
        (
            r"^what city is this patrol car from$",
            r"municipality name on this patrol car",
        ),
        (r"^what cities orchestra played (.+)$", r"orchestra location names \1"),
        (r"^what city name is mentioned (.+)$", r"municipality name \1"),
        (
            r"^what year was the beverage on the right started in$",
            r"founding year on the right beverage",
        ),
        (r"^what is the licence plate number of (.+)$", r"licence plate on \1"),
        (r"^what is on (.+)$", r"visible content on \1"),
    )
    for pattern, replacement in rewrites:
        if re.fullmatch(pattern, text):
            return re.sub(pattern, replacement, text)

    generic = re.fullmatch(r"what is (.+)", text)
    if generic:
        return generic.group(1).strip()
    return text


def _unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        cleaned = value.strip()
        if cleaned and cleaned not in result:
            result.append(cleaned)
    return result


def _extract_quoted_terms(query: str) -> list[str]:
    patterns = (
        r"“([^”]+)”",
        r'"([^"]+)"',
        r"「([^」]+)」",
        r"『([^』]+)』",
    )
    return _unique(
        match
        for pattern in patterns
        for match in re.findall(pattern, query)
        if len(match.strip()) >= 2
    )


def _remove_terms(text: str, terms: Iterable[str]) -> str:
    cleaned = text
    for term in sorted(set(terms), key=len, reverse=True):
        if term:
            if re.search(r"[a-z0-9]", term, re.IGNORECASE):
                cleaned = re.sub(
                    rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])",
                    " ",
                    cleaned,
                    flags=re.IGNORECASE,
                )
            else:
                cleaned = cleaned.replace(term, " ")
    return cleaned


def _contains_term(text: str, term: str) -> bool:
    if re.search(r"[a-z0-9]", term, re.IGNORECASE):
        return bool(
            re.search(
                rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])",
                text,
                flags=re.IGNORECASE,
            )
        )
    return term in text


def _relation_match(text: str, markers: Iterable[str]) -> tuple[str, int, int] | None:
    for marker in sorted(markers, key=len, reverse=True):
        if not marker:
            continue
        if re.search(r"[a-z0-9]", marker, re.IGNORECASE):
            match = re.search(
                rf"(?<![a-z0-9]){re.escape(marker)}(?![a-z0-9])",
                text,
                flags=re.IGNORECASE,
            )
            if match:
                return marker, match.start(), match.end()
        else:
            start = text.find(marker)
            if start >= 0:
                return marker, start, start + len(marker)
    return None


def _binding_value(attribute: str, object_phrase: str) -> str:
    separator = " " if re.search(r"[a-z0-9]", object_phrase) else ""
    return f"{attribute}{separator}{object_phrase}"


def _clean_object_phrase(
    text: str,
    policy: Mapping[str, Any],
    removable_terms: Iterable[str],
) -> str:
    cleaned = _remove_terms(text, removable_terms)
    cleaned = _remove_terms(cleaned, policy.get("query_noise_terms", []))
    cleaned = _remove_terms(cleaned, policy.get("visual_container_terms", []))
    cleaned = cleaned.strip()
    cleaned = re.sub(r"^[一二两三四五六七八九十0-9]+[张个只幅页份]?", "", cleaned)
    cleaned = re.sub(r"(?:的|地|得)$", "", cleaned)
    cleaned = re.sub(r"[，。！？?、；;：:（）()\[\]{},.!]+", " ", cleaned)
    cleaned = " ".join(cleaned.split())
    return cleaned if len(cleaned) >= 1 else ""


def _requirement_prompt(kind: str, value: str) -> str:
    if kind == "object":
        return f"画面中清楚可见{value}，且它是可辨认的主体或重要对象。"
    if kind == "scene":
        return f"画面的整体场景明确是{value}。"
    if kind == "relation":
        return f"画面中清楚显示{value}这一动作或空间关系。"
    if kind == "binding":
        return f"画面中同一个对象明确满足“{value}”的属性绑定。"
    if kind == "color":
        return f"画面具有明确的{value}视觉颜色证据。"
    if kind == "ocr":
        return f"同一候选的 OCR 文字完整包含“{value}”。"
    raise ValueError(f"Unsupported requirement kind: {kind}")


def _make_requirement(
    kind: str,
    value: str,
    policy: Mapping[str, Any],
    ordinal: int,
) -> AttributeRequirement:
    definition = policy["thresholds"][kind]
    material = f"{kind}\n{value}\n{ordinal}".encode()
    suffix = hashlib.sha256(material).hexdigest()[:8]
    return AttributeRequirement(
        requirement_id=f"{kind}_{ordinal}_{suffix}",
        kind=kind,
        value=value,
        prompt=_requirement_prompt(kind, value),
        threshold=float(definition["minimum"]),
        full_score=float(definition["full_score"]),
        weight=float(definition.get("weight", 1.0)),
        mandatory=bool(definition.get("mandatory", True)),
    )


def _plan_fingerprint(
    query: str,
    requirements: Iterable[AttributeRequirement],
    parser_version: int,
) -> str:
    payload = {
        "query": query,
        "parser_version": parser_version,
        "requirements": [row.to_dict() for row in requirements],
    }
    material = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def decompose_visual_query(
    query: str,
    policy: Mapping[str, Any],
) -> AttributePlan:
    """Split a Chinese or English query into auditable requirements.

    The parser is intentionally deterministic.  It handles explicit colors,
    configured scenes, strong relation markers, quoted OCR strings, and the
    remaining object phrases.  A later learned parser can be compared against
    this transparent baseline without changing the evaluation protocol.
    """

    normalized = _canonicalize_english_question(_normalize_query(query))
    parser_version = int(policy.get("parser_version", 1))
    quoted_terms = _extract_quoted_terms(normalized)
    visual_text = normalized
    for value in quoted_terms:
        visual_text = visual_text.replace(value, " ")
    visual_text = re.sub(r"[“”\"「」『』]", " ", visual_text)

    color_aliases: Mapping[str, list[str]] = policy.get("color_aliases", {})
    color_hits: list[tuple[str, str]] = []
    for canonical, aliases in color_aliases.items():
        for alias in sorted(aliases, key=len, reverse=True):
            if alias and _contains_term(visual_text, alias):
                color_hits.append((canonical, alias))
                break
    scene_terms = [
        term
        for term in policy.get("scene_terms", [])
        if _contains_term(visual_text, term)
    ]
    matched_relation = _relation_match(
        visual_text,
        policy.get("relation_markers", []),
    )
    relation_marker = matched_relation[0] if matched_relation else None

    removable = [alias for _, alias in color_hits] + scene_terms
    object_phrases: list[str] = []
    relation_value = ""
    binding_values: list[str] = []

    if relation_marker and matched_relation:
        _, marker_start, marker_end = matched_relation
        left = visual_text[:marker_start]
        right = visual_text[marker_end:]
        left_object = _clean_object_phrase(left, policy, removable)
        right_object = _clean_object_phrase(right, policy, removable)
        object_phrases = _unique((left_object, right_object))
        if left_object and right_object:
            separator = " " if re.search(r"[a-z]", relation_marker) else ""
            relation_value = (
                f"{left_object}{separator}{relation_marker}{separator}{right_object}"
            )
        elif left_object or right_object:
            relation_value = _clean_object_phrase(visual_text, policy, removable)
        for canonical, alias in color_hits:
            if _contains_term(left, alias) and left_object:
                binding_values.append(_binding_value(canonical, left_object))
            elif _contains_term(right, alias) and right_object:
                binding_values.append(_binding_value(canonical, right_object))
    else:
        residual = _remove_terms(visual_text, removable)
        pieces = re.split(r"(?:以及|并且|同时|和|与|、)", residual)
        object_phrases = _unique(
            _clean_object_phrase(piece, policy, ()) for piece in pieces
        )
        if len(object_phrases) == 1:
            for canonical, _ in color_hits:
                binding_values.append(_binding_value(canonical, object_phrases[0]))

    rows: list[tuple[str, str]] = []
    rows.extend(("color", canonical) for canonical, _ in color_hits)
    rows.extend(("scene", value) for value in _unique(scene_terms))
    rows.extend(("object", value) for value in object_phrases)
    rows.extend(("binding", value) for value in _unique(binding_values))
    if relation_value:
        rows.append(("relation", relation_value))
    rows.extend(("ocr", value) for value in quoted_terms)

    max_requirements = int(policy.get("max_requirements", 8))
    requirements = tuple(
        _make_requirement(kind, value, policy, ordinal)
        for ordinal, (kind, value) in enumerate(rows[:max_requirements], 1)
    )
    mandatory_visual_count = sum(
        row.mandatory for row in requirements if row.kind != "ocr"
    )
    compositional = bool(
        relation_value
        or mandatory_visual_count
        >= int(policy.get("minimum_compositional_requirements", 2))
    )
    fingerprint = _plan_fingerprint(normalized, requirements, parser_version)
    return AttributePlan(
        query=normalized,
        requirements=requirements,
        compositional=compositional,
        parser_version=parser_version,
        fingerprint=fingerprint,
    )


def _calibrated_strength(raw_score: float, requirement: AttributeRequirement) -> float:
    lower = requirement.threshold - (requirement.full_score - requirement.threshold)
    denominator = requirement.full_score - lower
    if denominator <= 0:
        return 0.0
    return min(1.0, max(0.0, (raw_score - lower) / denominator))


def aggregate_candidate_evidence(
    plan: AttributePlan,
    evidence: Mapping[str, float],
    global_score: float,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Aggregate one candidate without allowing a strong slot to hide a miss."""

    mandatory = [row for row in plan.requirements if row.mandatory]
    if not plan.compositional or not mandatory:
        return {
            "enabled": False,
            "accepted": True,
            "score": float(global_score),
            "coverage_ratio": 1.0,
            "missing_requirement_ids": [],
            "weakest_requirement_id": None,
            "requirements": [],
        }

    details: list[dict[str, Any]] = []
    total_weight = 0.0
    weighted_sum = 0.0
    weighted_log_sum = 0.0
    missing: list[str] = []
    weakest_id: str | None = None
    weakest_margin = math.inf
    weakest_strength = 1.0

    for requirement in mandatory:
        raw_score = float(evidence.get(requirement.requirement_id, 0.0))
        margin = raw_score - requirement.threshold
        passed = margin >= 0.0
        strength = _calibrated_strength(raw_score, requirement)
        if not passed:
            missing.append(requirement.requirement_id)
        if margin < weakest_margin:
            weakest_margin = margin
            weakest_id = requirement.requirement_id
        weakest_strength = min(weakest_strength, strength)
        total_weight += requirement.weight
        weighted_sum += requirement.weight * strength
        weighted_log_sum += requirement.weight * math.log(max(strength, 1e-6))
        details.append(
            {
                **requirement.to_dict(),
                "raw_score": round(raw_score, 6),
                "margin": round(margin, 6),
                "strength": round(strength, 6),
                "passed": passed,
            }
        )

    coverage_ratio = (len(mandatory) - len(missing)) / len(mandatory)
    weighted_mean = weighted_sum / total_weight
    geometric_mean = math.exp(weighted_log_sum / total_weight)
    global_floor = float(policy.get("global_score_floor", 0.35))
    global_full = float(policy.get("global_score_full", 0.60))
    global_strength = min(
        1.0,
        max(
            0.0,
            (float(global_score) - global_floor)
            / max(global_full - global_floor, 1e-6),
        ),
    )
    weights = policy.get(
        "aggregation_weights",
        {"global": 0.20, "mean": 0.35, "geometric": 0.25, "weakest": 0.20},
    )
    final_score = (
        float(weights.get("global", 0.20)) * global_strength
        + float(weights.get("mean", 0.35)) * weighted_mean
        + float(weights.get("geometric", 0.25)) * geometric_mean
        + float(weights.get("weakest", 0.20)) * weakest_strength
    )
    return {
        "enabled": True,
        "accepted": not missing,
        "score": round(final_score, 6),
        "coverage_ratio": round(coverage_ratio, 6),
        "missing_requirement_ids": missing,
        "weakest_requirement_id": weakest_id,
        "weakest_margin": round(weakest_margin, 6),
        "weakest_strength": round(weakest_strength, 6),
        "attribute_mean": round(weighted_mean, 6),
        "attribute_geometric_mean": round(geometric_mean, 6),
        "global_strength": round(global_strength, 6),
        "requirements": details,
    }
