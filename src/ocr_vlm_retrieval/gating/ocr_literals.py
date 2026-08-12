"""Extract and verify high-precision literal OCR constraints from a query."""

from __future__ import annotations

import calendar
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

from ocr_vlm_retrieval.gating.candidate_verification import (
    evaluate_ocr_requirement,
    normalize_ocr_text,
)

CHINESE_DATE = re.compile(r"(\d{4})年(\d{1,2})月(\d{1,2})日")
STANDALONE_YEAR = re.compile(r"(?<!\d)((?:19|20)\d{2})年")
LATIN_PHRASE = re.compile(
    r"(?<![A-Za-z0-9])((?:\d+[A-Za-z][A-Za-z0-9]*|[A-Za-z][A-Za-z0-9]*)"
    r"(?:\s+(?:&\s+)?(?:\d+[A-Za-z][A-Za-z0-9]*|"
    r"[A-Za-z][A-Za-z0-9]*))*)(?![A-Za-z0-9])"
)
CHINESE_NAME = re.compile(
    r"(?:查找|姓名为|申请人为|患者)"
    r"([\u3400-\u4dbf\u4e00-\u9fff]{2,4})(?:的|，|,|？|\?|。|$)"
)
CHINESE_ORGANIZATION = re.compile(
    r"机构名为([\u3400-\u4dbf\u4e00-\u9fff]{2,12}?)(?:的|，|,|。|$)"
)
CHINESE_QUOTED_ENTITY = re.compile(r"实体[“\"]([^”\"]{2,20})[”\"]")
PHONE_SUFFIX = re.compile(r"(?:电话|联系电话|手机号)[^\d]{0,8}末位为(\d)")
PHONE_LIKE = re.compile(r"(?<!\d)\d[\d\s()\-]{5,}\d(?!\d)")
CURRENCY_AMOUNT = re.compile(
    r"[$￥]?\s*(\d{1,3}(?:,\d{3})+|\d+)\s*(?:美元|元|REWARD)",
    re.IGNORECASE,
)

TRANSLATED_CONSTRAINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("黑色墨水", ("black ink", "黑色墨水")),
    ("蓝色墨水", ("blue ink", "蓝色墨水")),
    ("大写字母", ("BLOCK CAPITALS", "大写字母")),
    ("必填项", ("mandatory", "必填")),
    ("带星号", ("mandatory", "marked with *", "带*")),
    ("被动吸烟", ("passive smoking", "被动吸烟")),
    ("主动吸烟", ("active smoking", "主动吸烟")),
    ("烟草行业", ("tobacco industry", "烟草行业")),
    ("烟草业", ("tobacco industry", "烟草业")),
    ("联邦贸易委员会", ("Federal Trade Commission", "联邦贸易委员会")),
)


@dataclass(frozen=True)
class OcrLiteralGroup:
    """Alternative surface forms for one mandatory textual condition."""

    label: str
    variants: tuple[str, ...]
    source: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = " ".join(str(value).split()).strip(" ,，。？?！!")
        normalized = normalize_ocr_text(cleaned)
        if cleaned and normalized and normalized not in seen:
            seen.add(normalized)
            result.append(cleaned)
    return tuple(result)


def _date_variants(year: int, month: int, day: int) -> tuple[str, ...]:
    if not 1 <= month <= 12 or not 1 <= day <= 31:
        return ()
    short_year = year % 100
    month_name = calendar.month_name[month]
    month_abbreviation = calendar.month_abbr[month]
    return _unique(
        (
            f"{year}年{month}月{day}日",
            f"{year}-{month}-{day}",
            f"{year}/{month}/{day}",
            f"{month}/{day}/{year}",
            f"{month}-{day}-{year}",
            f"{month}/{day}/{short_year:02d}",
            f"{month}-{day}-{short_year:02d}",
            f"{month}.{day}.{short_year:02d}",
            f"{month_name} {day}, {year}",
            f"{month_abbreviation} {day}, {year}",
        )
    )


def extract_ocr_literal_groups(query: str) -> tuple[OcrLiteralGroup, ...]:
    """Return label-blind literal constraints that OCR can verify reliably."""

    groups: list[OcrLiteralGroup] = []
    date_matches = list(CHINESE_DATE.finditer(query))
    for match in date_matches:
        year, month, day = (int(value) for value in match.groups())
        variants = _date_variants(year, month, day)
        if variants:
            groups.append(
                OcrLiteralGroup(match.group(0), variants, "structured_date")
            )
    for match in STANDALONE_YEAR.finditer(query):
        if any(
            match.start() >= date.start() and match.end() <= date.end()
            for date in date_matches
        ):
            continue
        year_text = match.group(1)
        groups.append(
            OcrLiteralGroup(
                f"{year_text}年",
                (year_text,),
                "structured_year",
            )
        )
    for match in CURRENCY_AMOUNT.finditer(query):
        formatted = match.group(1)
        compact = formatted.replace(",", "")
        groups.append(
            OcrLiteralGroup(
                match.group(0).strip(),
                _unique((formatted, compact, f"${formatted}")),
                "structured_amount",
            )
        )
    for phrase in LATIN_PHRASE.findall(query):
        variants = _unique((phrase,))
        if variants and len(normalize_ocr_text(phrase).replace(" ", "")) >= 2:
            groups.append(OcrLiteralGroup(phrase, variants, "latin_phrase"))
    for marker, variants in TRANSLATED_CONSTRAINTS:
        if marker in query:
            groups.append(
                OcrLiteralGroup(marker, _unique(variants), "translation_alias")
            )
    name_match = CHINESE_NAME.search(query)
    if name_match:
        name = name_match.group(1)
        groups.append(OcrLiteralGroup(name, (name,), "chinese_name"))
    for entity_pattern in (CHINESE_ORGANIZATION, CHINESE_QUOTED_ENTITY):
        entity_match = entity_pattern.search(query)
        if entity_match:
            entity = entity_match.group(1)
            groups.append(OcrLiteralGroup(entity, (entity,), "chinese_entity"))
    phone_suffix_match = PHONE_SUFFIX.search(query)
    if phone_suffix_match:
        digit = phone_suffix_match.group(1)
        groups.append(
            OcrLiteralGroup(
                f"phone_number_ends_with_{digit}",
                (digit,),
                "phone_suffix",
            )
        )

    deduplicated: list[OcrLiteralGroup] = []
    covered: set[str] = set()
    for group in groups:
        normalized_variants = {
            normalize_ocr_text(variant) for variant in group.variants
        }
        if normalized_variants.intersection(covered):
            continue
        covered.update(normalized_variants)
        deduplicated.append(group)
    return tuple(deduplicated)


def evaluate_ocr_literal_groups(
    groups: Iterable[OcrLiteralGroup],
    lines: Iterable[str],
    *,
    fuzzy_threshold: float = 0.88,
) -> dict[str, Any]:
    """Require at least one OCR match for every mandatory literal group."""

    collected_groups = tuple(groups)
    collected_lines = tuple(str(line) for line in lines)
    evidence: list[dict[str, Any]] = []
    for group in collected_groups:
        if group.source == "phone_suffix":
            suffix = group.variants[0]
            phone_numbers = [
                "".join(
                    character
                    for character in match.group(0)
                    if character.isdigit()
                )
                for line in collected_lines
                for match in PHONE_LIKE.finditer(line)
            ]
            matched = any(
                len(number) >= 7 and number.endswith(suffix)
                for number in phone_numbers
            )
            evidence.append(
                {
                    **group.to_dict(),
                    "matched": matched,
                    "best_variant": suffix,
                    "match_level": "exact" if matched else "none",
                    "similarity": 1.0 if matched else 0.0,
                }
            )
            continue
        group_fuzzy_threshold = (
            1.01
            if group.source.startswith("structured_")
            else fuzzy_threshold
        )
        alternatives = [
            evaluate_ocr_requirement(
                variant,
                collected_lines,
                fuzzy_threshold=group_fuzzy_threshold,
            )
            for variant in group.variants
        ]
        best = max(
            alternatives,
            key=lambda row: (
                row["match_level"] == "exact",
                row["match_level"] == "fuzzy",
                float(row["similarity"]),
            ),
        )
        evidence.append(
            {
                **group.to_dict(),
                "matched": best["match_level"] in {"exact", "fuzzy"},
                "best_variant": best["term"],
                "match_level": best["match_level"],
                "similarity": best["similarity"],
            }
        )
    matched_count = sum(bool(row["matched"]) for row in evidence)
    return {
        "constraint_count": len(collected_groups),
        "matched_constraint_count": matched_count,
        "all_constraints_matched": bool(collected_groups)
        and matched_count == len(collected_groups),
        "evidence": evidence,
    }
