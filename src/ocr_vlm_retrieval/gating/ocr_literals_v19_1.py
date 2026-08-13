"""V19.1 structured OCR parsing and necessary-condition completeness."""

from __future__ import annotations

import calendar
import re
from collections.abc import Iterable, Sequence
from difflib import SequenceMatcher
from typing import Any

from ocr_vlm_retrieval.gating.candidate_verification import normalize_ocr_text
from ocr_vlm_retrieval.gating.ocr_literals import (
    OcrLiteralGroup,
    evaluate_ocr_literal_groups,
    extract_ocr_literal_groups,
)

CHINESE_DATE_VALUE = re.compile(r"(\d{4})年(\d{1,2})月(\d{1,2})日")
YMD_DATE = re.compile(
    r"(?<!\d)((?:19|20)\d{2})\s*[-/.年]\s*(\d{1,2})"
    r"\s*[-/.月]\s*(\d{1,2})(?:\s*日)?(?!\d)",
    re.IGNORECASE,
)
MDY_DATE = re.compile(
    r"(?<!\d)(\d{1,2})\s*[-/.]\s*(\d{1,2})\s*[-/.]"
    r"\s*(\d{2}|(?:19|20)\d{2})(?!\d)",
    re.IGNORECASE,
)
MONTH_DATE = re.compile(
    r"\b("
    + "|".join(calendar.month_name[1:])
    + "|"
    + "|".join(calendar.month_abbr[1:])
    + r")\s*[-/.]?\s*(\d{1,2})(?:st|nd|rd|th)?"
    r"\s*[,./-]?\s*((?:19|20)?\d{2})\b",
    re.IGNORECASE,
)
IDENTIFIER_KEY = (
    r"(?:报告)?(?:编号|序号|编码|代码|号码|注册号|批号|"
    r"(?<![A-Za-z0-9])No\.?(?![A-Za-z0-9])|#)"
)
IDENTIFIER_VALUE = re.compile(
    IDENTIFIER_KEY + r"\s*(?:为|是|[:：])?\s*"
    r"([A-Za-z0-9][A-Za-z0-9._/-]{0,31})",
    re.IGNORECASE,
)
IDENTIFIER_MARKER = re.compile(IDENTIFIER_KEY, re.IGNORECASE)
PHONE_SUFFIX_VALUE = re.compile(
    r"(?:电话|联系电话|手机号).*?末位为\s*([0-9])", re.DOTALL
)
FLEXIBLE_YEAR = re.compile(r"(?<!\d)((?:19|20)\d{2})\s+年")
GENERIC_PERSON = re.compile(
    r"(?:姓名为|申请人为|患者|记录了|写有)"
    r"([\u3400-\u4dbf\u4e00-\u9fff]{2,4})(?:和|、|，|,|的)"
)
NAMED_FIELD_VALUE = re.compile(
    r"(?:验收部位|报名岗位|应聘岗位|岗位|品牌|项目标题|标题|主题|收件人|"
    r"填写人|制造商|发件机构|开票机构)\s*(?:为|是|[:：])\s*"
    r"([^、，,。？?]{2,64}?)(?=、|，|,|。|？|\?|\s*的|$)"
)
BOUND_LATIN_TERM = re.compile(r"(?<![A-Za-z0-9])([A-Za-z]+-[A-Za-z]+)(?![A-Za-z0-9])")
PHONE_SUFFIX_MARKER = re.compile(r"(?:电话|联系电话|手机号).*?末位为", re.DOTALL)
ORGANIZATION_SUFFIX = (
    "有限责任公司|股份有限公司|有限公司|公司|大学|学院|医院|研究院|研究所|"
    "委员会|管理局|公安局|消防局|银行|基金会|中心"
)
GENERIC_ORGANIZATION = re.compile(
    r"([\u3400-\u4dbf\u4e00-\u9fffA-Za-z0-9&（）()·-]{2,48}?(?:"
    + ORGANIZATION_SUFFIX
    + r"))"
)
ORGANIZATION_MARKER = re.compile(ORGANIZATION_SUFFIX)
DOCUMENT_SUFFIX = (
    "报名表|登记表|申请表|确认表|登记单|申请书|审批页|规格页|证明|报告|合同|"
    "发票|名单|通知|公函|函件"
)
GENERIC_DOCUMENT_TOPIC = re.compile(
    r"([\u3400-\u4dbf\u4e00-\u9fffA-Za-z0-9&（）()·_-]{2,32}?(?:"
    + DOCUMENT_SUFFIX
    + r"))"
)
ORGANIZATION_PREFIXES = (
    "哪一页记录了",
    "哪一份",
    "哪份",
    "查找",
    "找出",
    "找到",
    "检索",
    "定位",
    "浏览",
    "找",
    "发件机构为",
    "机构名称为",
    "机构名为",
    "申请人为",
    "申报人为",
    "单位名称为",
    "单位为",
    "企业名称为",
    "公司名称为",
    "工程名称为",
)
GROUP_SOURCE_PRIORITY = {
    "structured_identifier": 30,
    "translation_alias": 30,
    "structured_date": 30,
    "structured_year": 30,
    "phone_suffix": 30,
    "document_topic": 20,
    "named_field": 20,
    "bound_latin_term": 20,
    "chinese_name": 20,
    "chinese_entity": 10,
    "latin_phrase": 10,
}
NON_PERSON_NAMES = frozenset(
    {"表现", "资料", "内容", "结果", "情况", "页面", "记录", "报告"}
)
LATIN_ALIAS_GROUPS = {
    "HS-40": ("HS-40", "hypersensitive site-40"),
}


def _unique_groups(groups: Iterable[OcrLiteralGroup]) -> tuple[OcrLiteralGroup, ...]:
    """Deduplicate one semantic value while preserving the stronger parser."""

    result: list[OcrLiteralGroup] = []
    positions: dict[str, int] = {}
    for group in groups:
        key = normalize_ocr_text(group.label)
        if not key:
            continue
        position = positions.get(key)
        if position is None:
            positions[key] = len(result)
            result.append(group)
            continue
        previous = result[position]
        if GROUP_SOURCE_PRIORITY.get(group.source, 0) > GROUP_SOURCE_PRIORITY.get(
            previous.source, 0
        ):
            result[position] = group
    return tuple(result)


def _clean_organization(value: str) -> str:
    cleaned = value.strip(" ，。；;：:")
    changed = True
    while changed:
        changed = False
        for prefix in ORGANIZATION_PREFIXES:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix) :].strip(" ，。；;：:")
                changed = True
    cleaned = re.sub(r"^(?:19|20)\d{2}年", "", cleaned)
    for marker in (
        "机构名为",
        "机构名称为",
        "单位名称为",
        "单位为",
        "企业名称为",
        "公司名称为",
        "工程名称为",
        "申请人为",
        "同时写有",
        "写有",
    ):
        if marker in cleaned:
            cleaned = cleaned.rsplit(marker, 1)[-1].strip()
    return cleaned


def _clean_document_topic(value: str) -> str:
    cleaned = value.strip(" ，。；;：:的")
    cleaned = re.sub(r"^(?:查找|找出|找到|检索|定位|浏览|哪一页|哪一份)", "", cleaned)
    if "的" in cleaned:
        cleaned = cleaned.rsplit("的", 1)[-1]
    return cleaned.strip(" ，。；;：:的")


def extract_v19_1_literal_groups(query: str) -> tuple[OcrLiteralGroup, ...]:
    """Extend frozen V19 extraction without reading benchmark metadata."""

    groups = [
        group
        for group in extract_ocr_literal_groups(query)
        if not (group.source == "chinese_name" and group.label in NON_PERSON_NAMES)
    ]
    for match in IDENTIFIER_VALUE.finditer(query):
        value = match.group(1).strip(" .-/")
        if value:
            groups.append(OcrLiteralGroup(value, (value,), "structured_identifier"))
    if not any(group.source == "phone_suffix" for group in groups):
        phone_match = PHONE_SUFFIX_VALUE.search(query)
        if phone_match:
            digit = phone_match.group(1)
            groups.append(
                OcrLiteralGroup(
                    f"phone_number_ends_with_{digit}",
                    (digit,),
                    "phone_suffix",
                )
            )
    for match in FLEXIBLE_YEAR.finditer(query):
        label = f"{match.group(1)}年"
        groups.append(OcrLiteralGroup(label, (match.group(1),), "structured_year"))
    for match in GENERIC_PERSON.finditer(query):
        name = match.group(1)
        if name not in NON_PERSON_NAMES:
            groups.append(OcrLiteralGroup(name, (name,), "chinese_name"))
    for match in NAMED_FIELD_VALUE.finditer(query):
        value = match.group(1).strip(" ，。；;：:的")
        if value:
            groups.append(OcrLiteralGroup(value, (value,), "named_field"))
    for label, variants in LATIN_ALIAS_GROUPS.items():
        if label.casefold() in query.casefold():
            groups.append(OcrLiteralGroup(label, variants, "latin_phrase"))
    for match in BOUND_LATIN_TERM.finditer(query):
        value = match.group(1)
        groups.append(OcrLiteralGroup(value, (value,), "bound_latin_term"))
    for match in GENERIC_ORGANIZATION.finditer(query):
        entity = _clean_organization(match.group(1))
        if len(normalize_ocr_text(entity).replace(" ", "")) >= 4:
            groups.append(OcrLiteralGroup(entity, (entity,), "chinese_entity"))
    for match in GENERIC_DOCUMENT_TOPIC.finditer(query):
        topic = _clean_document_topic(match.group(1))
        if len(normalize_ocr_text(topic).replace(" ", "")) >= 4:
            groups.append(OcrLiteralGroup(topic, (topic,), "document_topic"))
    deduplicated = _unique_groups(groups)
    non_document = [group for group in deduplicated if group.source != "document_topic"]
    if len(non_document) >= 2:
        return tuple(non_document)
    return deduplicated


def completeness_report(
    query: str, groups: Sequence[OcrLiteralGroup]
) -> dict[str, Any]:
    """Report whether every high-risk literal marker produced a constraint."""

    sources = {group.source for group in groups}
    missing: list[str] = []
    if CHINESE_DATE_VALUE.search(query) and "structured_date" not in sources:
        missing.append("structured_date")
    if IDENTIFIER_MARKER.search(query) and "structured_identifier" not in sources:
        missing.append("structured_identifier")
    if PHONE_SUFFIX_MARKER.search(query) and "phone_suffix" not in sources:
        missing.append("phone_suffix")
    organization_covered = bool(
        sources.intersection({"chinese_entity", "translation_alias"})
    )
    if ORGANIZATION_MARKER.search(query) and not organization_covered:
        missing.append("chinese_entity")
    return {
        "complete": not missing,
        "missing_condition_sources": missing,
        "extracted_condition_count": len(groups),
        "extracted_sources": sorted(sources),
    }


def _four_digit_year(value: int) -> int:
    if value >= 100:
        return value
    return 1900 + value if value >= 50 else 2000 + value


def _valid_date(year: int, month: int, day: int) -> tuple[int, int, int] | None:
    try:
        maximum_day = calendar.monthrange(year, month)[1]
    except calendar.IllegalMonthError:
        return None
    if not 1900 <= year <= 2099 or not 1 <= day <= maximum_day:
        return None
    return year, month, day


def parsed_ocr_dates(lines: Iterable[str]) -> set[tuple[int, int, int]]:
    """Parse exact date components across common OCR punctuation variants."""

    corpus = " ".join(str(line) for line in lines)
    result: set[tuple[int, int, int]] = set()
    for match in YMD_DATE.finditer(corpus):
        value = _valid_date(*(int(part) for part in match.groups()))
        if value:
            result.add(value)
    for match in MDY_DATE.finditer(corpus):
        month, day, year = (int(part) for part in match.groups())
        value = _valid_date(_four_digit_year(year), month, day)
        if value:
            result.add(value)
    month_lookup = {
        name.casefold(): index
        for index in range(1, 13)
        for name in (calendar.month_name[index], calendar.month_abbr[index])
    }
    for match in MONTH_DATE.finditer(corpus):
        month = month_lookup[match.group(1).casefold()]
        day = int(match.group(2))
        year = _four_digit_year(int(match.group(3)))
        value = _valid_date(year, month, day)
        if value:
            result.add(value)
    return result


def _structured_date_evidence(
    group: OcrLiteralGroup, lines: Sequence[str]
) -> dict[str, Any]:
    match = CHINESE_DATE_VALUE.fullmatch(group.label)
    if match is None:
        raise ValueError(f"invalid structured date label: {group.label}")
    target = tuple(int(part) for part in match.groups())
    candidates = parsed_ocr_dates(lines)
    matched = target in candidates
    return {
        **group.to_dict(),
        "matched": matched,
        "best_variant": group.label,
        "match_level": "structured_exact" if matched else "none",
        "similarity": 1.0 if matched else 0.0,
        "parsed_date_candidates": [list(value) for value in sorted(candidates)],
    }


def _structured_identifier_evidence(
    group: OcrLiteralGroup, lines: Sequence[str]
) -> dict[str, Any]:
    target = normalize_ocr_text(group.variants[0]).replace(" ", "")
    normalized_lines = [normalize_ocr_text(line).replace(" ", "") for line in lines]
    matched = bool(
        target
        and any(
            target == line or (len(target) >= 4 and target in line)
            for line in normalized_lines
        )
    )
    return {
        **group.to_dict(),
        "matched": matched,
        "best_variant": group.variants[0],
        "match_level": "structured_exact" if matched else "none",
        "similarity": 1.0 if matched else 0.0,
    }


def _latin_variant_tokens(value: str) -> list[str]:
    normalized = value.casefold().replace("α", "alpha").replace("β", "beta")
    return re.findall(r"[a-z0-9]+", normalized)


def _latin_phrase_evidence(
    group: OcrLiteralGroup, lines: Sequence[str], *, fuzzy_threshold: float
) -> dict[str, Any]:
    raw_corpus = " ".join(lines)
    raw_corpus = re.sub(r"-\s+", "", raw_corpus)
    corpus_tokens = _latin_variant_tokens(raw_corpus)
    best_similarity = 0.0
    best_variant = group.variants[0]
    matched = False
    for variant in group.variants:
        target_tokens = _latin_variant_tokens(variant)
        if not target_tokens:
            continue
        token_similarities: list[float] = []
        for target in target_tokens:
            similarity = max(
                (
                    SequenceMatcher(None, target, candidate).ratio()
                    for candidate in corpus_tokens
                ),
                default=0.0,
            )
            token_similarities.append(similarity)
        variant_similarity = min(token_similarities)
        if variant_similarity > best_similarity:
            best_similarity = variant_similarity
            best_variant = variant
        token_matches = [
            similarity
            >= (
                1.0
                if len(target) <= 5
                else min(fuzzy_threshold, 0.85)
                if len(target) >= 7
                else fuzzy_threshold
            )
            for target, similarity in zip(
                target_tokens, token_similarities, strict=True
            )
        ]
        if all(token_matches):
            matched = True
            best_variant = variant
            best_similarity = variant_similarity
            break
    return {
        **group.to_dict(),
        "matched": matched,
        "best_variant": best_variant,
        "match_level": "token_coverage" if matched else "none",
        "similarity": round(best_similarity, 8),
    }


def _bound_latin_evidence(
    group: OcrLiteralGroup, lines: Sequence[str], *, fuzzy_threshold: float
) -> dict[str, Any]:
    corpus = " ".join(lines).casefold().replace("α", "alpha").replace("β", "beta")
    corpus = re.sub(r"-\s+", "", corpus)
    corpus_tokens = re.findall(r"[a-z0-9]+", corpus)
    target_tokens = _latin_variant_tokens(group.variants[0])
    best_similarity = 0.0
    matched = False
    for start in range(0, len(corpus_tokens) - len(target_tokens) + 1):
        window = corpus_tokens[start : start + len(target_tokens)]
        similarities = [
            SequenceMatcher(None, target, candidate).ratio()
            for target, candidate in zip(target_tokens, window, strict=True)
        ]
        score = min(similarities, default=0.0)
        best_similarity = max(best_similarity, score)
        if all(
            similarity >= (1.0 if len(target) <= 5 else min(fuzzy_threshold, 0.85))
            for target, similarity in zip(target_tokens, similarities, strict=True)
        ):
            matched = True
            best_similarity = score
            break
    return {
        **group.to_dict(),
        "matched": matched,
        "best_variant": group.variants[0],
        "match_level": "bound_token_sequence" if matched else "none",
        "similarity": round(best_similarity, 8),
    }


def evaluate_v19_1_literal_groups(
    groups: Iterable[OcrLiteralGroup],
    lines: Iterable[str],
    *,
    fuzzy_threshold: float = 0.88,
) -> dict[str, Any]:
    """Require every extracted condition, with structured field parsers."""

    collected_groups = tuple(groups)
    collected_lines = tuple(str(line) for line in lines)
    evidence: list[dict[str, Any]] = []
    for group in collected_groups:
        if group.source == "structured_date":
            evidence.append(_structured_date_evidence(group, collected_lines))
        elif group.source == "structured_identifier":
            evidence.append(_structured_identifier_evidence(group, collected_lines))
        elif group.source == "latin_phrase":
            evidence.append(
                _latin_phrase_evidence(
                    group,
                    collected_lines,
                    fuzzy_threshold=fuzzy_threshold,
                )
            )
        elif group.source == "bound_latin_term":
            evidence.append(
                _bound_latin_evidence(
                    group,
                    collected_lines,
                    fuzzy_threshold=fuzzy_threshold,
                )
            )
        else:
            base = evaluate_ocr_literal_groups(
                (group,), collected_lines, fuzzy_threshold=fuzzy_threshold
            )
            evidence.append(dict(base["evidence"][0]))
    matched_count = sum(bool(row["matched"]) for row in evidence)
    return {
        "constraint_count": len(collected_groups),
        "matched_constraint_count": matched_count,
        "all_constraints_matched": bool(collected_groups)
        and matched_count == len(collected_groups),
        "evidence": evidence,
    }
