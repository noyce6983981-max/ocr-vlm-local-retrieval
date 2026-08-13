"""Explicit necessary-condition parsing for V19.2 automatic development."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from ocr_vlm_retrieval.gating.candidate_verification import normalize_ocr_text
from ocr_vlm_retrieval.gating.ocr_literals import OcrLiteralGroup
from ocr_vlm_retrieval.gating.ocr_literals_v19_1 import (
    evaluate_v19_1_literal_groups,
    extract_v19_1_literal_groups,
)

EXPLICIT_CONDITION_INTENT = re.compile(r"(?:必须包含|同时包含)")
QUOTED_VALUE = re.compile(r"[“\"]([^”\"]{2,80})[”\"]")


def extract_v19_2_literal_groups(query: str) -> tuple[OcrLiteralGroup, ...]:
    """Add every quoted value when the query explicitly requires conjunction."""

    explicit: list[OcrLiteralGroup] = []
    if EXPLICIT_CONDITION_INTENT.search(query):
        explicit.extend(
            OcrLiteralGroup(value.strip(), (value.strip(),), "explicit_required")
            for value in QUOTED_VALUE.findall(query)
            if value.strip()
        )
    if len(explicit) >= 2:
        return tuple(explicit)
    groups: list[OcrLiteralGroup] = list(explicit)
    groups.extend(extract_v19_1_literal_groups(query))
    result: list[OcrLiteralGroup] = []
    covered: set[str] = set()
    for group in groups:
        key = normalize_ocr_text(group.label)
        if key and key not in covered:
            covered.add(key)
            result.append(group)
    return tuple(result)


def evaluate_v19_2_literal_groups(
    groups: Iterable[OcrLiteralGroup],
    lines: Iterable[str],
    *,
    fuzzy_threshold: float,
) -> dict[str, Any]:
    """Evaluate explicit quoted values exactly and other V19.1 groups normally."""

    collected_groups = tuple(groups)
    collected_lines = tuple(str(line) for line in lines)
    evidence: list[dict[str, Any]] = []
    for group in collected_groups:
        if group.source == "explicit_required":
            target = normalize_ocr_text(group.variants[0])
            corpus = normalize_ocr_text(" ".join(collected_lines))
            matched = bool(target and target in corpus)
            evidence.append(
                {
                    **group.to_dict(),
                    "matched": matched,
                    "best_variant": group.variants[0],
                    "match_level": "exact" if matched else "none",
                    "similarity": 1.0 if matched else 0.0,
                }
            )
            continue
        result = evaluate_v19_1_literal_groups(
            (group,), collected_lines, fuzzy_threshold=fuzzy_threshold
        )
        evidence.append(dict(result["evidence"][0]))
    matched_count = sum(bool(row["matched"]) for row in evidence)
    return {
        "constraint_count": len(collected_groups),
        "matched_constraint_count": matched_count,
        "all_constraints_matched": bool(collected_groups)
        and matched_count == len(collected_groups),
        "evidence": evidence,
    }
