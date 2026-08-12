"""Conservative candidate selection from mandatory OCR literal evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ocr_vlm_retrieval.gating.ocr_literals import (
    OcrLiteralGroup,
    evaluate_ocr_literal_groups,
    extract_ocr_literal_groups,
)

OCR_LITERAL_STRATUM = "ocr_literal_lookup"
TOPIC_DISCOVERY_STRATUM = "topic_discovery"
ENTITY_IDENTIFIER_STRATUM = "entity_identifier"
TOPIC_ALLOWED_SOURCES = frozenset(
    {"latin_phrase", "structured_year", "translation_alias"}
)
TOPIC_NAMED_SOURCES = frozenset({"latin_phrase", "translation_alias"})
ENTITY_EXACT_SOURCES = frozenset({"chinese_name", "chinese_entity"})


def literal_override_is_eligible(
    content_stratum: str,
    groups: Sequence[OcrLiteralGroup],
) -> bool:
    """Restrict literal acceptance to routes with a high-precision contract."""

    if not groups:
        return False
    if content_stratum == OCR_LITERAL_STRATUM:
        return True
    sources = {group.source for group in groups}
    if content_stratum == ENTITY_IDENTIFIER_STRATUM:
        return sources.issubset(ENTITY_EXACT_SOURCES)
    if content_stratum != TOPIC_DISCOVERY_STRATUM:
        return False
    return sources.issubset(TOPIC_ALLOWED_SOURCES) and bool(
        sources.intersection(TOPIC_NAMED_SOURCES)
    )


def select_literal_candidate(
    query: str,
    candidate_item_ids: Sequence[str],
    lines_by_item: Mapping[str, Sequence[str]],
    *,
    fuzzy_threshold: float,
) -> dict[str, Any]:
    """Select the first candidate satisfying every extracted OCR constraint."""

    groups = extract_ocr_literal_groups(query)
    candidates: list[dict[str, Any]] = []
    selected_item_id: str | None = None
    for rank, item_id in enumerate(candidate_item_ids, start=1):
        evidence = evaluate_ocr_literal_groups(
            groups,
            lines_by_item.get(item_id, ()),
            fuzzy_threshold=fuzzy_threshold,
        )
        candidates.append(
            {
                "item_id": item_id,
                "retrieval_rank": rank,
                **evidence,
            }
        )
        if selected_item_id is None and evidence["all_constraints_matched"]:
            selected_item_id = item_id
    return {
        "constraint_groups": [group.to_dict() for group in groups],
        "candidate_evidence": candidates,
        "accepted": selected_item_id is not None,
        "selected_item_id": selected_item_id,
        "reason": (
            "all_ocr_literal_constraints_matched"
            if selected_item_id is not None
            else "no_candidate_matches_all_ocr_literal_constraints"
        ),
    }
