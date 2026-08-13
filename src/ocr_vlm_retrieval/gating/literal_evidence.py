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
EXACT_ENTITY_SOURCES = frozenset({"chinese_name", "chinese_entity"})
HIGH_PRECISION_LITERAL_SOURCES = frozenset(
    {"structured_date", "phone_suffix", "translation_alias"}
)
TOPIC_NAMED_SOURCES = frozenset({"latin_phrase", "translation_alias"})


def literal_override_is_eligible(
    query: str,
    groups: Sequence[OcrLiteralGroup],
) -> bool:
    """Decide from deployable query evidence, never evaluation metadata.

    ``content_stratum`` used to be supplied by the benchmark authoring record.
    That label is unavailable for a live query, so it must not control runtime
    behavior.  The frozen contract below uses only text visible to the system:

    * exact Chinese person or organization names;
    * dates, phone suffixes, and curated translation aliases; or
    * an explicit discovery request (``浏览``) with a named literal.

    Standalone amounts and arbitrary Latin words are deliberately insufficient
    because they also occur frequently in visual/layout queries.
    """

    if not groups:
        return False
    sources = {group.source for group in groups}
    if sources.intersection(EXACT_ENTITY_SOURCES):
        return True
    if sources.intersection(HIGH_PRECISION_LITERAL_SOURCES):
        return True
    return "浏览" in query and bool(sources.intersection(TOPIC_NAMED_SOURCES))


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
