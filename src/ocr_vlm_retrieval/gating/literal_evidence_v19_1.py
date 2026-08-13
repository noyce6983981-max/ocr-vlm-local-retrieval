"""V19.1 query-only eligibility and complete literal evidence selection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ocr_vlm_retrieval.gating.candidate_verification import normalize_ocr_text
from ocr_vlm_retrieval.gating.ocr_literals import OcrLiteralGroup
from ocr_vlm_retrieval.gating.ocr_literals_v19_1 import (
    completeness_report,
    evaluate_v19_1_literal_groups,
    extract_v19_1_literal_groups,
)

EXACT_ENTITY_SOURCES = frozenset({"chinese_name", "chinese_entity"})
HIGH_PRECISION_SOURCES = frozenset(
    {"structured_date", "phone_suffix", "translation_alias"}
)
TOPIC_NAMED_SOURCES = frozenset(
    {"latin_phrase", "bound_latin_term", "translation_alias"}
)


def v19_1_override_eligibility(query: str) -> dict[str, Any]:
    """Return an auditable decision using query text only."""

    groups = extract_v19_1_literal_groups(query)
    completeness = completeness_report(query, groups)
    sources = {group.source for group in groups}
    eligible = False
    reason = "no_high_precision_literal_contract"
    if not completeness["complete"]:
        reason = "necessary_condition_extraction_incomplete"
    elif sources.intersection(EXACT_ENTITY_SOURCES):
        eligible = True
        reason = "exact_entity_contract"
    elif sources.intersection(HIGH_PRECISION_SOURCES):
        eligible = True
        reason = "high_precision_structured_contract"
    elif (
        "structured_identifier" in sources
        and len({normalize_ocr_text(group.label) for group in groups}) >= 2
    ):
        eligible = True
        reason = "identifier_plus_additional_condition_contract"
    elif "浏览" in query and sources.intersection(TOPIC_NAMED_SOURCES):
        eligible = True
        reason = "explicit_named_discovery_contract"
    return {
        "eligible": eligible,
        "reason": reason,
        "constraint_groups": [group.to_dict() for group in groups],
        "completeness": completeness,
    }


def select_v19_1_literal_candidate(
    query: str,
    candidate_item_ids: Sequence[str],
    lines_by_item: Mapping[str, Sequence[str]],
    *,
    fuzzy_threshold: float,
) -> dict[str, Any]:
    """Select the first candidate satisfying every V19.1 condition."""

    eligibility = v19_1_override_eligibility(query)
    groups = tuple(
        OcrLiteralGroup(
            label=str(group["label"]),
            variants=tuple(str(value) for value in group["variants"]),
            source=str(group["source"]),
        )
        for group in eligibility["constraint_groups"]
    )
    candidates: list[dict[str, Any]] = []
    selected_item_id: str | None = None
    if eligibility["eligible"]:
        for rank, item_id in enumerate(candidate_item_ids, start=1):
            evidence = evaluate_v19_1_literal_groups(
                groups,
                lines_by_item.get(item_id, ()),
                fuzzy_threshold=fuzzy_threshold,
            )
            candidates.append({"item_id": item_id, "retrieval_rank": rank, **evidence})
            if selected_item_id is None and evidence["all_constraints_matched"]:
                selected_item_id = item_id
    return {
        **eligibility,
        "candidate_evidence": candidates,
        "accepted": selected_item_id is not None,
        "selected_item_id": selected_item_id,
        "selection_reason": (
            "all_v19_1_literal_constraints_matched"
            if selected_item_id is not None
            else "no_candidate_matches_complete_v19_1_contract"
        ),
    }
