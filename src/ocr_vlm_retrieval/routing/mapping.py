"""Deterministic mapping from V19 evidence fields to retrieval routes."""

from __future__ import annotations

from ocr_vlm_retrieval.routing.schema import IntentEvidence, Route


def map_evidence_to_route(evidence: IntentEvidence) -> Route:
    """Map structured evidence needs without model-controlled thresholds."""

    if evidence.needs_exact_entity:
        return "entity_exact"
    if evidence.needs_literal_text and (
        evidence.needs_visual_semantics
        or evidence.needs_layout_structure
    ):
        return "mixed"
    if evidence.needs_layout_structure:
        return "visual_metadata"
    if evidence.needs_visual_semantics:
        return "visual_discovery"
    if evidence.needs_topic_discovery:
        return "topic_discovery"
    if evidence.needs_literal_text:
        return "text_evidence"
    return "mixed"
