"""Acceptance and evidence-gating components."""

from .attribute_coverage import (
    AttributePlan,
    AttributeRequirement,
    aggregate_candidate_evidence,
    decompose_visual_query,
    load_attribute_policy,
)

__all__ = [
    "AttributePlan",
    "AttributeRequirement",
    "aggregate_candidate_evidence",
    "decompose_visual_query",
    "load_attribute_policy",
]

