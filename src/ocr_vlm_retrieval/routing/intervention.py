"""Safety boundary between semantic routing and frozen V18 execution."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

from ocr_vlm_retrieval.routing.schema import IntentEvidence, Route

IntentRoutingMode: TypeAlias = Literal["off", "shadow", "guarded", "active"]
INTENT_ROUTING_MODES: Final[tuple[IntentRoutingMode, ...]] = (
    "off",
    "shadow",
    "guarded",
    "active",
)
DISCOVERY_ROUTES: Final[frozenset[Route]] = frozenset(
    {"visual_discovery", "topic_discovery"}
)
FACTUAL_ROUTES: Final[frozenset[Route]] = frozenset(
    {"text_evidence", "visual_metadata", "entity_exact", "mixed"}
)


@dataclass(frozen=True, slots=True)
class TransitionDecision:
    """Candidate route, applied route and the reason for preserving V18."""

    candidate_route: Route
    applied_route: Route
    guard_reason: str | None
    legacy_branches: dict[str, bool]
    candidate_branches: dict[str, bool]


def validate_intent_routing_mode(value: str) -> IntentRoutingMode:
    if value == "off":
        return "off"
    if value == "shadow":
        return "shadow"
    if value == "guarded":
        return "guarded"
    if value == "active":
        return "active"
    raise ValueError(f"Unsupported intent routing mode: {value}")


def guard_v18_transition(
    *,
    query: str,
    method: str,
    legacy_route: Route,
    candidate_route: Route,
    evidence: IntentEvidence | None,
    extract_strict_entity_term: Callable[[str], str | None],
    required_search_branches: Callable[
        [str, str, bool], Mapping[str, bool]
    ],
) -> TransitionDecision:
    """Reject candidate policies that weaken frozen V18 safety semantics."""

    legacy_exploratory = legacy_route in DISCOVERY_ROUTES
    candidate_exploratory = candidate_route in DISCOVERY_ROUTES
    legacy_branches = dict(
        required_search_branches(method, legacy_route, legacy_exploratory)
    )
    candidate_branches = dict(
        required_search_branches(method, candidate_route, candidate_exploratory)
    )
    guard_reason: str | None = None
    if legacy_route in FACTUAL_ROUTES and candidate_route in DISCOVERY_ROUTES:
        guard_reason = "preserve_factual_acceptance"
    elif (
        candidate_route == "entity_exact"
        and extract_strict_entity_term(query) is None
    ):
        guard_reason = "entity_exact_without_strict_term"
    elif evidence is not None and evidence.is_compositional and any(
        legacy_branches.get(branch, False)
        and not candidate_branches.get(branch, False)
        for branch in ("text", "bm25", "visual")
    ):
        guard_reason = "compositional_candidate_removes_legacy_branch"
    elif any(
        legacy_branches.get(branch, False)
        and not candidate_branches.get(branch, False)
        for branch in ("text", "bm25", "visual")
    ):
        guard_reason = "candidate_removes_legacy_branch"
    return TransitionDecision(
        candidate_route=candidate_route,
        applied_route=legacy_route if guard_reason else candidate_route,
        guard_reason=guard_reason,
        legacy_branches=legacy_branches,
        candidate_branches=candidate_branches,
    )
