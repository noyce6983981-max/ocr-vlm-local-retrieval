"""Failure-safe rule and local-LLM hybrid routing for V19."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ocr_vlm_retrieval.routing.arbitration import guard_llm_route
from ocr_vlm_retrieval.routing.llm_router import LLMDecision, LLMRouter
from ocr_vlm_retrieval.routing.rule_router import RuleDecision, RuleRouter
from ocr_vlm_retrieval.routing.schema import Route

DecisionSource = Literal["rule", "llm", "rule_guard", "rule_fallback"]


@dataclass(frozen=True, slots=True)
class HybridDecision:
    """Final route plus complete provenance needed for evaluation."""

    route: Route
    source: DecisionSource
    rule: RuleDecision
    llm: LLMDecision | None = None
    fallback_error_type: str | None = None
    guard_reason: str | None = None


class HybridRouter:
    """Call the LLM only for deterministic ambiguity and always fall back."""

    def __init__(self, rule_router: RuleRouter, llm_router: LLMRouter) -> None:
        self._rule_router = rule_router
        self._llm_router = llm_router

    def route(self, query: str) -> HybridDecision:
        """Route one query with schema and runtime failure isolation."""

        rule_decision = self._rule_router.route(query)
        if not rule_decision.ambiguous:
            return HybridDecision(
                route=rule_decision.route,
                source="rule",
                rule=rule_decision,
            )
        try:
            llm_decision = self._llm_router.route(query)
        except Exception as exc:
            return HybridDecision(
                route=rule_decision.route,
                source="rule_fallback",
                rule=rule_decision,
                fallback_error_type=type(exc).__name__,
            )
        guarded_route, guard_reason = guard_llm_route(
            query,
            rule_route=rule_decision.route,
            llm_route=llm_decision.route,
        )
        return HybridDecision(
            route=guarded_route,
            source="rule_guard" if guard_reason else "llm",
            rule=rule_decision,
            llm=llm_decision,
            guard_reason=guard_reason,
        )
