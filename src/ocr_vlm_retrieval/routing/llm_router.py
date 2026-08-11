"""Model-agnostic structured intent router for the V19 pilot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ocr_vlm_retrieval.routing.mapping import map_evidence_to_route
from ocr_vlm_retrieval.routing.schema import IntentEvidence, Route


class IntentBackend(Protocol):
    """Minimal boundary implemented later by a local model process."""

    @property
    def name(self) -> str:
        """Return a stable backend and model identifier."""

    def generate_intent_json(self, query: str) -> str:
        """Return exactly one JSON object for ``query``."""


@dataclass(frozen=True, slots=True)
class LLMDecision:
    """Validated model evidence and its deterministic route."""

    route: Route
    evidence: IntentEvidence
    backend_name: str


class LLMRouter:
    """Validate model JSON before applying deterministic route mapping."""

    def __init__(self, backend: IntentBackend) -> None:
        self._backend = backend

    def route(self, query: str) -> LLMDecision:
        """Generate, validate and map one query."""

        normalized = " ".join(query.split())
        if not normalized:
            raise ValueError("query must not be empty")
        evidence = IntentEvidence.from_json(
            self._backend.generate_intent_json(normalized)
        )
        return LLMDecision(
            route=map_evidence_to_route(evidence),
            evidence=evidence,
            backend_name=self._backend.name,
        )
