"""Framework-free persistent service contract for V19 intent routing."""

from __future__ import annotations

import time
from typing import Any

from ocr_vlm_retrieval.routing.hybrid_router import HybridRouter


def route_payload(router: HybridRouter, query: str) -> dict[str, Any]:
    """Return the small auditable payload exposed by the loopback server."""

    normalized = " ".join(query.split())
    if not normalized:
        raise ValueError("query must not be empty")
    started = time.perf_counter()
    decision = router.route(normalized)
    route_latency_ms = round((time.perf_counter() - started) * 1000, 3)
    return {
        "route": decision.route,
        "source": decision.source,
        "llm_invoked": decision.source != "rule",
        "fallback_error_type": decision.fallback_error_type,
        "guard_reason": decision.guard_reason,
        "rule_route": decision.rule.route,
        "rule_reason_codes": list(decision.rule.reason_codes),
        "backend_name": (
            decision.llm.backend_name if decision.llm is not None else None
        ),
        "evidence": (
            decision.llm.evidence.to_mapping()
            if decision.llm is not None
            else None
        ),
        "route_latency_ms": route_latency_ms,
    }
