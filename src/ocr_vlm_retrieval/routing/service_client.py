"""Failure-safe client for the optional persistent V19 routing service."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, cast
from urllib import error, request

from ocr_vlm_retrieval.routing.rule_router import RuleRouter
from ocr_vlm_retrieval.routing.schema import IntentEvidence, Route, validate_route


@dataclass(frozen=True, slots=True)
class ServiceRouteDecision:
    """Validated service result or deterministic local fallback."""

    route: Route
    source: str
    fallback_error_type: str | None = None
    guard_reason: str | None = None
    llm_invoked: bool = False
    rule_route: Route | None = None
    rule_reason_codes: tuple[str, ...] = ()
    backend_name: str | None = None
    evidence: IntentEvidence | None = None
    route_latency_ms: float | None = None

    def to_mapping(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "source": self.source,
            "fallback_error_type": self.fallback_error_type,
            "guard_reason": self.guard_reason,
            "llm_invoked": self.llm_invoked,
            "rule_route": self.rule_route,
            "rule_reason_codes": list(self.rule_reason_codes),
            "backend_name": self.backend_name,
            "evidence": (
                self.evidence.to_mapping() if self.evidence is not None else None
            ),
            "route_latency_ms": self.route_latency_ms,
        }


def _validated_service_decision(payload: Any) -> ServiceRouteDecision:
    if not isinstance(payload, dict):
        raise ValueError("V19 routing service response must be a JSON object")
    route_value = payload.get("route")
    if not isinstance(route_value, str):
        raise ValueError("V19 routing service route must be a string")
    route = validate_route(route_value)
    source = payload.get("source")
    if source not in {"rule", "llm", "rule_guard", "rule_fallback"}:
        raise ValueError("V19 routing service returned an invalid source")
    fallback_error_type = payload.get("fallback_error_type")
    guard_reason = payload.get("guard_reason")
    llm_invoked = payload.get("llm_invoked", False)
    if fallback_error_type is not None and not isinstance(
        fallback_error_type, str
    ):
        raise ValueError("fallback_error_type must be a string or null")
    if guard_reason is not None and not isinstance(guard_reason, str):
        raise ValueError("guard_reason must be a string or null")
    if not isinstance(llm_invoked, bool):
        raise ValueError("llm_invoked must be a boolean")
    rule_route_value = payload.get("rule_route")
    rule_route = (
        validate_route(rule_route_value)
        if isinstance(rule_route_value, str)
        else None
    )
    rule_reason_codes_value = payload.get("rule_reason_codes", [])
    if not isinstance(rule_reason_codes_value, list) or not all(
        isinstance(value, str) for value in rule_reason_codes_value
    ):
        raise ValueError("rule_reason_codes must be a string list")
    backend_name = payload.get("backend_name")
    if backend_name is not None and not isinstance(backend_name, str):
        raise ValueError("backend_name must be a string or null")
    evidence_value = payload.get("evidence")
    if evidence_value is not None and not isinstance(evidence_value, dict):
        raise ValueError("evidence must be an object or null")
    evidence = (
        IntentEvidence.from_mapping(cast(dict[str, Any], evidence_value))
        if evidence_value is not None
        else None
    )
    route_latency_ms = payload.get("route_latency_ms")
    if route_latency_ms is not None and (
        isinstance(route_latency_ms, bool)
        or not isinstance(route_latency_ms, (int, float))
        or route_latency_ms < 0
    ):
        raise ValueError("route_latency_ms must be a non-negative number or null")
    return ServiceRouteDecision(
        route=route,
        source=source,
        fallback_error_type=fallback_error_type,
        guard_reason=guard_reason,
        llm_invoked=llm_invoked,
        rule_route=rule_route,
        rule_reason_codes=tuple(rule_reason_codes_value),
        backend_name=backend_name,
        evidence=evidence,
        route_latency_ms=(
            float(route_latency_ms) if route_latency_ms is not None else None
        ),
    )


def request_v19_route(
    service_url: str,
    query: str,
    *,
    timeout_seconds: float = 2.0,
) -> ServiceRouteDecision:
    """Request one route from a loopback service and validate its contract."""

    normalized = " ".join(query.split())
    if not normalized:
        raise ValueError("query must not be empty")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    endpoint = service_url.rstrip("/") + "/route"
    body = json.dumps({"query": normalized}, ensure_ascii=False).encode("utf-8")
    http_request = request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with request.urlopen(http_request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        raise RuntimeError(f"V19 routing service HTTP {exc.code}") from exc
    except error.URLError as exc:
        raise ConnectionError("V19 routing service is unavailable") from exc
    except json.JSONDecodeError as exc:
        raise ValueError("V19 routing service returned invalid JSON") from exc
    return _validated_service_decision(payload)


def request_v19_route_with_fallback(
    service_url: str,
    query: str,
    *,
    timeout_seconds: float = 2.0,
) -> ServiceRouteDecision:
    """Always return a route, falling back to the frozen V18 rule."""

    try:
        service_decision = request_v19_route(
            service_url,
            query,
            timeout_seconds=timeout_seconds,
        )
        if service_decision.source != "rule_fallback":
            return service_decision
        decision = RuleRouter.legacy().route(query)
        return ServiceRouteDecision(
            route=decision.route,
            source="rule_fallback",
            fallback_error_type=service_decision.fallback_error_type,
            guard_reason="restore_frozen_v18_after_service_fallback",
            llm_invoked=service_decision.llm_invoked,
            rule_route=decision.route,
            rule_reason_codes=decision.reason_codes,
            backend_name=service_decision.backend_name,
            evidence=service_decision.evidence,
            route_latency_ms=service_decision.route_latency_ms,
        )
    except Exception as exc:
        decision = RuleRouter.legacy().route(query)
        return ServiceRouteDecision(
            route=decision.route,
            source="rule_fallback",
            fallback_error_type=type(exc).__name__,
            llm_invoked=False,
            rule_route=decision.route,
            rule_reason_codes=decision.reason_codes,
        )
