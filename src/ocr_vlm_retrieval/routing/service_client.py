"""Failure-safe client for the optional persistent V19 routing service."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib import error, request

from ocr_vlm_retrieval.routing.rule_router import RuleRouter
from ocr_vlm_retrieval.routing.schema import Route, validate_route


@dataclass(frozen=True, slots=True)
class ServiceRouteDecision:
    """Validated service result or deterministic local fallback."""

    route: Route
    source: str
    fallback_error_type: str | None = None
    guard_reason: str | None = None
    llm_invoked: bool = False

    def to_mapping(self) -> dict[str, str | bool | None]:
        return {
            "route": self.route,
            "source": self.source,
            "fallback_error_type": self.fallback_error_type,
            "guard_reason": self.guard_reason,
            "llm_invoked": self.llm_invoked,
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
    return ServiceRouteDecision(
        route=route,
        source=source,
        fallback_error_type=fallback_error_type,
        guard_reason=guard_reason,
        llm_invoked=llm_invoked,
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
    """Always return a route, falling back to the calibrated deterministic rule."""

    try:
        return request_v19_route(
            service_url,
            query,
            timeout_seconds=timeout_seconds,
        )
    except Exception as exc:
        decision = RuleRouter.v19_calibrated().route(query)
        return ServiceRouteDecision(
            route=decision.route,
            source="rule_fallback",
            fallback_error_type=type(exc).__name__,
            llm_invoked=False,
        )
