"""Safety-first V18.1 intent-routing wrapper around frozen live search."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
PACKAGE_ROOT = PROJECT_ROOT / "src"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from ocr_vlm_retrieval.routing.intervention import (  # noqa: E402
    DISCOVERY_ROUTES,
    INTENT_ROUTING_MODES,
    IntentRoutingMode,
    TransitionDecision,
    guard_v18_transition,
    validate_intent_routing_mode,
)
from ocr_vlm_retrieval.routing.schema import Route, validate_route  # noqa: E402
from ocr_vlm_retrieval.routing.service_client import (  # noqa: E402
    ServiceRouteDecision,
    request_v19_route_with_fallback,
)
from scripts import live_search  # noqa: E402

RETRIEVAL_ROUTES = (
    "text_evidence",
    "visual_discovery",
    "visual_metadata",
    "entity_exact",
    "topic_discovery",
    "mixed",
)


def optional_v19_route(
    *,
    enabled: bool,
    service_url: str,
    query: str,
    timeout_seconds: float,
) -> ServiceRouteDecision | None:
    """Resolve an optional candidate route with frozen-V18 fallback."""

    if not enabled:
        return None
    return request_v19_route_with_fallback(
        service_url,
        query,
        timeout_seconds=timeout_seconds,
    )


def resolve_v18_1_search_intent(
    method: str,
    query: str,
    retrieval_route: str | None = None,
) -> tuple[str, bool, str, str | None]:
    """Apply an external route without modifying the frozen core."""

    if retrieval_route is None:
        return live_search.resolve_search_intent(method, query)
    route = validate_route(retrieval_route)
    strict_entity_term = (
        live_search.extract_strict_entity_term(query)
        if route == "entity_exact"
        else None
    )
    if strict_entity_term is not None:
        return route, False, query, strict_entity_term
    if route == "visual_discovery":
        return route, True, live_search.expand_visual_query(query), None
    if route == "topic_discovery":
        return route, True, query, None
    return route, False, query, None


def select_applied_transition(
    *,
    mode: IntentRoutingMode,
    method: str,
    query: str,
    legacy_route: Route,
    candidate_route: Route,
    decision: ServiceRouteDecision | None,
) -> TransitionDecision:
    """Choose whether a candidate may alter frozen V18 execution."""

    guarded = guard_v18_transition(
        query=query,
        method=method,
        legacy_route=legacy_route,
        candidate_route=candidate_route,
        evidence=decision.evidence if decision is not None else None,
        extract_strict_entity_term=live_search.extract_strict_entity_term,
        required_search_branches=live_search.required_search_branches,
    )
    if mode in {"off", "shadow"}:
        return TransitionDecision(
            candidate_route=candidate_route,
            applied_route=legacy_route,
            guard_reason=(
                "routing_disabled" if mode == "off" else "shadow_only"
            ),
            legacy_branches=guarded.legacy_branches,
            candidate_branches=guarded.candidate_branches,
        )
    if mode == "guarded":
        return guarded
    if (
        candidate_route == "entity_exact"
        and live_search.extract_strict_entity_term(query) is None
    ):
        return TransitionDecision(
            candidate_route=candidate_route,
            applied_route=legacy_route,
            guard_reason="entity_exact_without_strict_term",
            legacy_branches=guarded.legacy_branches,
            candidate_branches=guarded.candidate_branches,
        )
    return TransitionDecision(
        candidate_route=candidate_route,
        applied_route=candidate_route,
        guard_reason=None,
        legacy_branches=guarded.legacy_branches,
        candidate_branches=guarded.candidate_branches,
    )


def policy_snapshot(method: str, route: Route) -> dict[str, Any]:
    exploratory = route in DISCOVERY_ROUTES
    branches = dict(
        live_search.required_search_branches(method, route, exploratory)
    )
    return {
        "route": route,
        "exploratory": exploratory,
        "branches": branches,
        "factual_open_set_gate_eligible": (
            not exploratory
            and all(branches.get(name, False) for name in ("text", "bm25", "visual"))
        ),
    }


def result_snapshot(payload: dict[str, Any], method: str) -> dict[str, Any]:
    rankings = payload.get("rankings", {})
    rows = rankings.get(method, []) if isinstance(rankings, dict) else []
    top1 = rows[0].get("item_id") if rows else None
    acceptance = payload.get("acceptance", {})
    method_acceptance = (
        acceptance.get(method) if isinstance(acceptance, dict) else None
    )
    return {
        "top1_item_id": top1,
        "acceptance": method_acceptance,
        "executed_branches": payload.get("executed_branches"),
    }


def parse_wrapper_args(
    argv: list[str],
) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--enable-v19-intent-routing", action="store_true")
    parser.add_argument(
        "--intent-routing-mode",
        choices=INTENT_ROUTING_MODES,
        default="off",
    )
    parser.add_argument(
        "--v19-intent-routing-url",
        default="http://127.0.0.1:8765",
    )
    parser.add_argument(
        "--v19-intent-timeout-seconds",
        type=float,
        default=2.0,
    )
    parser.add_argument("--retrieval-route", choices=RETRIEVAL_ROUTES)
    return parser.parse_known_args(argv)


def main() -> None:
    wrapper_started = time.perf_counter()
    wrapper_args, live_args = parse_wrapper_args(sys.argv[1:])
    if wrapper_args.v19_intent_timeout_seconds <= 0:
        raise ValueError("--v19-intent-timeout-seconds must be positive")
    mode = validate_intent_routing_mode(wrapper_args.intent_routing_mode)
    if wrapper_args.enable_v19_intent_routing and mode == "off":
        # Backward-compatible alias now chooses the conservative active mode.
        mode = "guarded"

    sys.argv = [sys.argv[0], *live_args]
    standard_args = live_search.parse_args()
    query = " ".join(standard_args.query.split())
    legacy_values = live_search.resolve_search_intent(standard_args.method, query)
    legacy_route = validate_route(legacy_values[0])

    route_started = time.perf_counter()
    decision = optional_v19_route(
        enabled=(mode != "off" and wrapper_args.retrieval_route is None),
        service_url=wrapper_args.v19_intent_routing_url,
        query=query,
        timeout_seconds=wrapper_args.v19_intent_timeout_seconds,
    )
    route_latency_ms = round((time.perf_counter() - route_started) * 1000, 3)
    candidate_route = (
        validate_route(wrapper_args.retrieval_route)
        if wrapper_args.retrieval_route is not None
        else decision.route
        if decision is not None
        else legacy_route
    )
    effective_mode: IntentRoutingMode = "active" if (
        wrapper_args.retrieval_route is not None
    ) else mode
    transition = select_applied_transition(
        mode=effective_mode,
        method=standard_args.method,
        query=query,
        legacy_route=legacy_route,
        candidate_route=candidate_route,
        decision=decision,
    )

    def resolve(method: str, routed_query: str):
        if transition.applied_route == legacy_route:
            return live_search_resolve(method, routed_query)
        return resolve_v18_1_search_intent(
            method,
            routed_query,
            transition.applied_route,
        )

    live_search_resolve = live_search.resolve_search_intent
    original_write = live_search.write_json_atomic

    def audited_write(path: Path, payload: Any) -> None:
        if (
            isinstance(payload, dict)
            and "rankings" in payload
            and "acceptance" in payload
        ):
            legacy_policy = policy_snapshot(standard_args.method, legacy_route)
            candidate_policy = policy_snapshot(
                standard_args.method, candidate_route
            )
            payload["v18_1_intent_routing"] = {
                "routing_mode": effective_mode,
                "service_url": (
                    wrapper_args.v19_intent_routing_url
                    if mode != "off"
                    else None
                ),
                "legacy_route": legacy_route,
                "candidate_route": candidate_route,
                "applied_route": transition.applied_route,
                "transition_guard_reason": transition.guard_reason,
                "route_latency_ms": route_latency_ms,
                "wall_total_seconds": round(
                    time.perf_counter() - wrapper_started, 3
                ),
                "decision": (
                    decision.to_mapping()
                    if decision is not None
                    else {
                        "route": candidate_route,
                        "source": (
                            "explicit_override"
                            if wrapper_args.retrieval_route is not None
                            else "frozen_default"
                        ),
                        "llm_invoked": False,
                    }
                ),
                "policy_diff": {
                    "legacy": legacy_policy,
                    "candidate": candidate_policy,
                    "changes_exploratory_semantics": (
                        legacy_policy["exploratory"]
                        != candidate_policy["exploratory"]
                    ),
                    "removes_legacy_branch": any(
                        transition.legacy_branches.get(branch, False)
                        and not transition.candidate_branches.get(branch, False)
                        for branch in ("text", "bm25", "visual")
                    ),
                },
                "result": result_snapshot(payload, standard_args.method),
            }
        original_write(path, payload)

    live_search.resolve_search_intent = resolve
    live_search.write_json_atomic = audited_write
    live_search.main()


if __name__ == "__main__":
    main()
