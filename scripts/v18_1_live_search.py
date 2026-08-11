"""Opt-in V18.1 intent-routing wrapper around the frozen live-search core."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
PACKAGE_ROOT = PROJECT_ROOT / "src"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

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
    """Resolve the optional service route with deterministic fallback."""

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
    """Apply an external route without modifying the frozen V16/V17 core."""

    if retrieval_route is None:
        return live_search.resolve_search_intent(method, query)
    if retrieval_route not in RETRIEVAL_ROUTES:
        raise ValueError(f"Unsupported retrieval route: {retrieval_route}")
    strict_entity_term = (
        live_search.extract_strict_entity_term(query)
        if retrieval_route == "entity_exact"
        else None
    )
    if strict_entity_term is not None:
        return retrieval_route, False, query, strict_entity_term
    if retrieval_route == "visual_discovery":
        return retrieval_route, True, live_search.expand_visual_query(query), None
    if retrieval_route == "topic_discovery":
        return retrieval_route, True, query, None
    return retrieval_route, False, query, None


def parse_wrapper_args(
    argv: list[str],
) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--enable-v19-intent-routing", action="store_true")
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
    wrapper_args, live_args = parse_wrapper_args(sys.argv[1:])
    if wrapper_args.v19_intent_timeout_seconds <= 0:
        raise ValueError("--v19-intent-timeout-seconds must be positive")

    sys.argv = [sys.argv[0], *live_args]
    standard_args = live_search.parse_args()
    query = " ".join(standard_args.query.split())
    decision = optional_v19_route(
        enabled=(
            wrapper_args.enable_v19_intent_routing
            and wrapper_args.retrieval_route is None
        ),
        service_url=wrapper_args.v19_intent_routing_url,
        query=query,
        timeout_seconds=wrapper_args.v19_intent_timeout_seconds,
    )
    route_override = (
        decision.route
        if decision is not None
        else wrapper_args.retrieval_route
    )

    def resolve(method: str, routed_query: str):
        return resolve_v18_1_search_intent(
            method,
            routed_query,
            route_override,
        )

    original_write = live_search.write_json_atomic

    def audited_write(path: Path, payload: Any) -> None:
        if (
            isinstance(payload, dict)
            and "rankings" in payload
            and "acceptance" in payload
        ):
            payload["v18_1_intent_routing"] = {
                "enabled": wrapper_args.enable_v19_intent_routing,
                "service_url": (
                    wrapper_args.v19_intent_routing_url
                    if wrapper_args.enable_v19_intent_routing
                    else None
                ),
                "decision": (
                    decision.to_mapping()
                    if decision is not None
                    else {
                        "route": route_override,
                        "source": (
                            "explicit_override"
                            if route_override is not None
                            else "frozen_default"
                        ),
                        "llm_invoked": False,
                    }
                ),
            }
        original_write(path, payload)

    live_search.resolve_search_intent = resolve
    live_search.write_json_atomic = audited_write
    live_search.main()


if __name__ == "__main__":
    main()
