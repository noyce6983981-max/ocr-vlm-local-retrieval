from __future__ import annotations

import json
import sys
from http.server import ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from urllib import error, request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from ocr_vlm_retrieval.routing import HybridRouter, LLMRouter, RuleRouter
from ocr_vlm_retrieval.routing.service import route_payload
from ocr_vlm_retrieval.routing.service_client import (
    _validated_service_decision,
    request_v19_route,
    request_v19_route_with_fallback,
)
from scripts.v19_intent_server import handler_for
from scripts.v18_1_live_search import (
    optional_v19_route,
    resolve_v18_1_search_intent,
)


class VisualBackend:
    name = "stub-visual"

    def generate_intent_json(self, query: str) -> str:
        return json.dumps(
            {
                "needs_literal_text": False,
                "needs_visual_semantics": True,
                "needs_layout_structure": False,
                "needs_exact_entity": False,
                "needs_topic_discovery": False,
                "is_compositional": False,
            }
        )


class FakeResponse:
    def __init__(self, payload: object) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def test_service_payload_keeps_route_provenance() -> None:
    router = HybridRouter(
        RuleRouter(lambda query: "mixed"),
        LLMRouter(VisualBackend()),
    )
    payload = route_payload(router, "找一张适合作为封面的城市夜景")
    assert payload["route"] == "visual_discovery"
    assert payload["source"] == "llm"
    assert payload["llm_invoked"] is True
    assert payload["rule_route"] == "mixed"
    assert isinstance(payload["route_latency_ms"], float)


def test_service_response_contract_rejects_unknown_source() -> None:
    try:
        _validated_service_decision(
            {"route": "mixed", "source": "model_magic"}
        )
    except ValueError as exc:
        assert "invalid source" in str(exc)
    else:
        raise AssertionError("invalid source was accepted")


def test_client_normalizes_query_and_validates_success(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def respond(http_request: object, timeout: float) -> FakeResponse:
        captured["request"] = http_request
        captured["timeout"] = timeout
        return FakeResponse(
            {
                "route": "visual_discovery",
                "source": "llm",
                "llm_invoked": True,
                "fallback_error_type": None,
                "guard_reason": None,
                "rule_route": "mixed",
                "rule_reason_codes": ["rule_route_mixed"],
                "backend_name": "stub-visual",
                "evidence": {
                    "needs_literal_text": False,
                    "needs_visual_semantics": True,
                    "needs_layout_structure": False,
                    "needs_exact_entity": False,
                    "needs_topic_discovery": False,
                    "is_compositional": False,
                },
                "route_latency_ms": 12.5,
            }
        )

    monkeypatch.setattr(
        "ocr_vlm_retrieval.routing.service_client.request.urlopen",
        respond,
    )
    decision = request_v19_route(
        "http://127.0.0.1:8765/",
        "  城市   夜景  ",
        timeout_seconds=1.25,
    )
    assert decision.route == "visual_discovery"
    assert decision.llm_invoked is True
    assert decision.rule_route == "mixed"
    assert decision.rule_reason_codes == ("rule_route_mixed",)
    assert decision.backend_name == "stub-visual"
    assert decision.evidence is not None
    assert decision.route_latency_ms == 12.5
    assert captured["timeout"] == 1.25
    http_request = captured["request"]
    assert isinstance(http_request, request.Request)
    assert http_request.full_url.endswith("/route")
    assert http_request.data is not None
    assert json.loads(http_request.data.decode("utf-8")) == {
        "query": "城市 夜景"
    }


def test_client_decision_mapping_keeps_audit_fields() -> None:
    decision = _validated_service_decision(
        {
            "route": "mixed",
            "source": "rule_guard",
            "llm_invoked": True,
            "fallback_error_type": None,
            "guard_reason": "preserve_visual_route",
        }
    )
    assert decision.to_mapping() == {
        "route": "mixed",
        "source": "rule_guard",
        "fallback_error_type": None,
        "guard_reason": "preserve_visual_route",
        "llm_invoked": True,
        "rule_route": None,
        "rule_reason_codes": [],
        "backend_name": None,
        "evidence": None,
        "route_latency_ms": None,
    }


def test_loopback_server_and_client_complete_real_http_round_trip() -> None:
    router = HybridRouter(
        RuleRouter(lambda query: "mixed"),
        LLMRouter(VisualBackend()),
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(router))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        with request.urlopen(f"http://{host}:{port}/health", timeout=1) as response:
            health = json.loads(response.read().decode("utf-8"))
        decision = request_v19_route(
            f"http://{host}:{port}",
            "找一张城市夜景",
            timeout_seconds=1,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert health == {"status": "ok", "feature": "v19_intent_routing"}
    assert decision.route == "visual_discovery"
    assert decision.source == "llm"
    assert decision.llm_invoked is True


def test_service_failure_falls_back_to_frozen_legacy_rule(monkeypatch) -> None:
    def unavailable(*args: object, **kwargs: object) -> object:
        raise error.URLError("offline")

    monkeypatch.setattr(
        "ocr_vlm_retrieval.routing.service_client.request.urlopen",
        unavailable,
    )
    query = "材料中有没有ZX-410？"
    decision = request_v19_route_with_fallback(
        "http://127.0.0.1:1",
        query,
        timeout_seconds=0.01,
    )
    legacy = RuleRouter.legacy().route(query)
    assert decision.route == legacy.route
    assert decision.rule_route == legacy.route
    assert decision.source == "rule_fallback"
    assert decision.fallback_error_type == "ConnectionError"


def test_server_side_rule_fallback_is_also_restored_to_frozen_v18(
    monkeypatch,
) -> None:
    query = "材料中有没有ZX-410？"

    def respond(http_request: object, timeout: float) -> FakeResponse:
        del http_request, timeout
        return FakeResponse(
            {
                "route": "entity_exact",
                "source": "rule_fallback",
                "llm_invoked": True,
                "fallback_error_type": "RuntimeError",
                "rule_route": "entity_exact",
                "rule_reason_codes": ["calibrated_recovery"],
                "backend_name": "stub",
                "evidence": None,
                "route_latency_ms": 20.0,
            }
        )

    monkeypatch.setattr(
        "ocr_vlm_retrieval.routing.service_client.request.urlopen",
        respond,
    )
    decision = request_v19_route_with_fallback(
        "http://127.0.0.1:8765",
        query,
    )
    legacy = RuleRouter.legacy().route(query)
    assert decision.route == legacy.route
    assert decision.rule_route == legacy.route
    assert decision.guard_reason == "restore_frozen_v18_after_service_fallback"
    assert decision.llm_invoked is True


def test_live_search_v19_flag_is_disabled_by_default() -> None:
    assert (
        optional_v19_route(
            enabled=False,
            service_url="http://127.0.0.1:1",
            query="任意查询",
            timeout_seconds=0.01,
        )
        is None
    )


def test_v18_1_wrapper_injects_route_without_mutating_frozen_core() -> None:
    route, exploratory, visual_query, strict_entity = (
        resolve_v18_1_search_intent(
            "quality_hybrid",
            "找一张城市夜景",
            "visual_discovery",
        )
    )
    assert route == "visual_discovery"
    assert exploratory is True
    assert visual_query
    assert strict_entity is None
