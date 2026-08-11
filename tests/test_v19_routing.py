from __future__ import annotations

import json

import pytest

from ocr_vlm_retrieval.routing import (
    HybridRouter,
    IntentEvidence,
    IntentSchemaError,
    LLMRouter,
    RuleRouter,
    map_evidence_to_route,
)


class StubBackend:
    def __init__(
        self,
        payload: dict[str, bool | float] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.payload = payload
        self.error = error
        self.call_count = 0

    @property
    def name(self) -> str:
        return "stub-v19"

    def generate_intent_json(self, query: str) -> str:
        assert query
        self.call_count += 1
        if self.error is not None:
            raise self.error
        assert self.payload is not None
        return json.dumps(self.payload)


def evidence_payload(**overrides: bool | float) -> dict[str, bool | float]:
    value: dict[str, bool | float] = {
        "needs_literal_text": False,
        "needs_visual_semantics": False,
        "needs_layout_structure": False,
        "needs_exact_entity": False,
        "needs_topic_discovery": False,
        "is_compositional": False,
    }
    value.update(overrides)
    return value


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (evidence_payload(needs_exact_entity=True), "entity_exact"),
        (
            evidence_payload(
                needs_literal_text=True,
                needs_visual_semantics=True,
                is_compositional=True,
            ),
            "mixed",
        ),
        (evidence_payload(needs_layout_structure=True), "visual_metadata"),
        (evidence_payload(needs_visual_semantics=True), "visual_discovery"),
        (evidence_payload(needs_topic_discovery=True), "topic_discovery"),
        (evidence_payload(needs_literal_text=True), "text_evidence"),
        (evidence_payload(), "mixed"),
    ],
)
def test_deterministic_mapping(
    payload: dict[str, bool | float], expected: str
) -> None:
    assert (
        map_evidence_to_route(IntentEvidence.from_mapping(payload))
        == expected
    )


def test_schema_rejects_missing_unknown_and_non_boolean_fields() -> None:
    missing = evidence_payload()
    del missing["is_compositional"]
    with pytest.raises(IntentSchemaError, match="Missing"):
        IntentEvidence.from_mapping(missing)

    unknown = evidence_payload(extra=True)
    with pytest.raises(IntentSchemaError, match="Unknown"):
        IntentEvidence.from_mapping(unknown)

    invalid = evidence_payload(needs_literal_text=1.0)
    with pytest.raises(IntentSchemaError, match="must be a boolean"):
        IntentEvidence.from_mapping(invalid)


def test_schema_rejects_prose_and_json_arrays() -> None:
    with pytest.raises(IntentSchemaError, match="valid JSON"):
        IntentEvidence.from_json("这里是解释文字")
    with pytest.raises(IntentSchemaError, match="JSON object"):
        IntentEvidence.from_json("[]")


def test_clear_rule_route_does_not_call_backend() -> None:
    backend = StubBackend(payload=evidence_payload())
    router = HybridRouter(
        RuleRouter(lambda query: "text_evidence"),
        LLMRouter(backend),
    )
    decision = router.route("哪页写着会议通知？")
    assert decision.route == "text_evidence"
    assert decision.source == "rule"
    assert backend.call_count == 0


def test_mixed_rule_route_calls_llm_and_ignores_self_confidence() -> None:
    backend = StubBackend(
        payload=evidence_payload(
            needs_visual_semantics=True,
            confidence=0.01,
        )
    )
    router = HybridRouter(
        RuleRouter(lambda query: "mixed"),
        LLMRouter(backend),
    )
    decision = router.route("帮我找一下像公园夜景的那页")
    assert decision.route == "visual_discovery"
    assert decision.source == "llm"
    assert decision.llm is not None
    assert decision.llm.evidence.confidence == 0.01
    assert backend.call_count == 1


@pytest.mark.parametrize(
    "error",
    [TimeoutError("pilot timeout"), RuntimeError("backend unavailable")],
)
def test_backend_errors_fall_back_to_rule(error: Exception) -> None:
    backend = StubBackend(error=error)
    router = HybridRouter(
        RuleRouter(lambda query: "mixed"),
        LLMRouter(backend),
    )
    decision = router.route("一条歧义查询")
    assert decision.route == "mixed"
    assert decision.source == "rule_fallback"
    assert decision.fallback_error_type == type(error).__name__


def test_invalid_model_json_falls_back_to_rule() -> None:
    class InvalidBackend:
        name = "invalid-json"

        def generate_intent_json(self, query: str) -> str:
            return "not-json"

    router = HybridRouter(
        RuleRouter(lambda query: "mixed"),
        LLMRouter(InvalidBackend()),
    )
    decision = router.route("一条歧义查询")
    assert decision.route == "mixed"
    assert decision.source == "rule_fallback"
    assert decision.fallback_error_type == "IntentSchemaError"


def test_guard_preserves_visual_route_when_llm_invents_literal_need() -> None:
    backend = StubBackend(payload=evidence_payload(needs_literal_text=True))
    router = HybridRouter(
        RuleRouter(
            lambda query: "visual_discovery",
            lambda query, route: ("test_ambiguity",),
        ),
        LLMRouter(backend),
    )
    decision = router.route("白色列车在铁路桥上行驶")
    assert decision.route == "visual_discovery"
    assert decision.source == "rule_guard"
    assert decision.guard_reason == (
        "preserve_visual_route_without_literal_requirement"
    )


def test_guard_allows_text_route_for_explicit_receipt_total() -> None:
    backend = StubBackend(payload=evidence_payload(needs_literal_text=True))
    router = HybridRouter(
        RuleRouter(lambda query: "mixed"),
        LLMRouter(backend),
    )
    decision = router.route("一张小票，文字总额为174600")
    assert decision.route == "text_evidence"
    assert decision.source == "llm"


def test_guard_preserves_visual_route_from_unneeded_mixed_expansion() -> None:
    backend = StubBackend(
        payload=evidence_payload(
            needs_literal_text=True,
            needs_visual_semantics=True,
            is_compositional=True,
        )
    )
    router = HybridRouter(
        RuleRouter(
            lambda query: "visual_discovery",
            lambda query, route: ("test_ambiguity",),
        ),
        LLMRouter(backend),
    )
    decision = router.route("银色电脑屏幕显示森林，右侧放着光盘")
    assert decision.route == "visual_discovery"
    assert decision.source == "rule_guard"


def test_guard_rejects_entity_route_for_long_compound_description() -> None:
    backend = StubBackend(payload=evidence_payload(needs_exact_entity=True))
    router = HybridRouter(
        RuleRouter(lambda query: "mixed"),
        LLMRouter(backend),
    )
    decision = router.route(
        "一张员工年休假申请表，申请人为桑新平，户籍所在地为四川省绵阳市"
    )
    assert decision.route == "mixed"
    assert decision.source == "rule_guard"
    assert decision.guard_reason == "reject_entity_route_for_long_compound_query"


def test_legacy_rule_router_adapter_is_available() -> None:
    decision = RuleRouter.legacy().route("找一张蓝色海边照片")
    assert decision.route == "visual_discovery"
    assert decision.ambiguous is False


def test_legacy_gate_escalates_missed_natural_person_lookup() -> None:
    router = RuleRouter.legacy()
    decision = router.route("材料里有没有林雨琪？")
    assert decision.route != "entity_exact"
    assert decision.ambiguous is True
    assert "probable_person_lookup_missed_by_rule" in decision.reason_codes


def test_legacy_gate_keeps_clear_topic_query_on_rules() -> None:
    decision = RuleRouter.legacy().route("整理一下与储能安全相关的材料")
    assert decision.ambiguous is False


def test_legacy_gate_escalates_explicit_compound_query_missed_by_rule() -> None:
    decision = RuleRouter.legacy().route("哪份申请表右下角盖了红色印章？")
    assert decision.route != "mixed"
    assert decision.ambiguous is True
    assert "explicit_compound_evidence_missed_by_rule" in decision.reason_codes


def test_legacy_gate_keeps_explicit_compound_rule_route() -> None:
    decision = RuleRouter.legacy().route(
        "我记得山景照片里的路牌写着海拔三千米。"
    )
    assert decision.route == "mixed"
    assert decision.ambiguous is False


def test_legacy_gate_does_not_escalate_text_locator_only_query() -> None:
    decision = RuleRouter.legacy().route(
        "标题写着“设备采购清单”的页面在哪里？"
    )
    assert decision.route == "text_evidence"
    assert decision.ambiguous is False


def test_v19_calibrated_gate_recovers_short_person_lookup() -> None:
    decision = RuleRouter.v19_calibrated().route("材料里有没有梁思远？")
    assert decision.route == "entity_exact"
    assert decision.ambiguous is False


def test_v19_calibrated_gate_recovers_short_identifier_lookup() -> None:
    decision = RuleRouter.v19_calibrated().route("材料中有没有ZX-410？")
    assert decision.route == "entity_exact"
    assert decision.ambiguous is False


def test_v19_calibrated_gate_recovers_pure_visual_scene() -> None:
    decision = RuleRouter.v19_calibrated().route(
        "我记得一张图中厨师正把锅抛起，旁边有火苗。"
    )
    assert decision.route == "visual_discovery"
    assert decision.ambiguous is False


def test_v19_calibrated_gate_keeps_visual_text_compound_ambiguous() -> None:
    decision = RuleRouter.v19_calibrated().route(
        "找蓝色瓶子且标签写着丙酮的照片。"
    )
    assert decision.route != "visual_discovery"
    assert decision.ambiguous is True
    assert "explicit_compound_evidence_missed_by_rule" in decision.reason_codes
