from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from ocr_vlm_retrieval.routing.intervention import guard_v18_transition
from ocr_vlm_retrieval.routing.schema import IntentEvidence
from scripts.v18_1_live_search import select_applied_transition


def branches(
    method: str,
    route: str,
    exploratory: bool,
) -> dict[str, bool]:
    del method, exploratory
    if route == "text_evidence":
        return {"text": True, "bm25": True, "visual": False}
    return {"text": True, "bm25": True, "visual": True}


def test_guard_blocks_factual_to_discovery_transition() -> None:
    transition = guard_v18_transition(
        query="找出明确写有金额的页面",
        method="quality_hybrid",
        legacy_route="mixed",
        candidate_route="visual_discovery",
        evidence=None,
        extract_strict_entity_term=lambda query: None,
        required_search_branches=branches,
    )
    assert transition.applied_route == "mixed"
    assert transition.guard_reason == "preserve_factual_acceptance"


def test_guard_rejects_entity_without_strict_term() -> None:
    transition = guard_v18_transition(
        query="帮我找一个可能的人名",
        method="quality_hybrid",
        legacy_route="text_evidence",
        candidate_route="entity_exact",
        evidence=None,
        extract_strict_entity_term=lambda query: None,
        required_search_branches=branches,
    )
    assert transition.applied_route == "text_evidence"
    assert transition.guard_reason == "entity_exact_without_strict_term"


def test_guard_rejects_candidate_that_removes_legacy_branch() -> None:
    evidence = IntentEvidence(
        needs_literal_text=True,
        needs_visual_semantics=False,
        needs_layout_structure=False,
        needs_exact_entity=False,
        needs_topic_discovery=False,
        is_compositional=True,
    )
    transition = guard_v18_transition(
        query="金额和红色印章必须同时满足",
        method="quality_hybrid",
        legacy_route="mixed",
        candidate_route="text_evidence",
        evidence=evidence,
        extract_strict_entity_term=lambda query: None,
        required_search_branches=branches,
    )
    assert transition.applied_route == "mixed"
    assert transition.guard_reason == (
        "compositional_candidate_removes_legacy_branch"
    )


def test_shadow_never_applies_candidate(monkeypatch) -> None:
    monkeypatch.setattr(
        "scripts.v18_1_live_search.live_search.required_search_branches",
        branches,
    )
    transition = select_applied_transition(
        mode="shadow",
        method="quality_hybrid",
        query="找自然景观",
        legacy_route="mixed",
        candidate_route="visual_discovery",
        decision=None,
    )
    assert transition.applied_route == "mixed"
    assert transition.guard_reason == "shadow_only"


def test_guarded_allows_branch_preserving_factual_transition(
    monkeypatch,
) -> None:
    def all_branches(*args: Any) -> dict[str, bool]:
        return {"text": True, "bm25": True, "visual": True}

    monkeypatch.setattr(
        "scripts.v18_1_live_search.live_search.required_search_branches",
        all_branches,
    )
    transition = select_applied_transition(
        mode="guarded",
        method="quality_hybrid",
        query="找右上角带校徽的封面",
        legacy_route="mixed",
        candidate_route="visual_metadata",
        decision=None,
    )
    assert transition.applied_route == "visual_metadata"
    assert transition.guard_reason is None
