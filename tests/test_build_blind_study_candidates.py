from __future__ import annotations

import pytest

from scripts.build_blind_study_candidates import (
    blinded_candidate_order,
    pool_payloads,
    ranking_for_method,
)


def payload(method, ids, *, policy=15, low_confidence=False):
    ranking = [
        {"item_id": item_id, "score": 1.0 - index / 10}
        for index, item_id in enumerate(ids, start=1)
    ]
    return {
        "search_policy_version": policy,
        "retrieval_config_revision": "config-v1",
        "library_revision": "library-v1",
        "retrieval_route": "hybrid",
        "query_mode": "semantic",
        "rankings": {method: [] if low_confidence else ranking},
        "low_confidence_rankings": {method: ranking if low_confidence else []},
        "acceptance": {
            "quality_hybrid": {
                "accepted": not low_confidence,
                "reason": "test decision",
            }
        },
    }


def test_rejected_query_uses_preserved_low_confidence_candidates() -> None:
    source = payload(
        "quality_hybrid", ["item_a", "item_b"], low_confidence=True
    )
    assert [
        row["item_id"] for row in ranking_for_method(source, "quality_hybrid")
    ] == ["item_a", "item_b"]


def test_pool_uses_multiple_routes_and_deduplicates_items() -> None:
    payloads = {
        "quality_hybrid": payload("quality_hybrid", ["shared", "hybrid"]),
        "text": payload("text", ["text", "shared"]),
        "visual": payload("visual", ["visual", "shared"]),
    }
    pooled = pool_payloads(
        payloads,
        methods=["quality_hybrid", "text", "visual"],
        top_per_method=2,
        pool_size=4,
    )
    assert pooled["candidate_item_ids"][0] == "shared"
    assert set(pooled["candidate_item_ids"]) == {
        "shared",
        "hybrid",
        "text",
        "visual",
    }
    shared = pooled["candidate_results"][0]
    assert shared["pool_sources"] == ["quality_hybrid", "text", "visual"]
    assert pooled["pool_methods"] == ["quality_hybrid", "text", "visual"]


def test_pool_refuses_mixed_policy_versions() -> None:
    payloads = {
        "quality_hybrid": payload("quality_hybrid", ["a"]),
        "text": payload("text", ["b"], policy=16),
    }
    with pytest.raises(ValueError, match="版本不一致"):
        pool_payloads(
            payloads,
            methods=["quality_hybrid", "text"],
            top_per_method=2,
            pool_size=4,
        )


def test_assessor_order_is_stable_and_not_the_ranked_order() -> None:
    item_ids = [f"item_{index}" for index in range(10)]
    first = blinded_candidate_order(
        item_ids,
        query_id="blind_001",
        query_set_sha256="frozen-digest",
    )
    second = blinded_candidate_order(
        item_ids,
        query_id="blind_001",
        query_set_sha256="frozen-digest",
    )
    assert first == second
    assert set(first) == set(item_ids)
    assert first != item_ids
