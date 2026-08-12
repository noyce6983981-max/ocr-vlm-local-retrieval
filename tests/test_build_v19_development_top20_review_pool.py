from __future__ import annotations

from scripts.build_v19_development_top20_review_pool import (
    build_packets,
    pool_item_ids,
)


def payload(*item_ids: str) -> dict:
    return {
        "rankings": {"quality_hybrid": [{"item_id": item_id} for item_id in item_ids]}
    }


def test_pool_unions_baseline_and_guarded_without_duplicates() -> None:
    assert pool_item_ids(payload("a", "b"), payload("b", "c"), pool_size=20) == [
        "a",
        "b",
        "c",
    ]


def test_packets_are_blinded_and_development_only() -> None:
    assignments = {
        "split": "v19_reviewed_development_only",
        "assignments": [
            {
                "query_id": "q",
                "query": "找页面",
                "query_role": "answerable_positive",
                "content_stratum": "layout_table",
            }
        ],
    }
    packets = build_packets(
        assignments,
        baseline_payloads={"q": payload("a", "b")},
        guarded_payloads={"q": payload("b", "c")},
        manifest={
            "a": {"source_path": "a.jpg"},
            "b": {"source_path": "b.jpg"},
            "c": {"source_path": "c.jpg"},
        },
    )
    assert packets[0]["split"] == "development"
    assert packets[0]["candidate_count"] == 3
    assert "rank" not in str(packets[0]["candidates"])
    assert "score" not in str(packets[0]["candidates"])
