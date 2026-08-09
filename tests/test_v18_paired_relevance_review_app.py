from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts/v18_paired_relevance_review_app.py"
    )
    spec = importlib.util.spec_from_file_location(
        "v18_paired_relevance_review_app", path
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _packet(group: int, query: int, item_ids: list[str]) -> dict[str, object]:
    return {
        "group_id": f"g{group:02d}",
        "query_id": f"q{query:02d}",
        "study_fingerprint": "study",
        "candidates": [
            {"item_id": item_id, "candidate_id": item_id, "review_asset": {}}
            for item_id in item_ids
        ],
    }


def test_packet_pairs_turn_eighty_queries_into_forty_tasks() -> None:
    module = _load_module()
    packets = [
        _packet(group, query, [f"item-{group}"])
        for group in range(40)
        for query in (group * 2, group * 2 + 1)
    ]
    assert len(module.packet_pairs(packets)) == 40


def test_packet_pairs_rejects_an_unpaired_group() -> None:
    module = _load_module()
    with pytest.raises(ValueError, match="exactly two"):
        module.packet_pairs([_packet(1, 1, ["a"])])


def test_union_candidates_renders_shared_items_once() -> None:
    module = _load_module()
    pair = (
        _packet(1, 1, ["a", "b"]),
        _packet(1, 2, ["b", "c"]),
    )
    assert [
        row["item_id"] for row in module.union_candidates(pair)
    ] == ["a", "b", "c"]


def test_secondary_assignment_reviews_twelve_complete_pairs() -> None:
    module = _load_module()
    pairs = [
        (
            _packet(group, group * 2, ["a"]),
            _packet(group, group * 2 + 1, ["b"]),
        )
        for group in range(40)
    ]
    selected = module.secondary_pairs(pairs)
    assert len(selected) == 12
    assert all(first["group_id"] == second["group_id"] for first, second in selected)


def test_adjudicator_assignment_reviews_every_conflict_pair() -> None:
    module = _load_module()
    pairs = [
        (_packet(group, group * 2, ["a"]), _packet(group, group * 2 + 1, ["b"]))
        for group in range(3)
    ]
    assert module.assigned_pairs(pairs, "adjudicator") == pairs
