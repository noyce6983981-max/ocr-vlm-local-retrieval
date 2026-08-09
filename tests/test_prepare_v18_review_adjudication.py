from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts/prepare_v18_review_adjudication.py"
    )
    spec = importlib.util.spec_from_file_location(
        "prepare_v18_review_adjudication", path
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _packet(query_id: str, group_id: str) -> dict[str, object]:
    return {"query_id": query_id, "group_id": group_id, "split": "holdout"}


def _judgment(
    query_id: str, reviewer_id: str, reviewer_role: str, *, relevant: bool
) -> dict[str, object]:
    return {
        "query_id": query_id,
        "reviewer_id": reviewer_id,
        "reviewer_role": reviewer_role,
        "pool_relevance": (
            "relevant_candidate_in_pool"
            if relevant
            else "no_relevant_candidate_in_pool"
        ),
        "candidate_relevance": {"item": relevant},
    }


def test_conflict_packet_builder_preserves_complete_pairs() -> None:
    module = _load_module()
    packets = [
        _packet("q1", "g1"),
        _packet("q2", "g1"),
        _packet("q3", "g2"),
        _packet("q4", "g2"),
    ]
    primary = [
        _judgment(qid, "r1", "primary", relevant=True)
        for qid in ("q1", "q2", "q3", "q4")
    ]
    secondary = [
        _judgment(qid, "r2", "secondary", relevant=qid != "q1")
        for qid in ("q1", "q2")
    ]
    selected, report = module.build_adjudication_packets(
        packets, primary, secondary, expected_split="holdout"
    )
    assert [row["query_id"] for row in selected] == ["q1", "q2"]
    assert report["conflict_query_ids"] == ["q1"]
    assert report["adjudication_pair_count"] == 1
