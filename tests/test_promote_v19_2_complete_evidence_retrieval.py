from __future__ import annotations

from scripts.promote_v19_2_complete_evidence_retrieval import promote_ranking


def test_promote_ranking_preserves_evidence_and_retrieval_order() -> None:
    assert promote_ranking(["a", "b", "c"], ["c", "x"]) == ["c", "x", "a", "b"]
