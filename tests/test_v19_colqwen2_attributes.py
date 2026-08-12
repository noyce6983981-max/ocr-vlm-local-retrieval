from __future__ import annotations

import pytest

from scripts.score_v19_colqwen2_attributes_development import full_score_index


def test_full_score_index_uses_query_candidate_pair() -> None:
    payload = {
        "results": [
            {
                "query_id": "q",
                "candidate_item_ids": ["a", "b"],
                "scores": [0.1, 0.2],
            }
        ]
    }
    assert full_score_index(payload) == {("q", "a"): 0.1, ("q", "b"): 0.2}


def test_full_score_index_rejects_misaligned_payload() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        full_score_index(
            {
                "results": [
                    {
                        "query_id": "q",
                        "candidate_item_ids": ["a"],
                        "scores": [],
                    }
                ]
            }
        )
