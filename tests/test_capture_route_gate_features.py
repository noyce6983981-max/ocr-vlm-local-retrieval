from scripts.capture_route_gate_features import summarize


def test_summarize_keeps_aligned_route_gate_evidence() -> None:
    query = {
        "query_id": "q1",
        "query": "找年度报告",
        "is_no_answer": "false",
        "relevant_item_ids": "item_b",
        "reviewer_type": "human",
    }
    payload = {
        "retrieval_route": "text_evidence",
        "search_policy_version": 15,
        "acceptance": {"quality_hybrid": {"accepted": True}},
        "rankings": {
            "quality_hybrid": [
                {
                    "item_id": "item_a",
                    "raw_text_score": 0.51,
                    "raw_visual_score": 0.2,
                    "bm25_raw_score": 8.0,
                    "raw_metadata_score": 0.4,
                },
                {
                    "item_id": "item_b",
                    "raw_text_score": 0.49,
                    "raw_visual_score": 0.5,
                    "bm25_raw_score": 2.0,
                    "raw_metadata_score": 0.6,
                },
            ]
        },
        "exact_topic_evidence_item_ids": [],
        "composite_visual_query": False,
        "color_intent": None,
    }

    row = summarize(query, payload)

    assert row["relevant_rank"] == 2
    assert row["relevant_rank_text"] == ""
    assert row["relevant_rank_rrf"] == ""
    assert row["max10_raw_text"] == 0.51
    assert row["max10_raw_visual"] == 0.5
    assert row["aligned_text_bm25_count"] == 1
    assert row["aligned_visual_text_count"] == 0
