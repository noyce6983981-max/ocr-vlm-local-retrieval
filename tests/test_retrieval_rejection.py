from __future__ import annotations

from scripts.retrieval_rejection import (
    evaluate_ranking_acceptance,
    infer_query_mode,
)


def test_visual_description_query_is_routed_to_visual() -> None:
    assert (
        infer_query_mode("哪张图片展示了海边日落和水面倒影？")
        == "visual"
    )


def test_explicit_text_query_stays_mixed() -> None:
    assert (
        infer_query_mode("哪张图片包含北京理工大学成绩单和课程分数？")
        == "mixed"
    )


def test_low_visual_similarity_is_rejected() -> None:
    decision = evaluate_ranking_acceptance(
        "哪张图片展示了海边日落和水面倒影？",
        "quality_hybrid",
        [
            {
                "raw_visual_score": 0.357169,
                "raw_text_score": 0.2,
                "bm25_raw_score": 0.0,
            },
            {
                "raw_visual_score": 0.342312,
                "raw_text_score": 0.1,
                "bm25_raw_score": 0.0,
            },
        ],
    )
    assert decision["accepted"] is False
    assert decision["threshold"] == 0.45


def test_calibrated_visual_match_is_accepted() -> None:
    decision = evaluate_ranking_acceptance(
        "哪张图片展示蓝天湖泊和建筑？",
        "visual",
        [
            {"raw_visual_score": 0.66},
            {"raw_visual_score": 0.35},
        ],
    )
    assert decision["accepted"] is True
