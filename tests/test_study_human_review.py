from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from ocr_vlm_retrieval.studies.human_review import (
    build_blind_review_packet,
    validate_independent_candidate_reviews,
)


def _packet() -> dict[str, object]:
    return build_blind_review_packet(
        query_id="q1",
        query="Find the red sign beside a car",
        candidates=[
            {"item_id": "i1", "image_path": "images/i1.jpg"},
            {
                "item_id": "i2",
                "image_path": "images/i2.jpg",
                "ocr_evidence": "SALE",
            },
        ],
        study_fingerprint="f" * 64,
        split="calibration",
    )


def test_blind_packet_contains_only_safe_review_assets() -> None:
    packet = _packet()
    assert packet["candidates"][0] == {
        "candidate_id": "C01",
        "item_id": "i1",
        "review_asset": {"image_path": "images/i1.jpg"},
    }
    assert len(str(packet["pool_sha256"])) == 64
    with pytest.raises(ValueError, match="blind fields"):
        build_blind_review_packet(
            query_id="q1",
            query="query",
            candidates=[
                {
                    "item_id": "i1",
                    "image_path": "image.jpg",
                    "score": 0.9,
                }
            ],
            study_fingerprint="f" * 64,
            split="holdout",
        )


def test_two_reviewers_must_each_cover_the_exact_candidate_pool() -> None:
    packet = _packet()
    judgments = [
        {
            "query_id": "q1",
            "reviewer_id": reviewer,
            "candidate_relevance": {"i1": True, "i2": False},
        }
        for reviewer in ("human_a", "human_b")
    ]
    assert validate_independent_candidate_reviews([packet], judgments) == {
        "q1": ("human_a", "human_b")
    }
    judgments[1]["candidate_relevance"] = {"i1": True}
    with pytest.raises(ValueError, match="coverage mismatch"):
        validate_independent_candidate_reviews([packet], judgments)


def test_review_coverage_does_not_hide_missing_second_reviewer() -> None:
    with pytest.raises(ValueError, match="needs 2 reviewers"):
        validate_independent_candidate_reviews(
            [_packet()],
            [
                {
                    "query_id": "q1",
                    "reviewer_id": "human_a",
                    "candidate_relevance": {"i1": True, "i2": False},
                }
            ],
        )
