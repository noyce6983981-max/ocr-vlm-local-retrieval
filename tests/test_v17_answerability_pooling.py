from __future__ import annotations

import copy
import json

import pytest

from ocr_vlm_retrieval.evaluation.judgments import (
    CORPUS_ANSWERABILITY_TASK,
    CORPUS_NO_ANSWER_AFTER_HIGH_RECALL_POOLING,
    validate_corpus_no_answer_record,
)
from ocr_vlm_retrieval.evaluation.pooling import (
    DEFAULT_HIGH_RECALL_RUN_DEPTHS,
    blind_corpus_answerability_packet,
    build_high_recall_answerability_pool,
)


def ranked_runs() -> dict[str, list[dict[str, object]]]:
    return {
        run_id: [
            {
                "item_id": "shared",
                "image_path": "images/shared.png",
                "title": "secret title",
                "source_relpath": "private/source.pdf",
                "dataset_name": "secret dataset",
                "ocr_text_preview": "secret OCR",
            },
            {"item_id": f"{run_id}_only"},
        ]
        for run_id in DEFAULT_HIGH_RECALL_RUN_DEPTHS
    }


def test_high_recall_pool_uses_declared_depths_and_expansions() -> None:
    pool = build_high_recall_answerability_pool(
        "q1",
        ranked_runs(),
        expansions={
            "same_source": [{"item_id": "same_source_neighbor"}],
            "counterfactual": [{"item_id": "counterfactual_neighbor"}],
        },
    )
    assert pool["task_id"] == CORPUS_ANSWERABILITY_TASK
    assert len(pool["retrieval_runs"]) == len(DEFAULT_HIGH_RECALL_RUN_DEPTHS)
    assert pool["candidate_count"] == (
        1 + len(DEFAULT_HIGH_RECALL_RUN_DEPTHS) + 2
    )
    assert pool["answerability_status"] == "pending_independent_human_review"
    assert pool == build_high_recall_answerability_pool(
        "q1",
        ranked_runs(),
        expansions={
            "same_source": [{"item_id": "same_source_neighbor"}],
            "counterfactual": [{"item_id": "counterfactual_neighbor"}],
        },
    )


def test_blind_answerability_packet_hides_identity_and_source_metadata() -> None:
    audit = build_high_recall_answerability_pool("q1", ranked_runs())
    packet = blind_corpus_answerability_packet(audit)
    serialized = json.dumps(packet)
    assert packet["pool_sha256"] == audit["pool_sha256"]
    assert '"item_id"' not in serialized
    assert "secret title" not in serialized
    assert "private/source.pdf" not in serialized
    assert "secret dataset" not in serialized
    assert "secret OCR" not in serialized
    assert "retrieval_evidence" not in serialized
    assert packet["candidates"][0]["candidate_id"].startswith("C")


def test_corpus_no_answer_requires_full_audit_receipt() -> None:
    audit = build_high_recall_answerability_pool("q1", ranked_runs())
    record = {
        "task_id": CORPUS_ANSWERABILITY_TASK,
        "corpus_answerability": CORPUS_NO_ANSWER_AFTER_HIGH_RECALL_POOLING,
        "pool_sha256": audit["pool_sha256"],
        "retrieval_runs": audit["retrieval_runs"],
        "candidate_count": audit["candidate_count"],
        "reviewer_count": 2,
        "adjudicated": True,
        "known_relevant_outside_pool": False,
    }
    validate_corpus_no_answer_record(record)

    invalid = copy.deepcopy(record)
    invalid["reviewer_count"] = 1
    with pytest.raises(ValueError, match="two independent reviewers"):
        validate_corpus_no_answer_record(invalid)


def test_high_recall_pool_rejects_missing_required_run() -> None:
    runs = ranked_runs()
    runs.pop("candidate_verifier")
    with pytest.raises(ValueError, match="Missing high-recall runs"):
        build_high_recall_answerability_pool("q1", runs)
