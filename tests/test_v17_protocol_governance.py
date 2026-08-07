"""Validate the append-only V17 protocol and active pilot scope."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
V17_ROOT = PROJECT_ROOT / "data/evaluation/v17"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_original_protocol_is_immutable_and_amended() -> None:
    original_path = V17_ROOT / "protocol_frozen_v1.json"
    original = read_json(original_path)
    amendment = read_json(
        V17_ROOT / "amendments/001_scope_changed_to_compositional_80.json"
    )

    assert original["human_evaluation"]["target_query_count"] == 200
    assert sha256(original_path) == amendment["original_protocol"]["sha256"]
    assert amendment["original_protocol"]["immutable"] is True
    assert amendment["effective_scope"] == {
        "query_count": 80,
        "calibration_count": 40,
        "holdout_count": 40,
        "route": "compositional_visual",
        "frozen_query_set_sha256": (
            "5950bd68bf7680aa186132d187ce65741d3b6f4245577a000c00314be8828979"
        ),
        "freeze_receipt": "data/evaluation/v17/frozen_query_set/freeze_receipt.json",
    }
    assert not (V17_ROOT / "research_protocol.json").exists()


def test_study_status_hashes_protocol_chain_and_keeps_holdout_sealed() -> None:
    status = read_json(V17_ROOT / "study_status.json")
    for entry in status["protocol_chain"]:
        assert sha256(PROJECT_ROOT / entry["path"]) == entry["sha256"]

    scope = status["effective_scope"]
    assert scope["query_count"] == 80
    assert scope["calibration_count"] + scope["holdout_count"] == 80
    assert scope["route"] == "compositional_visual"
    assert status["holdout"]["sealed"] is True
    assert status["holdout"]["retrieval_executed"] is False
    assert status["holdout"]["final_method_locked"] is False
    assert status["public_claim_boundary"]["latest_independent_result"] == "V16"


def test_pool_relevance_is_not_promoted_to_corpus_answerability() -> None:
    amendment = read_json(
        V17_ROOT
        / "amendments/003_separate_pooled_relevance_from_corpus_answerability.json"
    )
    status = read_json(V17_ROOT / "study_status.json")
    assert amendment["evaluation_tasks"]["pooled_relevance"][
        "forbidden_claim"
    ] == "no_answer_in_corpus"
    assert status["evaluation_tasks"]["pooled_relevance"][
        "corpus_claim_supported"
    ] is False
    assert status["evaluation_tasks"]["corpus_answerability"][
        "independent_human_review_complete"
    ] is False


def test_parser_reference_is_calibration_only_and_not_human_gold() -> None:
    rows = read_jsonl(V17_ROOT / "parser/calibration_parser_reference.jsonl")

    assert len(rows) == 40
    assert len({row["query_id"] for row in rows}) == 40
    assert {row["split"] for row in rows} == {"calibration"}
    assert {row["provenance"] for row in rows} == {
        "codex_assisted_calibration_reference"
    }


def test_top5_extension_is_small_calibration_only_and_model_assisted() -> None:
    root = V17_ROOT / "human_study/calibration"
    packets = read_jsonl(root / "top5_extension_review_packets.jsonl")
    judgments = read_jsonl(root / "top5_extension_judgments.jsonl")

    assert len(packets) == len(judgments) == 7
    assert {row["query_id"] for row in packets} == {
        row["query_id"] for row in judgments
    }
    assert {row["split"] for row in packets + judgments} == {"calibration"}
    assert all(row["human_gold"] is False for row in judgments)
    assert all(row["reviewer_type"] == "model_assisted" for row in judgments)


def test_topk_selection_is_calibration_only_and_config_is_hashed() -> None:
    amendment = read_json(
        V17_ROOT / "amendments/004_topk_calibration_selects_k3_full_query.json"
    )
    status = read_json(V17_ROOT / "study_status.json")
    selected = amendment["selected_calibration_candidate"]

    assert selected["top_k"] == 3
    assert selected["contrastive_relations"] is False
    assert amendment["scope"]["holdout_read"] is False
    assert amendment["method_lock_status"]["final_method_locked"] is False
    config = selected["config"]
    assert sha256(PROJECT_ROOT / config["path"]) == config["sha256"]
    assert status["calibration"]["current_candidate"]["verified_rank_depth"] == 3
