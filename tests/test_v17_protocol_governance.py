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
    assert status["holdout"]["retrieval_executed"] is True
    assert status["holdout"]["human_relevance_judgments_complete"] is True
    assert status["holdout"]["model_assisted_labels_used"] is False
    assert status["holdout"]["final_method_locked"] is True
    assert status["holdout"]["evaluation_authorized"] is True
    assert status["holdout"]["one_shot_receipt_exists"] is True
    assert status["holdout"]["scored"] is True
    assert status["holdout"]["rerun_forbidden"] is True
    assert status["public_claim_boundary"]["latest_independent_result"] == (
        "V17 pooled-relevance holdout"
    )


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


def test_final_method_lock_does_not_claim_a_holdout_result() -> None:
    original_lock_amendment = read_json(
        V17_ROOT / "amendments/005_method_locked_holdout_execution_guarded.json"
    )
    ci_amendment = read_json(
        V17_ROOT / "amendments/006_pre_activation_ci_format_normalization.json"
    )
    amendment = read_json(
        V17_ROOT / "amendments/007_pre_evaluation_governance_correction.json"
    )
    effective = amendment["effective_lock_snapshot"]
    lock_path = PROJECT_ROOT / effective["path"]
    lock = read_json(lock_path)

    assert sha256(lock_path) == effective["sha256"]
    assert effective["lock_revision"] == 3
    assert original_lock_amendment["method_lock"]["sha256"] == (
        ci_amendment["superseded_lock_snapshot"]["sha256"]
    )
    assert ci_amendment["effective_lock_snapshot"]["sha256"] == amendment[
        "superseded_lock_snapshot"
    ]["sha256"]
    assert lock["status"] == "method_locked_holdout_sealed"
    assert lock["holdout"]["status"] == (
        "retrieval_and_blinded_human_review_complete_not_evaluated"
    )
    for relative, expected in {
        **lock["locked_files"],
        **lock["retrieval_dependency_files"],
        **lock["governance_files"],
    }.items():
        assert sha256(PROJECT_ROOT / relative) == expected
    assert lock["inference"]["top_k"] == 3
    assert lock["aggregation"] == {
        "method": "full_query",
        "threshold": 0.51,
        "selection_policy": "highest_retrieval_rank_among_passed",
        "contrastive_relations": False,
        "relation_margin_threshold": 0.0,
    }
    assert original_lock_amendment["execution_authorization"]["current_status"] == (
        "not_authorized"
    )
    assert original_lock_amendment["one_shot_guard"][
        "current_receipt_exists"
    ] is False
    assert amendment["holdout_safeguards"][
        "holdout_metrics_computed_before_revision"
    ] is False


def test_final_holdout_summary_is_one_shot_and_claim_bounded() -> None:
    status = read_json(V17_ROOT / "study_status.json")
    summary_path = PROJECT_ROOT / status["holdout"]["summary_path"]
    summary = read_json(summary_path)

    assert sha256(summary_path) == status["holdout"]["summary_sha256"]
    assert summary["status"] == "v17_holdout_evaluated_once"
    assert summary["locked_method"]["method_tuned_on_holdout"] is False
    assert summary["review_protocol"]["model_assisted_labels_used"] is False
    assert summary["rerun_policy"] == "forbidden"
    difference = summary["metrics"]["end_to_end_paired_difference"]
    interval = summary["metrics"]["end_to_end_group_bootstrap_95_ci"]
    assert difference == 0.32499999999999996
    assert interval[0] > 0
    false_accept_interval = summary["metrics"][
        "false_accept_group_bootstrap_95_ci"
    ]
    assert false_accept_interval[0] < 0 < false_accept_interval[1]
    assert "not a corpus-level no-answer estimate" in summary["claim_boundary"]
