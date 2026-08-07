from __future__ import annotations

from datetime import datetime, timezone

import pytest

from scripts.blind_query_study import (
    candidates_path,
    delete_submission,
    evaluator_query_set_sha256,
    freeze_study,
    initialize_study,
    load_protocol,
    materialize_formal_rows,
    progress_summary,
    read_candidates,
    read_submissions,
    replace_collecting_queries,
    save_candidate,
    save_relevance_review,
    submit_query,
    update_submission_expectation,
    verify_frozen_snapshot,
    write_formal_query_set,
    write_live_evaluation_protocol,
)
from scripts.evaluate_live_search_blind import query_set_fingerprint


NOW = datetime(2026, 8, 1, 3, 0, tzinfo=timezone.utc)
REVISION = "library-revision-v1"


def create_study(tmp_path):
    study_dir = tmp_path / "blind_v1"
    initialize_study(
        study_dir,
        study_id="blind_v1",
        library_id="library_a",
        library_revision=REVISION,
        minimum_count=2,
        target_count=3,
        maximum_count=5,
        minimum_no_answer_probes=0,
        now=NOW,
    )
    return study_dir


def submit(study_dir, query, expectation="unsure"):
    return submit_query(
        study_dir,
        query=query,
        expected_answerability=expectation,
        current_library_revision=REVISION,
        now=NOW,
    )


def candidate(query_id, *, policy=15):
    return {
        "query_id": query_id,
        "candidate_item_ids": ["item_a", "item_b"],
        "candidate_results": [
            {"item_id": "item_a", "pool_sources": ["quality_hybrid"]},
            {"item_id": "item_b", "pool_sources": ["visual"]},
        ],
        "system_accepted": True,
        "acceptance_reason": "accepted",
        "retrieval_route": "hybrid",
        "search_policy_version": policy,
        "retrieval_config_revision": "config-v1",
        "pool_methods": ["quality_hybrid", "text", "visual"],
    }


def test_collection_normalizes_rejects_duplicates_and_never_builds_candidates(
    tmp_path,
) -> None:
    study_dir = create_study(tmp_path)
    first = submit(study_dir, "  帮我找   蓝色汽车  ", "answerable")
    assert first["query"] == "帮我找 蓝色汽车"
    assert first["query_id"] == "blind_v1_001"
    with pytest.raises(ValueError, match="已经提交"):
        submit(study_dir, "帮我找 蓝色汽车")
    assert not candidates_path(study_dir).exists()
    assert progress_summary(study_dir)["expectations"]["answerable"] == 1


def test_collection_is_bound_to_unchanged_library(tmp_path) -> None:
    study_dir = create_study(tmp_path)
    with pytest.raises(ValueError, match="资料库内容已变化"):
        submit_query(
            study_dir,
            query="会议签到表",
            current_library_revision="changed",
        )


def test_delete_keeps_audit_ids_monotonic(tmp_path) -> None:
    study_dir = create_study(tmp_path)
    submit(study_dir, "第一条查询")
    second = submit(study_dir, "第二条查询")
    submit(study_dir, "第三条查询")
    delete_submission(
        study_dir,
        second["query_id"],
        current_library_revision=REVISION,
    )
    fourth = submit(study_dir, "替换后的查询")
    assert fourth["query_id"] == "blind_v1_004"
    assert len({row["query_id"] for row in read_submissions(study_dir)}) == 3


def test_expectation_metadata_can_only_change_before_freeze(tmp_path) -> None:
    study_dir = create_study(tmp_path)
    first = submit(study_dir, "实时天气查询")
    submit(study_dir, "查询二号")
    submit(study_dir, "查询三号")
    update_submission_expectation(
        study_dir,
        first["query_id"],
        "no_answer_probe",
        current_library_revision=REVISION,
    )
    assert read_submissions(study_dir)[0]["expected_answerability"] == (
        "no_answer_probe"
    )
    freeze_study(study_dir, current_library_revision=REVISION, now=NOW)
    with pytest.raises(ValueError, match="已冻结"):
        update_submission_expectation(
            study_dir,
            first["query_id"],
            "unsure",
            current_library_revision=REVISION,
        )


def test_replacement_archives_user_rows_and_marks_diagnostic_origin(
    tmp_path,
) -> None:
    study_dir = create_study(tmp_path)
    submit(study_dir, "原来的用户查询")
    archive = tmp_path / "archive" / "original.csv"
    rows = replace_collecting_queries(
        study_dir,
        [
            {"query": "查找年度报告", "expected_answerability": "unsure"},
            {
                "query": "找宠物戴红帽子的照片",
                "expected_answerability": "no_answer_probe",
            },
            {"query": "查找软件截图", "expected_answerability": "unsure"},
        ],
        query_origin="assistant_generated",
        current_library_revision=REVISION,
        archive_path=archive,
        now=NOW,
    )
    assert len(rows) == 3
    assert archive.is_file()
    assert "原来的用户查询" in archive.read_text(encoding="utf-8-sig")
    protocol = load_protocol(study_dir)
    assert protocol["query_origin"] == "assistant_generated"
    assert protocol["study_type"] == "assistant_generated_diagnostic"


def test_freeze_requires_target_and_detects_snapshot_tampering(tmp_path) -> None:
    study_dir = create_study(tmp_path)
    submit(study_dir, "校园里的山水照片")
    submit(study_dir, "带有电话号码的表格")
    with pytest.raises(ValueError, match="至少需要 3 条"):
        freeze_study(
            study_dir,
            current_library_revision=REVISION,
            now=NOW,
        )
    submit(study_dir, "库里应该没有的火星车照片", "no_answer_probe")
    protocol = freeze_study(
        study_dir,
        current_library_revision=REVISION,
        now=NOW,
    )
    assert protocol["status"] == "frozen"
    assert len(verify_frozen_snapshot(study_dir)) == 3
    frozen_path = study_dir / "frozen_queries.csv"
    frozen_path.write_text(
        frozen_path.read_text(encoding="utf-8-sig").replace("山水", "海边"),
        encoding="utf-8-sig",
    )
    with pytest.raises(ValueError, match="指纹不匹配"):
        verify_frozen_snapshot(study_dir)


def test_frozen_study_rejects_new_query(tmp_path) -> None:
    study_dir = create_study(tmp_path)
    for query in ("查询一号", "查询二号", "查询三号"):
        submit(study_dir, query)
    freeze_study(study_dir, current_library_revision=REVISION, now=NOW)
    with pytest.raises(ValueError, match="已冻结"):
        submit(study_dir, "查询四号")


def test_candidate_pool_is_resumable_and_policy_locked(tmp_path) -> None:
    study_dir = create_study(tmp_path)
    rows = [submit(study_dir, query) for query in ("查询一号", "查询二号", "查询三号")]
    freeze_study(study_dir, current_library_revision=REVISION, now=NOW)
    save_candidate(
        study_dir,
        candidate(rows[0]["query_id"]),
        current_library_revision=REVISION,
        now=NOW,
    )
    assert list(read_candidates(study_dir)) == [rows[0]["query_id"]]
    assert load_protocol(study_dir)["status"] == "reviewing"
    with pytest.raises(ValueError, match="策略"):
        save_candidate(
            study_dir,
            candidate(rows[1]["query_id"], policy=16),
            current_library_revision=REVISION,
            now=NOW,
        )


def test_reviews_require_generated_candidates_and_known_items(tmp_path) -> None:
    study_dir = create_study(tmp_path)
    rows = [submit(study_dir, query) for query in ("查询一号", "查询二号", "查询三号")]
    freeze_study(study_dir, current_library_revision=REVISION, now=NOW)
    save_candidate(
        study_dir,
        candidate(rows[0]["query_id"]),
        current_library_revision=REVISION,
        now=NOW,
    )
    with pytest.raises(ValueError, match="尚未生成"):
        save_relevance_review(
            study_dir,
            {"query_id": rows[1]["query_id"], "decision": "no_answer"},
            known_item_ids={"item_a", "item_b"},
        )
    with pytest.raises(ValueError, match="未知图片ID"):
        save_relevance_review(
            study_dir,
            {
                "query_id": rows[0]["query_id"],
                "decision": "answerable",
                "relevant_item_ids": "not_real",
            },
            known_item_ids={"item_a", "item_b"},
        )


def test_completed_reviews_materialize_evaluator_compatible_csv(tmp_path) -> None:
    study_dir = create_study(tmp_path)
    rows = [submit(study_dir, query) for query in ("查询一号", "查询二号", "查询三号")]
    freeze_study(study_dir, current_library_revision=REVISION, now=NOW)
    for row in rows:
        save_candidate(
            study_dir,
            candidate(row["query_id"]),
            current_library_revision=REVISION,
            now=NOW,
        )
    save_relevance_review(
        study_dir,
        {
            "query_id": rows[0]["query_id"],
            "decision": "answerable",
            "relevant_item_ids": "item_a;item_b",
            "evidence_scope": "library_browse",
            "reviewer_type": "model_consensus",
            "reviewer_id": "reviewer_a+reviewer_b",
            "review_confidence": "0.97",
        },
        known_item_ids={"item_a", "item_b"},
        now=NOW,
    )
    save_relevance_review(
        study_dir,
        {"query_id": rows[1]["query_id"], "decision": "no_answer"},
        known_item_ids={"item_a", "item_b"},
        now=NOW,
    )
    save_relevance_review(
        study_dir,
        {"query_id": rows[2]["query_id"], "decision": "excluded"},
        known_item_ids={"item_a", "item_b"},
        now=NOW,
    )
    formal = materialize_formal_rows(
        study_dir, minimum_eligible=2, now=NOW
    )
    assert [row["is_no_answer"] for row in formal] == ["false", "true"]
    assert formal[0]["reviewer_type"] == "model_consensus"
    output = tmp_path / "formal.csv"
    write_formal_query_set(output, formal)
    assert "review_status" in output.read_text(encoding="utf-8-sig")
    evaluator_rows = [
        {
            **row,
            "is_no_answer": row["is_no_answer"] == "true",
            "relevant_item_ids": set(
                item_id
                for item_id in row["relevant_item_ids"].split(";")
                if item_id
            ),
        }
        for row in formal
    ]
    assert evaluator_query_set_sha256(formal) == query_set_fingerprint(
        evaluator_rows
    )
    evaluation_protocol_path = tmp_path / "evaluation_protocol.json"
    evaluation_protocol = write_live_evaluation_protocol(
        evaluation_protocol_path,
        study_dir=study_dir,
        formal_rows=formal,
        query_file="data/evaluation/blind_study_v1/formal_queries.csv",
        library_dir="outputs/user_library",
        output="outputs/evaluation/library_retrieval/blind_v1.json",
        now=NOW,
    )
    assert evaluation_protocol["query_count"] == 2
    assert evaluation_protocol["expected_search_policy_version"] == 15
    assert load_protocol(study_dir)["status"] == "completed"
    evaluator_query_set_sha256,
