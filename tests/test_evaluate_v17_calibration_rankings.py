from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.evaluate_v17_calibration_rankings import RUN_NAMES, evaluate


def write_raw(raw_dir: Path, query_id: str, version: str, accepted: bool) -> None:
    payload = {
        "acceptance": {
            "quality_hybrid": {"accepted": accepted, "reason": f"{version} reason"}
        }
    }
    (raw_dir / f"{query_id}_{version}.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def fixtures(raw_dir: Path):
    raw_dir.mkdir()
    judgments = [
        {
            "query_id": "q1",
            "answerability": "answerable",
            "candidate_relevance": {"good": True, "bad": False},
        },
        {
            "query_id": "q2",
            "answerability": "no_answer",
            "candidate_relevance": {"bad": False, "other": False},
        },
        {
            "query_id": "q3",
            "answerability": "excluded",
            "candidate_relevance": {"bad": False},
        },
    ]
    packets = [
        {"query_id": "q1", "group_id": "g1", "query_family": "relation"},
        {"query_id": "q2", "group_id": "g2", "query_family": "relation"},
        {"query_id": "q3", "group_id": "g3", "query_family": "relation"},
    ]
    runs = {}
    for name in RUN_NAMES:
        runs[name] = [
            {
                "query_id": "q1",
                "ranking": [
                    {"item_id": "good" if name.startswith("v17") else "bad"},
                    {"item_id": "good"},
                ],
            },
            {"query_id": "q2", "ranking": [{"item_id": "bad"}]},
            {"query_id": "q3", "ranking": [{"item_id": "bad"}]},
        ]
    write_raw(raw_dir, "q1", "v16", True)
    write_raw(raw_dir, "q1", "v17", False)
    write_raw(raw_dir, "q2", "v16", True)
    write_raw(raw_dir, "q2", "v17", False)
    return judgments, packets, runs


def test_evaluate_flags_reject_all_and_preserves_ranking_gain(tmp_path: Path) -> None:
    judgments, packets, runs = fixtures(tmp_path / "raw")
    report, rows = evaluate(
        judgments=judgments,
        packets=packets,
        run_rows=runs,
        raw_dir=tmp_path / "raw",
        bootstrap_repetitions=25,
        seed=17,
    )

    assert len(rows) == 2
    assert report["scope"]["excluded_query_ids"] == ["q3"]
    assert report["ranking_metrics"]["v16_quality_hybrid"]["recall_at_1"] == 0.0
    assert report["ranking_metrics"]["v17_quality_hybrid"]["recall_at_1"] == 1.0
    assert report["open_set_quality_hybrid"]["v17"]["degenerate_reject_all"]
    assert not report["research_hypotheses"][
        "h1_false_accept_relative_reduction_at_least_30_percent"
    ]["valid_success"]
    assert (
        report["paired_group_bootstrap"]["answerable_recall_at_3_v17_minus_v16"][
            "repetitions"
        ]
        == 25
    )


def test_evaluate_rejects_top_three_result_outside_audited_pool(tmp_path: Path) -> None:
    judgments, packets, runs = fixtures(tmp_path / "raw")
    runs["bm25"][0]["ranking"][0]["item_id"] = "unknown"

    with pytest.raises(ValueError, match="outside the audited pool"):
        evaluate(
            judgments=judgments,
            packets=packets,
            run_rows=runs,
            raw_dir=tmp_path / "raw",
            bootstrap_repetitions=2,
            seed=17,
        )


def test_evaluate_rejects_answerable_query_without_relevant_candidate(
    tmp_path: Path,
) -> None:
    judgments, packets, runs = fixtures(tmp_path / "raw")
    judgments[0]["candidate_relevance"] = {"good": False, "bad": False}

    with pytest.raises(ValueError, match="has no relevant candidate"):
        evaluate(
            judgments=judgments,
            packets=packets,
            run_rows=runs,
            raw_dir=tmp_path / "raw",
            bootstrap_repetitions=2,
            seed=17,
        )
