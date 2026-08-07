from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.score_v17_candidate_verification import (
    build_verification_tasks,
    merge_packet_extensions,
    resume_results,
)


def packets():
    return [
        {
            "query_id": "q1",
            "query": "visible word on the engine",
            "group_id": "g1",
            "candidates": [
                {
                    "item_id": "a",
                    "review_metadata": {"image_path": "images/a.jpg"},
                },
                {
                    "item_id": "b",
                    "review_metadata": {"image_path": "images/b.jpg"},
                },
            ],
        }
    ]


def test_build_verification_tasks_joins_blinded_pool_without_labels() -> None:
    rankings = [
        {
            "query_id": "q1",
            "ranking": [
                {"item_id": "b", "score": 0.8},
                {"item_id": "a", "score": 0.7},
            ],
        }
    ]

    tasks = build_verification_tasks(packets(), rankings, top_k=1)

    assert tasks == [
        {
            "query_id": "q1",
            "query": "visible word on the engine",
            "group_id": "g1",
            "candidates": [
                {
                    "item_id": "b",
                    "image_path": "images/b.jpg",
                    "ranking_score": 0.8,
                }
            ],
        }
    ]
    assert "candidate_relevance" not in str(tasks)


def test_build_verification_tasks_rejects_candidate_outside_pool() -> None:
    rankings = [{"query_id": "q1", "ranking": [{"item_id": "unknown", "score": 0.8}]}]

    with pytest.raises(ValueError, match="outside the blinded review pool"):
        build_verification_tasks(packets(), rankings, top_k=1)


def test_build_verification_tasks_requires_requested_depth() -> None:
    rankings = [{"query_id": "q1", "ranking": [{"item_id": "a", "score": 0.8}]}]

    with pytest.raises(ValueError, match="fewer than 2"):
        build_verification_tasks(packets(), rankings, top_k=2)


def test_packet_extension_appends_candidate_without_mutating_input() -> None:
    base = packets()
    merged = merge_packet_extensions(
        base,
        [
            {
                "query_id": "q1",
                "query": "visible word on the engine",
                "split": "calibration",
                "candidates": [
                    {
                        "item_id": "c",
                        "review_metadata": {"image_path": "images/c.jpg"},
                    }
                ],
            }
        ],
    )

    assert [row["item_id"] for row in merged[0]["candidates"]] == ["a", "b", "c"]
    assert [row["item_id"] for row in base[0]["candidates"]] == ["a", "b"]


def test_resume_requires_exact_ordered_schema_v2_prefix(tmp_path: Path) -> None:
    output = tmp_path / "verification.json"
    identity = {"top_k": 5, "ranking_sha256": "ranking"}
    output.write_text(
        json.dumps(
            {
                "schema_version": 2,
                **identity,
                "elapsed_seconds": 12.5,
                "peak_reserved_gib": 4.1,
                "results": [{"query_id": "q1"}],
            }
        ),
        encoding="utf-8",
    )

    rows, elapsed, peak = resume_results(
        output,
        run_identity=identity,
        ordered_query_ids=["q1", "q2"],
    )
    assert [row["query_id"] for row in rows] == ["q1"]
    assert elapsed == 12.5
    assert peak == 4.1

    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["results"] = [{"query_id": "q2"}]
    output.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="ordered prefix"):
        resume_results(
            output,
            run_identity=identity,
            ordered_query_ids=["q1", "q2"],
        )
