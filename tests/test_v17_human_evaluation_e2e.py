from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def test_pool_to_double_review_report_without_model_weights(tmp_path: Path) -> None:
    queries = tmp_path / "queries.json"
    v16 = tmp_path / "v16.json"
    v17 = tmp_path / "v17.json"
    output_dir = tmp_path / "study"
    _write_json(
        queries,
        [
            {
                "query_id": "q001",
                "query": "yellow dolphin jumping over a pyramid",
                "split": "holdout",
                "group_id": "dolphin_binding_family",
                "query_family": "object_color_relation",
                "route": "compositional_visual",
            }
        ],
    )
    _write_json(
        v16,
        [
            {
                "query_id": "q001",
                "ranking": [
                    {"item_id": "yellow_pyramid", "score": 0.9},
                    {"item_id": "dolphin_scene", "score": 0.8},
                ],
            }
        ],
    )
    _write_json(
        v17,
        [
            {
                "query_id": "q001",
                "ranking": [
                    {"item_id": "dolphin_scene", "score": 0.88},
                    {"item_id": "yellow_pyramid", "score": 0.2},
                ],
            }
        ],
    )
    build = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/build_v17_human_pool.py"),
            "--queries",
            str(queries),
            "--run",
            f"v16={v16}",
            "--run",
            f"v17={v17}",
            "--output-dir",
            str(output_dir),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert build.returncode == 0, build.stderr
    packet = json.loads(
        (output_dir / "review_packets.jsonl").read_text(encoding="utf-8")
    )
    assert packet["candidate_count"] == 2
    assert "run_ids" not in packet
    assert "pool_score" not in json.dumps(packet)

    judgments = tmp_path / "judgments.json"
    _write_json(
        judgments,
        [
            {
                "query_id": "q001",
                "reviewer_id": "reviewer_a",
                "answerability": "answerable",
                "candidate_relevance": {
                    "yellow_pyramid": False,
                    "dolphin_scene": True,
                },
            },
            {
                "query_id": "q001",
                "reviewer_id": "reviewer_b",
                "answerability": "answerable",
                "candidate_relevance": {
                    "yellow_pyramid": False,
                    "dolphin_scene": True,
                },
            },
        ],
    )
    report_path = output_dir / "report.json"
    evaluate = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/evaluate_v17_human_study.py"),
            "--queries",
            str(queries),
            "--judgments",
            str(judgments),
            "--output",
            str(report_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert evaluate.returncode == 0, evaluate.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["annotation_coverage"]["valid"]
    assert report["agreement"]["candidate_raw_agreement"] == 1.0
