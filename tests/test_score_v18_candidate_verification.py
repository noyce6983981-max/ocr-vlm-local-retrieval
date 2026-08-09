from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


def _load_module():
    path = PROJECT_ROOT / "scripts/score_v18_candidate_verification.py"
    spec = importlib.util.spec_from_file_location(
        "score_v18_candidate_verification", path
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tasks_use_only_base_top_k_candidates_inside_pool() -> None:
    module = _load_module()
    pool = [
        {
            "query_id": "q1",
            "query": "query",
            "group_id": "g1",
            "candidates": [
                {
                    "item_id": f"i{index}",
                    "review_metadata": {"image_path": f"images/i{index}.jpg"},
                }
                for index in range(1, 4)
            ],
        }
    ]
    ranking = [
        {
            "query_id": "q1",
            "ranking": [
                {"item_id": "i2", "score": 0.9},
                {"item_id": "i1", "score": 0.8},
                {"item_id": "i3", "score": 0.7},
            ],
        }
    ]
    tasks = module.build_verification_tasks(pool, ranking, top_k=2)
    assert [row["item_id"] for row in tasks[0]["candidates"]] == ["i2", "i1"]
    assert [row["retrieval_rank"] for row in tasks[0]["candidates"]] == [1, 2]
    assert "candidate_relevance" not in str(tasks)


def test_tasks_reject_ranked_candidate_outside_pool() -> None:
    module = _load_module()
    pool = [
        {
            "query_id": "q1",
            "query": "query",
            "group_id": "g1",
            "candidates": [],
        }
    ]
    ranking = [{"query_id": "q1", "ranking": [{"item_id": "outside"}]}]
    with pytest.raises(ValueError, match="outside the pool"):
        module.build_verification_tasks(pool, ranking, top_k=1)


def test_output_lock_rejects_a_second_writer(tmp_path: Path) -> None:
    module = _load_module()
    output_path = tmp_path / "verification.json"
    first = module.acquire_output_lock(output_path)
    try:
        with pytest.raises(RuntimeError, match="already writing"):
            module.acquire_output_lock(output_path)
    finally:
        module.release_output_lock(first)
    second = module.acquire_output_lock(output_path)
    module.release_output_lock(second)
