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
    path = PROJECT_ROOT / "scripts/build_v18_calibration_pool.py"
    spec = importlib.util.spec_from_file_location("build_v18_calibration_pool", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ranking(prefix: str, count: int = 20) -> list[dict[str, object]]:
    return [
        {"item_id": f"{prefix}_{index:02d}", "score": 1.0 / index}
        for index in range(1, count + 1)
    ]


def test_capped_pool_keeps_base_top10_and_every_route_top1() -> None:
    module = _load_module()
    rankings = {
        "v17_quality_hybrid": _ranking("base"),
        "text": _ranking("text"),
        "bm25": _ranking("bm25"),
        "global_visual": _ranking("visual"),
        "non_attribute_quality_hybrid": _ranking("nonattr"),
        "v17_attribute_coverage": _ranking("attribute"),
    }
    pooled = module.pool_ranked_runs(rankings, top_per_run=20)
    selected, mandatory = module.select_capped_pool(
        pooled,
        rankings,
        pool_size=20,
        base_guaranteed_depth=10,
        all_route_guaranteed_depth=1,
    )
    selected_ids = {row["item_id"] for row in selected}
    assert len(selected) == 20
    assert {f"base_{index:02d}" for index in range(1, 11)} <= selected_ids
    assert {ranking[0]["item_id"] for ranking in rankings.values()} <= selected_ids
    assert mandatory <= selected_ids


def test_blind_packets_hide_run_names_and_ranks() -> None:
    module = _load_module()
    queries = [
        {
            "query_id": "q1",
            "query": "绿色树木位于岩石前方。",
            "split": "calibration",
            "group_id": "g1",
            "query_role": "positive",
        }
    ]
    runs = {
        run_id: {"q1": _ranking(prefix)}
        for run_id, prefix in {
            "v17_quality_hybrid": "base",
            "text": "text",
            "bm25": "bm25",
            "global_visual": "visual",
            "non_attribute_quality_hybrid": "nonattr",
            "v17_attribute_coverage": "attribute",
        }.items()
    }
    _, reviewer, _ = module.build_pools(
        queries,
        runs,
        retrieval_receipt_sha256="receipt",
    )
    assert len(reviewer[0]["candidates"]) == 20
    for candidate in reviewer[0]["candidates"]:
        assert set(candidate) == {"candidate_id", "item_id", "review_asset"}


def test_build_pools_rejects_an_incomplete_blind_pool() -> None:
    module = _load_module()
    queries = [
        {
            "query_id": "q1",
            "query": "绿色树木位于岩石前方。",
            "split": "calibration",
            "group_id": "g1",
            "query_role": "positive",
        }
    ]
    runs = {
        run_id: {"q1": [{"item_id": "only", "score": 1.0}]}
        for run_id in module.RUN_METHODS
    }
    with pytest.raises(ValueError, match="produced 1 unique candidates"):
        module.build_pools(
            queries,
            runs,
            retrieval_receipt_sha256="receipt",
        )
