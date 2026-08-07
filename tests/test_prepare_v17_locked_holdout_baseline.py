from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prepare_v17_locked_holdout_baseline import build_baseline_records


def fixtures() -> tuple[list[dict], list[dict], dict[str, dict]]:
    packets = [{"query_id": "q1", "query": "query", "group_id": "g1"}]
    rankings = [{"query_id": "q1", "ranking": [{"item_id": "a"}]}]
    raw = {
        "q1": {
            "query": "query",
            "search_policy_version": 16,
            "retrieval_config_revision": "config",
            "library_revision": "library",
            "rankings": {"quality_hybrid": [{"item_id": "a"}]},
            "acceptance": {
                "quality_hybrid": {
                    "accepted": True,
                    "signal_name": "probability",
                    "signal": 0.8,
                    "threshold": 0.2,
                }
            },
        }
    }
    return packets, rankings, raw


def test_builds_label_blind_baseline_decision() -> None:
    rows = build_baseline_records(*fixtures(), ranking_sha256="ranking")

    assert rows[0]["judgments_read"] is False
    assert rows[0]["v16_accepted"] is True
    assert rows[0]["selected_item_id"] == "a"
    assert "v16_pool_conditioned_correct" not in rows[0]


def test_rejected_baseline_has_no_selected_item() -> None:
    packets, rankings, raw = fixtures()
    raw["q1"]["acceptance"]["quality_hybrid"]["accepted"] = False

    rows = build_baseline_records(
        packets, rankings, raw, ranking_sha256="ranking"
    )

    assert rows[0]["selected_item_id"] is None
    assert rows[0]["selected_rank"] is None


def test_rejects_raw_and_frozen_ranking_mismatch() -> None:
    packets, rankings, raw = fixtures()
    changed = copy.deepcopy(raw)
    changed["q1"]["rankings"]["quality_hybrid"][0]["item_id"] = "b"

    with pytest.raises(ValueError, match="rankings differ"):
        build_baseline_records(
            packets, rankings, changed, ranking_sha256="ranking"
        )
