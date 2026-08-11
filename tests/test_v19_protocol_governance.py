from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_v19_protocol_keeps_llm_out_of_final_control() -> None:
    config = json.loads(
        (
            repository_root() / "config/studies/v19_intent_routing.json"
        ).read_text(encoding="utf-8")
    )
    assert config["version"] == "V19"
    assert config["scope"]["llm_controls_final_weights"] is False
    assert config["scope"]["llm_controls_rejection_thresholds"] is False
    assert (
        config["scope"]["llm_self_reported_confidence_used_for_gating"]
        is False
    )
    assert config["scope"]["v18_holdout_reused_as_v19_result"] is False
    assert config["architecture"]["default_feature_flag"] is False
    assert config["data_plan"]["formal"]["holdout_runs_after_lock"] == 1


def test_v19_pilot_is_balanced_unique_and_non_final() -> None:
    path = (
        repository_root()
        / "data/evaluation/v19/pilot/queries_draft.csv"
    )
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 72
    assert len({row["query_id"] for row in rows}) == 72
    assert len({row["query_text"] for row in rows}) == 72
    assert Counter(row["gold_route"] for row in rows) == {
        "text_evidence": 12,
        "visual_discovery": 12,
        "visual_metadata": 12,
        "entity_exact": 12,
        "topic_discovery": 12,
        "mixed": 12,
    }
    assert {row["status"] for row in rows} == {"draft_pilot_not_final"}
