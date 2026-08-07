from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUEUE = (
    PROJECT_ROOT
    / "data/evaluation/public_dataset_1500_retrieval_query_queue_100.csv"
)
PROTOCOL = (
    PROJECT_ROOT
    / "data/evaluation/public_dataset_1500_retrieval_query_protocol.json"
)


def test_retrieval_query_queue_has_frozen_balanced_shape() -> None:
    with QUEUE.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 100
    assert len({row["query_id"] for row in rows}) == 100
    assert len({row["query"] for row in rows}) == 100
    assert Counter(row["split"] for row in rows) == {
        "train": 60,
        "validation": 20,
        "test": 20,
    }
    assert sum(row["query_type"] == "no_answer" for row in rows) == 10
    assert all(
        not row["relevant_item_ids"]
        for row in rows
        if row["query_type"] == "no_answer"
    )
    assert all(
        len(row["candidate_item_ids"].split(";")) == 3
        for row in rows
        if row["query_type"] == "no_answer"
    )
    assert all(
        row["relevant_item_ids"]
        for row in rows
        if row["query_type"] != "no_answer"
    )
    answerable_groups = [
        row["relevant_item_ids"] for row in rows if row["relevant_item_ids"]
    ]
    assert len(answerable_groups) == len(set(answerable_groups))


def test_retrieval_query_protocol_matches_queue() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert protocol["query_count"] == 100
    assert protocol["answerable_count"] == 90
    assert protocol["no_answer_count"] == 10
    assert protocol["target_group_count"] == 90
    assert protocol["privacy_targets_included"] is False
    assert protocol["open_set_candidates_attached"] == 10
