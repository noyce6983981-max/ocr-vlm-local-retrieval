from __future__ import annotations

import csv
from pathlib import Path

from scripts.v17_query_review_app import read_csv, save_review


def test_query_review_storage_keeps_one_row_per_query(tmp_path: Path) -> None:
    path = tmp_path / "reviews.csv"
    save_review(
        path,
        {
            "query_id": "q1",
            "review_action": "approve",
            "reviewed_query": "first",
            "reviewer_id": "human_a",
            "human_notes": "",
            "reviewed_at": "2026-08-07T00:00:00+00:00",
        },
    )
    save_review(
        path,
        {
            "query_id": "q1",
            "review_action": "rewrite",
            "reviewed_query": "revised",
            "reviewer_id": "human_a",
            "human_notes": "clearer wording",
            "reviewed_at": "2026-08-07T00:01:00+00:00",
        },
    )
    rows = read_csv(path)
    assert len(rows) == 1
    assert rows[0]["review_action"] == "rewrite"
    assert rows[0]["reviewed_query"] == "revised"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 1
