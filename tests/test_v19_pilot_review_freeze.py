from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.finalize_v19_pilot_reviews import validate_and_merge


def rows_from_text(tmp_path: Path, name: str, text: str) -> list[dict[str, str]]:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_review_freeze_binds_route_to_exact_query_text(tmp_path: Path) -> None:
    queries = rows_from_text(
        tmp_path,
        "queries.csv",
        "query_id,query_text,gold_route,status\n"
        "q1,查金额,text_evidence,draft_pilot_not_final\n",
    )
    reviews = rows_from_text(
        tmp_path,
        "reviews.csv",
        "query_id,query_text,proposed_route,final_route,accepted_proposal\n"
        "q1,查金额,text_evidence,mixed,false\n",
    )
    merged = validate_and_merge(queries, reviews)
    assert merged[0]["gold_route"] == "mixed"
    assert merged[0]["status"] == "human_reviewed_pilot_not_final"


def test_review_freeze_rejects_query_text_mismatch(tmp_path: Path) -> None:
    queries = rows_from_text(
        tmp_path,
        "queries.csv",
        "query_id,query_text,gold_route\nq1,原查询,text_evidence\n",
    )
    reviews = rows_from_text(
        tmp_path,
        "reviews.csv",
        "query_id,query_text,proposed_route,final_route\n"
        "q1,被替换的查询,text_evidence,text_evidence\n",
    )
    with pytest.raises(ValueError, match="query text mismatch"):
        validate_and_merge(queries, reviews)


def test_review_freeze_requires_complete_unique_coverage(tmp_path: Path) -> None:
    queries = rows_from_text(
        tmp_path,
        "queries.csv",
        "query_id,query_text,gold_route\n"
        "q1,查询一,text_evidence\n"
        "q2,查询二,mixed\n",
    )
    reviews = rows_from_text(
        tmp_path,
        "reviews.csv",
        "query_id,query_text,proposed_route,final_route\n"
        "q1,查询一,text_evidence,text_evidence\n",
    )
    with pytest.raises(ValueError, match="coverage mismatch"):
        validate_and_merge(queries, reviews)
