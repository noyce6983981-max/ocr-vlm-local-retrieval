"""Utilities for recording human review of retrieval queries."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REVIEW_FIELDS = [
    "query_id",
    "decision",
    "reviewed_query",
    "relevant_item_ids",
    "human_notes",
    "reviewed_at",
]
VALID_DECISIONS = {"accepted", "revised", "no_answer", "excluded"}


def split_item_ids(value: str) -> list[str]:
    """Return unique semicolon-delimited item IDs while preserving order."""
    result: list[str] = []
    seen: set[str] = set()
    for item_id in value.replace(",", ";").split(";"):
        normalized = item_id.strip()
        if normalized and normalized not in seen:
            result.append(normalized)
            seen.add(normalized)
    return result


def validate_review(
    decision: str,
    reviewed_query: str,
    relevant_item_ids: str,
    known_item_ids: set[str],
) -> tuple[str, str]:
    """Validate and normalize one human query-review decision."""
    if decision not in VALID_DECISIONS:
        raise ValueError(f"Unsupported review decision: {decision}")

    query = " ".join(reviewed_query.split())
    relevant_ids = split_item_ids(relevant_item_ids)
    if decision != "excluded" and not query:
        raise ValueError("通过的查询不能为空。")
    if decision in {"accepted", "revised"} and not relevant_ids:
        raise ValueError("有答案查询至少需要一张相关图片。")
    if decision == "no_answer" and relevant_ids:
        raise ValueError("无答案查询不能填写正确图片ID。")

    unknown_ids = sorted(set(relevant_ids) - known_item_ids)
    if unknown_ids:
        raise ValueError("未知图片ID：" + "、".join(unknown_ids))
    return query, ";".join(relevant_ids)


def read_reviews(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            row["query_id"]: row
            for row in csv.DictReader(handle)
            if row.get("query_id")
        }


def save_review(
    path: Path,
    review: dict[str, Any],
    now: datetime | None = None,
) -> None:
    """Upsert a review row using an atomic file replacement."""
    reviews = read_reviews(path)
    row = {field: str(review.get(field, "")) for field in REVIEW_FIELDS}
    if not row["query_id"]:
        raise ValueError("query_id cannot be empty.")
    timestamp = now or datetime.now(timezone.utc)
    row["reviewed_at"] = timestamp.isoformat(timespec="seconds")
    reviews[row["query_id"]] = row

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(reviews.values())
    temporary_path.replace(path)
