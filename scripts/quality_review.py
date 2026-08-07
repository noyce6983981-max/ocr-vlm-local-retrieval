"""Record auditable human decisions for automatic ingestion-quality routes."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.taxonomy import (
    CATEGORY_LABELS,
    QUALITY_LABELS,
    serialize_quality_tags,
)


REVIEW_FIELDS = [
    "item_id",
    "decision",
    "revised_category",
    "quality_tags",
    "human_notes",
    "reviewed_at",
]
VALID_DECISIONS = {
    "accepted",
    "reclassified",
    "ocr_retry",
    "quarantined",
}
VALID_CATEGORIES = set(CATEGORY_LABELS)


def read_quality_reviews(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            row["item_id"]: row
            for row in csv.DictReader(handle)
            if row.get("item_id")
        }


def validate_quality_review(
    *,
    item_id: str,
    decision: str,
    revised_category: str,
    quality_tags: str | list[str] | None = None,
    known_item_ids: set[str],
) -> tuple[str, str]:
    if item_id not in known_item_ids:
        raise ValueError(f"Unknown item ID: {item_id}")
    if decision not in VALID_DECISIONS:
        raise ValueError(f"Unsupported decision: {decision}")
    normalized_category = revised_category.strip()
    if decision == "reclassified":
        if normalized_category not in VALID_CATEGORIES:
            raise ValueError("A valid revised category is required.")
    elif normalized_category and normalized_category not in VALID_CATEGORIES:
        raise ValueError(f"Unsupported category: {normalized_category}")
    normalized_quality_tags = serialize_quality_tags(quality_tags)
    unknown_quality_tags = (
        set(normalized_quality_tags.split(";")) - set(QUALITY_LABELS)
        if normalized_quality_tags
        else set()
    )
    if unknown_quality_tags:
        raise ValueError(
            "Unsupported quality tags: "
            + ", ".join(sorted(unknown_quality_tags))
        )
    return normalized_category, normalized_quality_tags


def save_quality_review(
    path: Path,
    review: dict[str, Any],
    now: datetime | None = None,
) -> None:
    reviews = read_quality_reviews(path)
    row = {
        field: str(review.get(field, ""))
        for field in REVIEW_FIELDS
        if field != "quality_tags"
    }
    row["quality_tags"] = serialize_quality_tags(
        review.get("quality_tags")
    )
    if not row["item_id"]:
        raise ValueError("item_id cannot be empty.")
    timestamp = now or datetime.now(timezone.utc)
    row["reviewed_at"] = timestamp.isoformat(timespec="seconds")
    reviews[row["item_id"]] = row

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(reviews.values())
    temporary_path.replace(path)
