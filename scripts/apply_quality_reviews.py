"""Publish human quality-review decisions into a library manifest."""

from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.taxonomy import parse_quality_tags


def apply_quality_reviews(
    manifest: list[dict[str, Any]],
    reviews: dict[str, dict[str, str]],
    *,
    applied_at: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    known_ids = {row["item_id"] for row in manifest}
    unknown_ids = sorted(set(reviews) - known_ids)
    if unknown_ids:
        raise ValueError(
            "Quality reviews contain unknown item IDs: "
            + ", ".join(unknown_ids)
        )

    decision_counts: Counter[str] = Counter()
    updated: list[dict[str, Any]] = []
    for source_row in manifest:
        row = dict(source_row)
        review = reviews.get(row["item_id"])
        if review is None:
            updated.append(row)
            continue
        decision = review["decision"]
        decision_counts[decision] += 1
        row["quality_decision"] = decision
        row["quality_review_applied_at"] = applied_at
        row["quality_review_notes"] = review.get("human_notes", "")
        reviewed_quality_tags = parse_quality_tags(
            review.get("quality_tags")
        )
        if reviewed_quality_tags:
            row["quality_tags"] = reviewed_quality_tags

        if decision == "accepted":
            row["search_enabled"] = True
            row["review_status"] = "人工确认可用"
            row["taxonomy_review_status"] = "reviewed"
        elif decision == "reclassified":
            revised_category = review.get("revised_category", "")
            if not revised_category:
                raise ValueError(
                    f"Missing revised category for {row['item_id']}"
                )
            row.setdefault("original_category", row.get("category", ""))
            row["category"] = revised_category
            row["search_enabled"] = True
            row["review_status"] = "人工重分类"
            row["taxonomy_review_status"] = "reviewed"
        elif decision == "ocr_retry":
            row["search_enabled"] = False
            row["review_status"] = "待OCR重试"
        elif decision == "quarantined":
            row["search_enabled"] = False
            row["review_status"] = "人工隔离"
        else:
            raise ValueError(
                f"Unsupported quality decision for {row['item_id']}: "
                f"{decision}"
            )
        updated.append(row)

    active_pages = sum(
        bool(row.get("search_enabled", True)) for row in updated
    )
    report = {
        "status": "success",
        "applied_at": applied_at,
        "review_count": len(reviews),
        "decision_counts": dict(sorted(decision_counts.items())),
        "total_pages": len(updated),
        "active_pages": active_pages,
        "inactive_pages": len(updated) - active_pages,
    }
    return updated, report


def publish_quality_reviews(
    manifest_path: Path,
    reviews: dict[str, dict[str, str]],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    timestamp = now or datetime.now(timezone.utc)
    applied_at = timestamp.isoformat(timespec="seconds")
    manifest = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    updated, report = apply_quality_reviews(
        manifest,
        reviews,
        applied_at=applied_at,
    )

    backup_dir = manifest_path.parent / "manifest_backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = (
        backup_dir
        / f"manifest_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}.jsonl"
    )
    shutil.copy2(manifest_path, backup_path)

    temporary_path = manifest_path.with_suffix(".jsonl.tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        for row in updated:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary_path, manifest_path)
    report["backup_path"] = str(backup_path)
    return report
