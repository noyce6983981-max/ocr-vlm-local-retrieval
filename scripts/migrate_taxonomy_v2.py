"""Migrate the active library to content categories plus quality tags."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.quality_review import read_quality_reviews
from scripts.taxonomy import (
    CATEGORY_LABELS,
    QUALITY_LABELS,
    migrate_manifest_row,
)


DEFAULT_LIBRARY_DIR = PROJECT_ROOT / "outputs/user_library"
DEFAULT_REVIEW_PATH = (
    PROJECT_ROOT
    / "data/evaluation/public_dataset_1500_quality_human_reviews.csv"
)


def load_manifest(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def migrate_rows(
    rows: list[dict[str, Any]],
    reviews: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    return [
        migrate_manifest_row(row, reviews.get(row["item_id"]))
        for row in rows
    ]


def write_manifest_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def write_review_queue(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    fields = [
        "item_id",
        "source_file_name",
        "taxonomy_v1_category",
        "proposed_category",
        "quality_tags",
        "search_enabled",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            if row.get("taxonomy_review_status") != "pending":
                continue
            writer.writerow(
                {
                    "item_id": row["item_id"],
                    "source_file_name": row.get("source_file_name", ""),
                    "taxonomy_v1_category": row.get(
                        "taxonomy_v1_category", ""
                    ),
                    "proposed_category": row["category"],
                    "quality_tags": ";".join(row.get("quality_tags", [])),
                    "search_enabled": bool(
                        row.get("search_enabled", True)
                    ),
                }
            )
    os.replace(temporary, path)


def build_report(
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
    backup_path: Path,
    queue_path: Path,
    migrated_at: str,
) -> dict[str, Any]:
    return {
        "status": "success",
        "taxonomy_version": 2,
        "migrated_at": migrated_at,
        "total_pages": len(after),
        "active_pages": sum(
            bool(row.get("search_enabled", True)) for row in after
        ),
        "before_categories": dict(
            sorted(Counter(row.get("category", "") for row in before).items())
        ),
        "after_categories": dict(
            sorted(Counter(row.get("category", "") for row in after).items())
        ),
        "taxonomy_review_status": dict(
            sorted(
                Counter(
                    row.get("taxonomy_review_status", "") for row in after
                ).items()
            )
        ),
        "category_labels": dict(CATEGORY_LABELS),
        "quality_labels": dict(QUALITY_LABELS),
        "backup_path": str(backup_path),
        "review_queue_path": str(queue_path),
    }


def migrate(
    library_dir: Path,
    review_path: Path,
) -> dict[str, Any]:
    manifest_path = library_dir / "manifest.jsonl"
    rows = load_manifest(manifest_path)
    if rows and all(row.get("taxonomy_version") == 2 for row in rows):
        raise ValueError("当前manifest已经是taxonomy v2，拒绝重复迁移。")
    reviews = read_quality_reviews(review_path)
    migrated = migrate_rows(rows, reviews)

    timestamp = datetime.now(timezone.utc)
    migrated_at = timestamp.isoformat(timespec="seconds")
    backup_dir = library_dir / "manifest_backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = (
        backup_dir
        / f"taxonomy_v1_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}.jsonl"
    )
    shutil.copy2(manifest_path, backup_path)
    write_manifest_atomic(manifest_path, migrated)

    taxonomy_dir = library_dir / "taxonomy"
    queue_path = taxonomy_dir / "taxonomy_v2_review_queue.csv"
    write_review_queue(queue_path, migrated)
    report = build_report(
        rows,
        migrated,
        backup_path,
        queue_path,
        migrated_at,
    )
    report_path = taxonomy_dir / "taxonomy_v2_migration_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report["report_path"] = str(report_path)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--library-dir",
        type=Path,
        default=DEFAULT_LIBRARY_DIR,
    )
    parser.add_argument(
        "--review-path",
        type=Path,
        default=DEFAULT_REVIEW_PATH,
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = migrate(
        arguments.library_dir.resolve(),
        arguments.review_path.resolve(),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
