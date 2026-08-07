"""Clear human review memory and rebuild the pending taxonomy queue."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.migrate_taxonomy_v2 import write_review_queue
from scripts.taxonomy import (
    AMBIGUOUS_LEGACY_CATEGORIES,
    normalize_category,
    source_tag_category_suggestion,
)


DEFAULT_LIBRARY_DIR = PROJECT_ROOT / "outputs/user_library"
DEFAULT_QUALITY_REVIEWS = (
    PROJECT_ROOT
    / "data/evaluation/public_dataset_1500_quality_human_reviews.csv"
)
DEFAULT_QUERY_REVIEWS = (
    PROJECT_ROOT
    / "data/evaluation/public_dataset_200_query_human_reviews.csv"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def reset_manifest_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    reset_rows: list[dict[str, Any]] = []
    auto_refined = 0
    for source_row in rows:
        row = dict(source_row)
        legacy_category = str(
            row.get("taxonomy_v1_category") or row.get("category", "")
        )
        row["category"] = normalize_category(legacy_category)
        row.pop("quality_review_applied_at", None)
        row.pop("quality_review_notes", None)
        row.pop("human_category", None)
        row.pop("human_notes", None)

        if legacy_category in AMBIGUOUS_LEGACY_CATEGORIES:
            row["taxonomy_review_status"] = "pending"
        else:
            row["taxonomy_review_status"] = "preserved"

        suggestion = source_tag_category_suggestion(row)
        if suggestion and suggestion != row["category"]:
            row["category"] = suggestion
            row["auto_category_reason"] = "trusted_source_tag"
            auto_refined += 1
        else:
            row.pop("auto_category_reason", None)
        reset_rows.append(row)
    return reset_rows, auto_refined


def reset_manual_reviews(
    library_dir: Path,
    quality_reviews: Path,
    query_reviews: Path,
) -> dict[str, Any]:
    library_dir = library_dir.resolve()
    project_root = PROJECT_ROOT.resolve()
    if not library_dir.is_relative_to(project_root):
        raise ValueError("Library must be inside the project.")
    review_paths = [
        quality_reviews.resolve(),
        query_reviews.resolve(),
    ]
    if any(not path.is_relative_to(project_root) for path in review_paths):
        raise ValueError("Review files must be inside the project.")
    manifest_path = library_dir / "manifest.jsonl"
    before = read_jsonl(manifest_path)
    after, auto_refined = reset_manifest_rows(before)
    write_jsonl_atomic(manifest_path, after)

    deleted_review_files: list[str] = []
    for path in review_paths:
        if path.is_file():
            path.unlink()
            deleted_review_files.append(
                path.relative_to(PROJECT_ROOT).as_posix()
            )

    feedback_dir = library_dir / "feedback_model"
    feedback_removed = feedback_dir.is_dir()
    if feedback_removed:
        shutil.rmtree(feedback_dir)

    queue_path = (
        library_dir / "taxonomy/taxonomy_v2_review_queue.csv"
    )
    write_review_queue(queue_path, after)
    return {
        "status": "reset",
        "page_count": len(after),
        "human_review_files_deleted": deleted_review_files,
        "feedback_model_removed": feedback_removed,
        "auto_refined_pages": auto_refined,
        "category_counts": dict(
            sorted(Counter(row["category"] for row in after).items())
        ),
        "review_status_counts": dict(
            sorted(
                Counter(
                    str(row.get("taxonomy_review_status", ""))
                    for row in after
                ).items()
            )
        ),
        "review_queue_path": str(queue_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--library-dir",
        type=Path,
        default=DEFAULT_LIBRARY_DIR,
    )
    parser.add_argument(
        "--quality-reviews",
        type=Path,
        default=DEFAULT_QUALITY_REVIEWS,
    )
    parser.add_argument(
        "--query-reviews",
        type=Path,
        default=DEFAULT_QUERY_REVIEWS,
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    report = reset_manual_reviews(
        arguments.library_dir,
        arguments.quality_reviews,
        arguments.query_reviews,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
