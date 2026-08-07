"""Permanently purge one source and its page-level derivatives from a library."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIBRARY_DIR = PROJECT_ROOT / "outputs/user_library"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
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


def filter_csv_atomic(
    path: Path,
    *,
    key: str,
    removed_values: set[str],
) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if key not in fields:
        return 0
    kept = [
        row for row in rows if str(row.get(key, "")) not in removed_values
    ]
    removed_count = len(rows) - len(kept)
    if not removed_count:
        return 0
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(kept)
    os.replace(temporary, path)
    return removed_count


def project_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    resolved = path.resolve()
    if not resolved.is_relative_to(PROJECT_ROOT.resolve()):
        raise ValueError(f"Refusing to delete outside project: {resolved}")
    return resolved


def purge_source(
    library_dir: Path,
    *,
    source_group_id: str | None = None,
    source_file_name: str | None = None,
) -> dict[str, Any]:
    library_dir = library_dir.resolve()
    if not library_dir.is_relative_to(PROJECT_ROOT.resolve()):
        raise ValueError("Library must be inside the project.")
    manifest_path = library_dir / "manifest.jsonl"
    rows = read_jsonl(manifest_path)
    matched = [
        row
        for row in rows
        if (
            source_group_id
            and row.get("source_group_id") == source_group_id
        )
        or (
            source_file_name
            and row.get("source_file_name") == source_file_name
        )
    ]
    if not matched:
        raise ValueError("No manifest pages matched the exact source selector.")

    item_ids = {str(row["item_id"]) for row in matched}
    source_group_ids = {
        str(row.get("source_group_id", ""))
        for row in matched
        if row.get("source_group_id")
    }
    write_jsonl_atomic(
        manifest_path,
        [row for row in rows if row["item_id"] not in item_ids],
    )

    sources_path = library_dir / "sources.jsonl"
    source_rows = read_jsonl(sources_path)
    removed_source_rows = [
        row
        for row in source_rows
        if str(row.get("source_group_id", "")) in source_group_ids
    ]
    write_jsonl_atomic(
        sources_path,
        [
            row
            for row in source_rows
            if str(row.get("source_group_id", ""))
            not in source_group_ids
        ],
    )

    csv_removals: dict[str, int] = {}
    for relative_path in [
        "ocr/summary.csv",
        "taxonomy/taxonomy_v2_review_queue.csv",
        "feedback_model/active_learning_queue.csv",
    ]:
        path = library_dir / relative_path
        csv_removals[relative_path] = filter_csv_atomic(
            path,
            key="item_id",
            removed_values=item_ids,
        )

    candidate_paths: set[Path] = set()
    for row in matched:
        candidate_paths.add(project_path(str(row["source_path"])))
        item_id = str(row["item_id"])
        candidate_paths.update(
            {
                library_dir / "ocr/json" / f"{item_id}.json",
                library_dir / "ocr/visualizations" / f"{item_id}.png",
                library_dir / "ocr/visualizations" / f"{item_id}.jpg",
                library_dir / "ocr/overrides" / f"{item_id}.json",
            }
        )
    for row in removed_source_rows:
        if row.get("stored_path"):
            candidate_paths.add(project_path(str(row["stored_path"])))

    deleted_files = 0
    deleted_bytes = 0
    for path in sorted(candidate_paths):
        resolved = project_path(path)
        if resolved.is_file():
            deleted_bytes += resolved.stat().st_size
            resolved.unlink()
            deleted_files += 1

    return {
        "status": "purged",
        "removed_page_count": len(matched),
        "removed_item_ids": sorted(item_ids),
        "removed_source_group_ids": sorted(source_group_ids),
        "deleted_file_count": deleted_files,
        "deleted_bytes": deleted_bytes,
        "csv_rows_removed": csv_removals,
        "indexes_need_compaction": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--library-dir",
        type=Path,
        default=DEFAULT_LIBRARY_DIR,
    )
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--source-group-id")
    selector.add_argument("--source-file-name")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    report = purge_source(
        arguments.library_dir,
        source_group_id=arguments.source_group_id,
        source_file_name=arguments.source_file_name,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
