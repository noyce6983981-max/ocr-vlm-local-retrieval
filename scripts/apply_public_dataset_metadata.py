"""Attach public-dataset provenance and review flags to an ingested library."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATEGORY_LABELS = {
    "clear_document": "清晰文档",
    "complex_academic": "复杂学术页面",
    "table_form_ticket": "表格表单票据",
    "ppt_poster_slide": "幻灯片与海报",
    "software_web_code": "软件网页代码",
    "scene_text": "场景文字",
    "degraded_document": "退化文档",
    "natural_no_text": "自然图像负样本",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-csv",
        type=Path,
        default=Path("data/evaluation/public_dataset_200_sources.csv"),
    )
    parser.add_argument(
        "--library-dir",
        type=Path,
        default=Path("outputs/user_library"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "data/evaluation/public_dataset_200_metadata_report.json"
        ),
    )
    return parser.parse_args()


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no", ""}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temp_path, path)


def public_fields(source: dict[str, str]) -> dict[str, Any]:
    category = source["category"]
    if category not in CATEGORY_LABELS:
        raise ValueError(f"Unknown public-dataset category: {category}")
    quality_tags = [
        value.strip()
        for value in source.get("quality_tags", "").split(",")
        if value.strip()
    ]
    return {
        "category": category,
        "has_text_expected": parse_bool(source["has_text"]),
        "review_status": source["review_status"],
        "dataset_name": source.get(
            "dataset_name", "public_dataset_200_candidate"
        ),
        "language": source["language"],
        "quality_tags": quality_tags,
        "public_source_name": source["source_name"],
        "public_source_url": source["source_url"],
        "public_source_file": source["source_file"],
        "license": source["license"],
        "license_url": source["license_url"],
        "author": source.get("author", ""),
        "origin_type": source.get("origin_type", ""),
        "ai_generated": parse_bool(source.get("ai_generated", "false")),
        "declared_ai_generated_percent": float(
            source.get("declared_ai_generated_percent", "0") or 0
        ),
        "hard_negative_group": source.get("hard_negative_group", ""),
        "perceptual_group": source.get("perceptual_group", ""),
        "privacy_review_required": parse_bool(
            source.get("privacy_review_required", "false")
        ),
        "category_review_required": parse_bool(
            source.get("category_review_required", "false")
        ),
    }


def enrich_manifest(
    manifest: list[dict[str, Any]],
    source_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    by_filename = {row["filename"]: row for row in source_rows}
    if len(by_filename) != len(source_rows):
        raise ValueError("Source CSV contains duplicate filenames.")

    enriched: list[dict[str, Any]] = []
    matched: set[str] = set()
    for row in manifest:
        filename = row.get("source_file_name")
        source = by_filename.get(str(filename))
        if source is None:
            enriched.append(dict(row))
            continue
        matched.add(str(filename))
        updated = {**row, **public_fields(source)}
        stem_number = Path(str(filename)).stem.rsplit("_", 1)[-1]
        updated["display_name_zh"] = (
            f"{CATEGORY_LABELS[source['category']]} {stem_number}"
        )
        updated["source"] = source["source_name"]
        enriched.append(updated)

    missing = sorted(set(by_filename) - matched)
    if missing:
        raise ValueError(
            f"{len(missing)} public pages are absent from the library: "
            + ", ".join(missing[:5])
        )
    return enriched


def enrich_rows_by_item_id(
    rows: list[dict[str, Any]],
    manifest_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    fields = (
        "display_name_zh",
        "category",
        "dataset_name",
        "review_status",
        "has_text_expected",
        "language",
        "quality_tags",
        "public_source_name",
        "public_source_url",
        "license",
        "license_url",
        "author",
        "origin_type",
        "ai_generated",
        "declared_ai_generated_percent",
        "privacy_review_required",
        "category_review_required",
    )
    enriched: list[dict[str, Any]] = []
    for row in rows:
        manifest_row = manifest_by_id.get(str(row.get("item_id")))
        if manifest_row is None:
            enriched.append(dict(row))
            continue
        enriched.append(
            {
                **row,
                **{
                    field: manifest_row[field]
                    for field in fields
                    if field in manifest_row
                },
            }
        )
    return enriched


def update_summary_csv(
    path: Path,
    manifest_by_id: dict[str, dict[str, Any]],
) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    updated = 0
    for row in rows:
        manifest_row = manifest_by_id.get(row["item_id"])
        if manifest_row is None:
            continue
        row["display_name_zh"] = manifest_row["display_name_zh"]
        row["category"] = manifest_row["category"]
        updated += 1
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp_path, path)
    return updated


def main() -> None:
    args = parse_args()
    source_csv = project_path(args.source_csv)
    library_dir = project_path(args.library_dir)
    report_path = project_path(args.report)

    with source_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    manifest_path = library_dir / "manifest.jsonl"
    manifest = enrich_manifest(read_jsonl(manifest_path), source_rows)
    manifest_by_id = {row["item_id"]: row for row in manifest}

    metadata_paths = [
        library_dir / "text_index/metadata.jsonl",
        library_dir / "visual_index/metadata.jsonl",
    ]
    enriched_metadata = {
        path: enrich_rows_by_item_id(read_jsonl(path), manifest_by_id)
        for path in metadata_paths
    }
    sources_path = library_dir / "sources.jsonl"
    sources = read_jsonl(sources_path)
    source_by_filename = {row["filename"]: row for row in source_rows}
    enriched_sources = []
    for row in sources:
        public_source = source_by_filename.get(str(row["original_name"]))
        enriched_sources.append(
            {
                **row,
                **(
                    public_fields(public_source)
                    if public_source is not None
                    else {}
                ),
            }
        )

    write_jsonl_atomic(manifest_path, manifest)
    for path, rows in enriched_metadata.items():
        write_jsonl_atomic(path, rows)
    write_jsonl_atomic(sources_path, enriched_sources)
    summary_updated = update_summary_csv(
        library_dir / "ocr/summary.csv", manifest_by_id
    )

    public_rows = [
        row
        for row in manifest
        if row.get("public_source_name")
    ]
    report = {
        "status": "success",
        "source_rows": len(source_rows),
        "public_library_pages": len(public_rows),
        "dataset_counts": dict(
            sorted(
                {
                    name: sum(
                        row.get("dataset_name") == name
                        for row in public_rows
                    )
                    for name in {
                        str(row.get("dataset_name", ""))
                        for row in public_rows
                    }
                }.items()
            )
        ),
        "summary_rows_updated": summary_updated,
        "privacy_review_required": sum(
            bool(row["privacy_review_required"]) for row in public_rows
        ),
        "category_review_required": sum(
            bool(row["category_review_required"]) for row in public_rows
        ),
        "no_text_negative_pages": sum(
            not bool(row["has_text_expected"]) for row in public_rows
        ),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
