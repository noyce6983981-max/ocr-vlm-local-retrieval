"""Independently audit the three-stage 1,500-page public research set."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGES = (
    (
        "initial_200",
        PROJECT_ROOT / "data/incoming/200页全部放这里",
        PROJECT_ROOT / "data/evaluation/public_dataset_200_sources.csv",
    ),
    (
        "document_expansion_300",
        PROJECT_ROOT / "data/incoming/expansion_300_real",
        PROJECT_ROOT
        / "data/evaluation/public_dataset_expansion_300_sources.csv",
    ),
    (
        "balanced_expansion_1000",
        PROJECT_ROOT / "data/incoming/expansion_1000_balanced",
        PROJECT_ROOT
        / "data/evaluation/public_dataset_balanced_expansion_1000_sources.csv",
    ),
)
OUTPUT_PATH = (
    PROJECT_ROOT / "data/evaluation/public_dataset_1500_audit.json"
)
EXPECTED_COUNTS = {
    "clear_document": 225,
    "complex_academic": 225,
    "table_form_ticket": 225,
    "ppt_poster_slide": 188,
    "software_web_code": 150,
    "scene_text": 187,
    "degraded_document": 150,
    "natural_no_text": 150,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def difference_hash(path: Path) -> int:
    with Image.open(path) as image:
        grayscale = ImageOps.grayscale(
            ImageOps.exif_transpose(image)
        ).resize((9, 8))
    if hasattr(grayscale, "get_flattened_data"):
        pixels = list(grayscale.get_flattened_data())
    else:
        pixels = list(grayscale.get_flattened_data())
    value = 0
    for row in range(8):
        for column in range(8):
            left = pixels[row * 9 + column]
            right = pixels[row * 9 + column + 1]
            value = (value << 1) | int(left > right)
    return value


def main() -> None:
    records: list[dict[str, Any]] = []
    stage_counts: dict[str, int] = {}
    missing_files: list[str] = []
    untracked_files: list[str] = []

    for stage, image_dir, manifest_path in STAGES:
        rows = read_csv(manifest_path)
        stage_counts[stage] = len(rows)
        manifest_filenames = {row["filename"] for row in rows}
        disk_filenames = {path.name for path in image_dir.glob("*.jpg")}
        missing_files.extend(
            f"{stage}:{filename}"
            for filename in sorted(manifest_filenames - disk_filenames)
        )
        untracked_files.extend(
            f"{stage}:{filename}"
            for filename in sorted(disk_filenames - manifest_filenames)
        )
        for row in rows:
            path = image_dir / row["filename"]
            if not path.is_file():
                continue
            payload = path.read_bytes()
            records.append(
                {
                    "stage": stage,
                    "filename": row["filename"],
                    "category": row["category"],
                    "source_name": row["source_name"],
                    "license": row["license"],
                    "ai_generated": row.get("ai_generated", "unknown"),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "dhash": difference_hash(path),
                    "size": len(payload),
                }
            )

    sha_groups: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        sha_groups.setdefault(row["sha256"], []).append(row)
    exact_duplicate_groups = [
        [
            f"{row['stage']}:{row['filename']}"
            for row in group
        ]
        for group in sha_groups.values()
        if len(group) > 1
    ]

    near_pairs: list[dict[str, Any]] = []
    new_near_pairs: list[dict[str, Any]] = []
    for first_index, first in enumerate(records):
        for second in records[first_index + 1 :]:
            distance = (first["dhash"] ^ second["dhash"]).bit_count()
            if distance > 5:
                continue
            pair = {
                "first": f"{first['stage']}:{first['filename']}",
                "second": f"{second['stage']}:{second['filename']}",
                "hamming_distance": distance,
            }
            near_pairs.append(pair)
            if (
                first["stage"] != "initial_200"
                or second["stage"] != "initial_200"
            ):
                new_near_pairs.append(pair)

    category_counts = Counter(row["category"] for row in records)
    source_counts = Counter(row["source_name"] for row in records)
    license_counts = Counter(row["license"] for row in records)
    ai_counts = Counter(row["ai_generated"] for row in records)
    report = {
        "status": (
            "passed"
            if (
                len(records) == 1500
                and dict(category_counts) == EXPECTED_COUNTS
                and not missing_files
                and not untracked_files
                and not exact_duplicate_groups
                and not new_near_pairs
            )
            else "failed"
        ),
        "page_count": len(records),
        "stage_counts": stage_counts,
        "category_counts": dict(category_counts),
        "expected_category_counts": EXPECTED_COUNTS,
        "source_counts": dict(source_counts),
        "license_counts": dict(license_counts),
        "ai_generated_value_counts": dict(ai_counts),
        "missing_files": missing_files,
        "untracked_files": untracked_files,
        "exact_duplicate_groups": exact_duplicate_groups,
        "all_perceptual_near_pairs": near_pairs,
        "near_pairs_involving_new_1300": new_near_pairs,
        "total_size_mb": round(
            sum(row["size"] for row in records) / 1024**2, 2
        ),
    }
    OUTPUT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
