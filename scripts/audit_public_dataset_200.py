"""Add perceptual-duplicate groups and review flags to the 200-page manifest."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = PROJECT_ROOT / "data/incoming/200页全部放这里"
MANIFEST_PATH = (
    PROJECT_ROOT / "data/evaluation/public_dataset_200_sources.csv"
)
AUDIT_PATH = (
    PROJECT_ROOT / "data/evaluation/public_dataset_200_audit.json"
)
PRIVACY_REVIEW = {
    "complex_academic_014.jpg",
    "complex_academic_018.jpg",
    "complex_academic_019.jpg",
    "complex_academic_020.jpg",
    "software_web_code_005.jpg",
}


def difference_hash(path: Path) -> int:
    with Image.open(path) as image:
        grayscale = ImageOps.grayscale(
            ImageOps.exif_transpose(image)
        ).resize((9, 8))
    pixels = list(grayscale.get_flattened_data())
    value = 0
    for row in range(8):
        for column in range(8):
            left = pixels[row * 9 + column]
            right = pixels[row * 9 + column + 1]
            value = (value << 1) | int(left > right)
    return value


def find(parent: list[int], index: int) -> int:
    while parent[index] != index:
        parent[index] = parent[parent[index]]
        index = parent[index]
    return index


def union(parent: list[int], first: int, second: int) -> None:
    first_root = find(parent, first)
    second_root = find(parent, second)
    if first_root != second_root:
        parent[second_root] = first_root


def category_review_required(filename: str) -> bool:
    stem = Path(filename).stem
    index = int(stem.rsplit("_", 1)[1])
    if stem.startswith("complex_academic_") and 11 <= index <= 25:
        return True
    if stem.startswith("ppt_poster_slide_") and 5 <= index <= 25:
        return True
    return stem.startswith("software_web_code_") and index <= 5


def main() -> None:
    with MANIFEST_PATH.open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 200:
        raise RuntimeError(f"Expected 200 rows, found {len(rows)}.")

    hashes = [
        difference_hash(IMAGE_DIR / row["filename"]) for row in rows
    ]
    parent = list(range(len(rows)))
    for first in range(len(rows)):
        for second in range(first + 1, len(rows)):
            if (hashes[first] ^ hashes[second]).bit_count() <= 5:
                union(parent, first, second)

    members_by_root: dict[int, list[int]] = {}
    for index in range(len(rows)):
        members_by_root.setdefault(find(parent, index), []).append(index)
    duplicate_groups = [
        members
        for members in members_by_root.values()
        if len(members) >= 2
    ]
    duplicate_groups.sort(key=lambda members: members[0])
    group_by_index: dict[int, str] = {}
    for group_number, members in enumerate(duplicate_groups, start=1):
        group_id = f"visual_near_duplicate_{group_number:03d}"
        for index in members:
            group_by_index[index] = group_id

    for index, row in enumerate(rows):
        row["perceptual_group"] = group_by_index.get(index, "")
        row["privacy_review_required"] = str(
            row["filename"] in PRIVACY_REVIEW
        ).lower()
        row["category_review_required"] = str(
            category_review_required(row["filename"])
        ).lower()

    with MANIFEST_PATH.open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    audit = {
        "page_count": len(rows),
        "perceptual_group_count": len(duplicate_groups),
        "pages_in_perceptual_groups": len(group_by_index),
        "largest_perceptual_group": max(
            (len(members) for members in duplicate_groups), default=1
        ),
        "privacy_review_required": sorted(PRIVACY_REVIEW),
        "privacy_review_count": sum(
            row["privacy_review_required"] == "true" for row in rows
        ),
        "category_review_count": sum(
            row["category_review_required"] == "true" for row in rows
        ),
        "status": "候选集已自动审计，仍待人工最终确认",
    }
    AUDIT_PATH.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
