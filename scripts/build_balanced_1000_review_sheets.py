"""Build four stratified review sheets for the balanced 1,000-page expansion."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = PROJECT_ROOT / "data/incoming/expansion_1000_balanced"
SOURCE_CSV = (
    PROJECT_ROOT
    / "data/evaluation/public_dataset_balanced_expansion_1000_sources.csv"
)
REVIEW_CSV = (
    PROJECT_ROOT
    / "data/evaluation/public_dataset_balanced_expansion_1000_review_sample.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "work/balanced_1000_review_sheets"


def evenly_spaced(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    if len(rows) <= count:
        return rows
    return [
        rows[round(index * (len(rows) - 1) / (count - 1))]
        for index in range(count)
    ]


def main() -> None:
    with SOURCE_CSV.open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    by_category: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)

    sample: list[dict[str, str]] = []
    for category in sorted(by_category):
        sample.extend(evenly_spaced(by_category[category], 10))
    for index, row in enumerate(sample, start=1):
        row["review_index"] = str(index)
        row["human_category_judgement"] = ""
        row["human_privacy_judgement"] = ""
        row["human_notes"] = ""

    REVIEW_CSV.parent.mkdir(parents=True, exist_ok=True)
    with REVIEW_CSV.open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sample[0]))
        writer.writeheader()
        writer.writerows(sample)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    columns = 4
    rows_per_sheet = 5
    per_sheet = columns * rows_per_sheet
    cell_width = 360
    cell_height = 300
    image_height = 240

    for sheet_index, start in enumerate(
        range(0, len(sample), per_sheet), start=1
    ):
        batch = sample[start : start + per_sheet]
        sheet = Image.new(
            "RGB",
            (columns * cell_width, rows_per_sheet * cell_height),
            "white",
        )
        draw = ImageDraw.Draw(sheet)
        for offset, row in enumerate(batch):
            x = (offset % columns) * cell_width
            y = (offset // columns) * cell_height
            with Image.open(IMAGE_DIR / row["filename"]) as opened:
                thumbnail = ImageOps.contain(
                    ImageOps.exif_transpose(opened).convert("RGB"),
                    (cell_width - 16, image_height - 8),
                )
            image_x = x + (cell_width - thumbnail.width) // 2
            sheet.paste(thumbnail, (image_x, y + 4))
            draw.rectangle(
                (x, y, x + cell_width - 1, y + cell_height - 1),
                outline="#b8b8b8",
                width=1,
            )
            draw.text(
                (x + 8, y + image_height + 2),
                f"{row['review_index']} {row['filename']}",
                fill="black",
            )
            draw.text(
                (x + 8, y + image_height + 20),
                f"{row['category']} | {row['source_name']}",
                fill="#333333",
            )
        output_path = OUTPUT_DIR / f"review_sheet_{sheet_index:02d}.jpg"
        sheet.save(output_path, quality=90)
        print(output_path)


if __name__ == "__main__":
    main()
