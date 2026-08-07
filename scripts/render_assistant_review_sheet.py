"""Render a temporary blind contact sheet for assistant visual review."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CELL_WIDTH = 420
CELL_HEIGHT = 540
LABEL_HEIGHT = 40
COLUMNS = 4


def fit_image(image: Image.Image, width: int, height: int) -> Image.Image:
    converted = image.convert("RGB")
    converted.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), "white")
    left = (width - converted.width) // 2
    top = (height - converted.height) // 2
    canvas.paste(converted, (left, top))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", type=int, choices=(4, 5), required=True)
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        required=True,
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    queue_path = (
        PROJECT_ROOT
        / "data/evaluation"
        / f"assistant_full_review_round{args.round}_queue.csv"
    )
    with queue_path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["split"] == args.split
        ]
    selected = rows[args.offset : args.offset + args.count]
    if not selected:
        raise ValueError("No rows selected.")

    sheet_rows = (len(selected) + COLUMNS - 1) // COLUMNS
    sheet = Image.new(
        "RGB",
        (CELL_WIDTH * COLUMNS, CELL_HEIGHT * sheet_rows),
        "#d9dee5",
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=22)
    for index, row in enumerate(selected):
        column = index % COLUMNS
        line = index // COLUMNS
        left = column * CELL_WIDTH
        top = line * CELL_HEIGHT
        image_path = PROJECT_ROOT / row["source_path"]
        with Image.open(image_path) as image:
            fitted = fit_image(
                ImageOps.exif_transpose(image),
                CELL_WIDTH - 12,
                CELL_HEIGHT - LABEL_HEIGHT - 12,
            )
        sheet.paste(fitted, (left + 6, top + LABEL_HEIGHT + 6))
        draw.rectangle(
            (left, top, left + CELL_WIDTH, top + LABEL_HEIGHT),
            fill="#10253b",
        )
        draw.text(
            (left + 12, top + 8),
            f"{row['blind_id']}  {row['item_id'][-6:]}",
            fill="white",
            font=font,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output, quality=90, optimize=True)
    print(args.output)


if __name__ == "__main__":
    main()
