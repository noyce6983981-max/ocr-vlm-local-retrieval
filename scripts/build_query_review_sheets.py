"""Render query, expected image, ranks, and reasons for fast review."""

from __future__ import annotations

import csv
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = PROJECT_ROOT / "data/incoming/200页全部放这里"
QUEUE_PATH = (
    PROJECT_ROOT
    / "data/evaluation/public_dataset_200_query_review_queue_30.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "work/query_review_queue_30_sheets"


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("C:/Windows/Fonts/msyh.ttc")
    return (
        ImageFont.truetype(str(path), size)
        if path.is_file()
        else ImageFont.load_default()
    )


def main() -> None:
    with QUEUE_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    title_font = font(21)
    text_font = font(18)
    alert_font = font(17)
    columns = 2
    rows_per_sheet = 4
    cell_width = 760
    cell_height = 650
    image_height = 430
    per_sheet = columns * rows_per_sheet

    for sheet_number, start in enumerate(
        range(0, len(rows), per_sheet), start=1
    ):
        batch = rows[start : start + per_sheet]
        sheet = Image.new(
            "RGB",
            (columns * cell_width, rows_per_sheet * cell_height),
            "white",
        )
        draw = ImageDraw.Draw(sheet)
        for offset, row in enumerate(batch):
            x = (offset % columns) * cell_width
            y = (offset // columns) * cell_height
            with Image.open(IMAGE_DIR / row["filename"]) as source:
                thumbnail = ImageOps.contain(
                    ImageOps.exif_transpose(source).convert("RGB"),
                    (cell_width - 18, image_height - 8),
                )
            sheet.paste(
                thumbnail,
                (
                    x + (cell_width - thumbnail.width) // 2,
                    y + 4,
                ),
            )
            draw.text(
                (x + 8, y + image_height + 3),
                f"{start + offset + 1:02d} {row['query_id']} "
                f"{row['filename']}",
                fill="black",
                font=title_font,
            )
            ranks = (
                f"rank T/V/A={row['diagnostic_text_rank'] or '-'}"
                f"/{row['diagnostic_visual_rank'] or '-'}"
                f"/{row['diagnostic_adaptive_rank'] or '-'}"
            )
            draw.text(
                (x + 8, y + image_height + 34),
                ranks,
                fill="#333333",
                font=text_font,
            )
            query = row["query"] or "【查询为空：需要人工编写或排除】"
            wrapped = textwrap.wrap(query, width=34)[:3]
            for line_number, line in enumerate(wrapped):
                draw.text(
                    (
                        x + 8,
                        y + image_height + 65 + line_number * 27,
                    ),
                    line,
                    fill="#153B66",
                    font=text_font,
                )
            draw.text(
                (x + 8, y + image_height + 150),
                (
                    "复核原因: "
                    + row["diagnostic_review_reasons"][:68]
                ),
                fill="#9C2C2C",
                font=alert_font,
            )
            draw.rectangle(
                (x, y, x + cell_width - 1, y + cell_height - 1),
                outline="#888888",
                width=2,
            )
        output_path = (
            OUTPUT_DIR / f"query_review_sheet_{sheet_number:02d}.jpg"
        )
        sheet.save(output_path, quality=92)
        print(output_path)


if __name__ == "__main__":
    main()
