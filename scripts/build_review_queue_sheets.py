"""Render the 30-page OCR review queue as three readable contact sheets."""

from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = PROJECT_ROOT / "data/incoming/200页全部放这里"
QUEUE_PATH = (
    PROJECT_ROOT
    / "data/evaluation/public_dataset_200_review_queue_30.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "work/review_queue_30_sheets"


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    if font_path.is_file():
        return ImageFont.truetype(str(font_path), size)
    return ImageFont.load_default()


def main() -> None:
    with QUEUE_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    title_font = load_font(21)
    detail_font = load_font(17)
    columns = 2
    rows_per_sheet = 5
    cell_width = 700
    cell_height = 520
    image_height = 420
    pages_per_sheet = columns * rows_per_sheet

    for sheet_index, start in enumerate(
        range(0, len(rows), pages_per_sheet), start=1
    ):
        batch = rows[start : start + pages_per_sheet]
        sheet = Image.new(
            "RGB",
            (columns * cell_width, rows_per_sheet * cell_height),
            "white",
        )
        draw = ImageDraw.Draw(sheet)
        for offset, row in enumerate(batch):
            column = offset % columns
            line = offset // columns
            x = column * cell_width
            y = line * cell_height
            with Image.open(IMAGE_DIR / row["filename"]) as source:
                thumbnail = ImageOps.contain(
                    source.convert("RGB"),
                    (cell_width - 16, image_height - 8),
                )
            image_x = x + (cell_width - thumbnail.width) // 2
            image_y = y + 4
            sheet.paste(thumbnail, (image_x, image_y))
            draw.text(
                (x + 8, y + image_height + 3),
                f"{start + offset + 1:02d}  {row['filename']}",
                fill="black",
                font=title_font,
            )
            detail = (
                f"{row['category']} | conf={row['mean_confidence']} | "
                f"chars={row['character_count']} | risk={row['risk_score']}"
            )
            draw.text(
                (x + 8, y + image_height + 34),
                detail,
                fill="#333333",
                font=detail_font,
            )
            draw.text(
                (x + 8, y + image_height + 62),
                f"原因: {row['review_reason']}",
                fill="#9C2C2C",
                font=detail_font,
            )
            draw.rectangle(
                (x, y, x + cell_width - 1, y + cell_height - 1),
                outline="#888888",
                width=2,
            )
        output_path = OUTPUT_DIR / f"review_sheet_{sheet_index:02d}.jpg"
        sheet.save(output_path, quality=92)
        print(output_path)


if __name__ == "__main__":
    main()
