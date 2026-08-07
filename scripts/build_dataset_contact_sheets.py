"""Build labeled contact sheets for fast human review of the 200-page set."""

from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = PROJECT_ROOT / "data/incoming/200页全部放这里"
MANIFEST = PROJECT_ROOT / "data/evaluation/public_dataset_200_sources.csv"
OUTPUT_DIR = PROJECT_ROOT / "work/dataset_200_contact_sheets"


def main() -> None:
    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    columns = 5
    rows_per_sheet = 5
    cell_width = 260
    cell_height = 230
    image_height = 195

    for sheet_index, start in enumerate(
        range(0, len(rows), columns * rows_per_sheet), start=1
    ):
        batch = rows[start : start + columns * rows_per_sheet]
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
                    (cell_width - 12, image_height - 8),
                )
            image_x = x + (cell_width - thumbnail.width) // 2
            image_y = y + 4
            sheet.paste(thumbnail, (image_x, image_y))
            label = f"{start + offset + 1:03d} {row['filename']}"
            draw.text((x + 5, y + image_height + 4), label, fill="black", font=font)
            draw.rectangle(
                (x, y, x + cell_width - 1, y + cell_height - 1),
                outline="#999999",
            )
        output = OUTPUT_DIR / f"sheet_{sheet_index:02d}.jpg"
        sheet.save(output, quality=90)
        print(output)


if __name__ == "__main__":
    main()
