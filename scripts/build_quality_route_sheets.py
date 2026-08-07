"""Render OCR quality-route candidates into compact human review sheets."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--library-dir",
        type=Path,
        default=Path("outputs/user_library"),
    )
    parser.add_argument(
        "--quality-csv",
        type=Path,
        default=Path(
            "data/evaluation/public_dataset_1500_quality_gate.csv"
        ),
    )
    parser.add_argument(
        "--routes",
        nargs="+",
        default=["category_text_leak", "category_review"],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("work/quality_route_sheets"),
    )
    return parser.parse_args()


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    args = parse_args()
    library_dir = project_path(args.library_dir)
    quality_csv = project_path(args.quality_csv)
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        row["item_id"]: row
        for row in read_jsonl(library_dir / "manifest.jsonl")
    }
    with quality_csv.open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        candidates = [
            row
            for row in csv.DictReader(handle)
            if row["quality_route"] in set(args.routes)
        ]

    columns = 4
    rows_per_sheet = 4
    cell_width = 380
    cell_height = 330
    per_sheet = columns * rows_per_sheet
    for sheet_number, start in enumerate(
        range(0, len(candidates), per_sheet), start=1
    ):
        batch = candidates[start : start + per_sheet]
        sheet = Image.new(
            "RGB",
            (columns * cell_width, rows_per_sheet * cell_height),
            "white",
        )
        draw = ImageDraw.Draw(sheet)
        for offset, row in enumerate(batch):
            item = manifest[row["item_id"]]
            path = PROJECT_ROOT / item["source_path"]
            with Image.open(path) as opened:
                thumbnail = ImageOps.contain(
                    ImageOps.exif_transpose(opened).convert("RGB"),
                    (cell_width - 16, 260),
                )
            x = (offset % columns) * cell_width
            y = (offset // columns) * cell_height
            sheet.paste(
                thumbnail,
                (x + (cell_width - thumbnail.width) // 2, y + 4),
            )
            draw.rectangle(
                (x, y, x + cell_width - 1, y + cell_height - 1),
                outline="#b8b8b8",
            )
            draw.text(
                (x + 8, y + 268),
                f"{start + offset + 1} {row['filename']}",
                fill="black",
            )
            draw.text(
                (x + 8, y + 286),
                (
                    f"{row['quality_route']} | chars="
                    f"{row['character_count']} | conf="
                    f"{float(row['mean_confidence']):.2f}"
                ),
                fill="#333333",
            )
        output_path = output_dir / f"quality_sheet_{sheet_number:02d}.jpg"
        sheet.save(output_path, quality=92)
        print(output_path)
    print(f"candidates={len(candidates)}")


if __name__ == "__main__":
    main()
