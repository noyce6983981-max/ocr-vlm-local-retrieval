"""Validate, OCR, and incrementally index one image in the user library."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PADDLE_PYTHON = PROJECT_ROOT / ".venv-paddle/Scripts/python.exe"
TEXT_PYTHON = PROJECT_ROOT / ".venv/Scripts/python.exe"
VISUAL_PYTHON = PROJECT_ROOT / ".venv-vl/Scripts/python.exe"
LIBRARY_DIR = PROJECT_ROOT / "outputs/user_library"
MANIFEST_PATH = LIBRARY_DIR / "manifest.jsonl"
SUMMARY_PATH = LIBRARY_DIR / "ocr/summary.csv"
ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SUMMARY_FIELDS = [
    "item_id",
    "display_name_zh",
    "category",
    "source_path",
    "status",
    "text_box_count",
    "character_count",
    "mean_confidence",
    "min_confidence",
    "json_path",
    "visualization_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--display-name", required=True)
    parser.add_argument("--category", default="personal_document")
    return parser.parse_args()


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def content_item_id(image_bytes: bytes) -> str:
    return f"user_{hashlib.sha256(image_bytes).hexdigest()[:12]}"


def run_checked(command: list[str], label: str) -> None:
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout)[-2000:]
        raise RuntimeError(f"{label} failed:\n{detail}")


def ocr_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    texts = [str(text).strip() for text in payload.get("rec_texts", [])]
    scores = [float(score) for score in payload.get("rec_scores", [])]
    return {
        "text_box_count": len(texts),
        "character_count": sum(len(text) for text in texts),
        "mean_confidence": (
            round(statistics.fmean(scores), 6) if scores else 0.0
        ),
        "min_confidence": round(min(scores), 6) if scores else 0.0,
    }


def append_summary(row: dict[str, Any]) -> None:
    existing: list[dict[str, str]] = []
    if SUMMARY_PATH.is_file():
        with SUMMARY_PATH.open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            existing = list(csv.DictReader(handle))
    existing = [
        current
        for current in existing
        if current["item_id"] != row["item_id"]
    ]
    existing.append(row)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_PATH.open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(existing)


def main() -> None:
    args = parse_args()
    source = project_path(args.image).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Image not found: {source}")
    if source.suffix.lower() not in ALLOWED_SUFFIXES:
        raise ValueError("Only JPG, PNG, WEBP, and BMP images are supported.")
    display_name = " ".join(args.display_name.split())
    if not display_name:
        raise ValueError("Display name cannot be empty.")

    image_bytes = source.read_bytes()
    item_id = content_item_id(image_bytes)
    existing = read_jsonl(MANIFEST_PATH)
    duplicate = next(
        (row for row in existing if row["item_id"] == item_id), None
    )
    if duplicate is not None:
        print(
            json.dumps(
                {"status": "duplicate", "item": duplicate},
                ensure_ascii=False,
            )
        )
        return

    with Image.open(source) as opened:
        opened.verify()

    images_dir = LIBRARY_DIR / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    final_image = images_dir / f"{item_id}{source.suffix.lower()}"
    final_image.write_bytes(image_bytes)
    relative_image = final_image.relative_to(PROJECT_ROOT).as_posix()

    item = {
        "item_id": item_id,
        "source_path": relative_image,
        "display_name_zh": display_name,
        "category": args.category,
        "has_text_expected": True,
        "review_status": "待人工审核",
        "source": "user_upload",
    }

    with tempfile.TemporaryDirectory(
        prefix=f"{item_id}_", dir=LIBRARY_DIR
    ) as temporary:
        staging = Path(temporary)
        one_manifest = staging / "manifest.jsonl"
        one_manifest.write_text(
            json.dumps(item, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        ocr_output = staging / "ocr"
        run_checked(
            [
                str(PADDLE_PYTHON),
                str(PROJECT_ROOT / "scripts/batch_ocr.py"),
                "--manifest",
                str(one_manifest),
                "--output",
                str(ocr_output),
                "--device",
                "gpu:0",
            ],
            "OCR",
        )
        ocr_json = ocr_output / "json" / f"{item_id}.json"
        visualization_candidates = list(
            (ocr_output / "visualizations").glob(f"{item_id}.*")
        )
        if not ocr_json.is_file() or not visualization_candidates:
            raise RuntimeError("OCR did not produce the expected artifacts.")

        common = [
            "--item-id",
            item_id,
            "--display-name",
            display_name,
            "--category",
            args.category,
            "--source-path",
            relative_image,
        ]
        run_checked(
            [
                str(TEXT_PYTHON),
                str(PROJECT_ROOT / "scripts/index_user_text.py"),
                *common,
                "--ocr-json",
                str(ocr_json),
            ],
            "text indexing",
        )
        run_checked(
            [
                str(VISUAL_PYTHON),
                str(PROJECT_ROOT / "scripts/index_user_visual.py"),
                *common,
            ],
            "visual indexing",
        )

        json_dir = LIBRARY_DIR / "ocr/json"
        visualization_dir = LIBRARY_DIR / "ocr/visualizations"
        json_dir.mkdir(parents=True, exist_ok=True)
        visualization_dir.mkdir(parents=True, exist_ok=True)
        final_json = json_dir / ocr_json.name
        final_visualization = (
            visualization_dir / visualization_candidates[0].name
        )
        shutil.copy2(ocr_json, final_json)
        shutil.copy2(visualization_candidates[0], final_visualization)

    payload = json.loads(final_json.read_text(encoding="utf-8"))
    metrics = ocr_metrics(payload)
    item["has_text_expected"] = metrics["text_box_count"] > 0
    with MANIFEST_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary_row = {
        "item_id": item_id,
        "display_name_zh": display_name,
        "category": args.category,
        "source_path": relative_image,
        "status": "success",
        **metrics,
        "json_path": final_json.relative_to(PROJECT_ROOT).as_posix(),
        "visualization_path": final_visualization.relative_to(
            PROJECT_ROOT
        ).as_posix(),
    }
    append_summary(summary_row)
    print(
        json.dumps(
            {"status": "success", "item": item, "ocr": metrics},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
