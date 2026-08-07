"""Run a reproducible PaddleOCR batch experiment from a JSONL manifest."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

# Prefer a mainland-China model source and skip repeated connectivity checks.
os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "BOS")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

# Keep this import order. In this dedicated environment, Torch is CPU-only and
# Paddle owns the GPU, which avoids Windows CUDA DLL conflicts.
import torch  # noqa: F401
import paddle  # noqa: F401
from paddleocr import PaddleOCR


PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
    "elapsed_seconds",
    "json_path",
    "visualization_path",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PP-OCRv5 on all images listed in a JSONL manifest."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifest/pilot_manifest.jsonl"),
        help="JSONL manifest path, relative to the project root by default.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/ocr_batch"),
        help="Batch output directory.",
    )
    parser.add_argument("--device", default="gpu:0", help="Inference device.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run items even when their JSON result already exists.",
    )
    parser.add_argument(
        "--progress-json",
        type=Path,
        help="Optional atomic progress file for a product UI or job monitor.",
    )
    return parser.parse_args()


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_manifest(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            item = json.loads(line)
            item_id = item.get("item_id")
            source_path = item.get("source_path")
            if not item_id or not source_path:
                raise ValueError(
                    f"Manifest line {line_number} needs item_id and source_path."
                )
            if item_id in seen_ids:
                raise ValueError(f"Duplicate item_id in manifest: {item_id}")
            seen_ids.add(item_id)
            items.append(item)

    if not items:
        raise ValueError(f"Manifest is empty: {path}")
    return items


def load_result_metrics(json_path: Path) -> dict[str, Any]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    texts = [str(text) for text in payload.get("rec_texts", [])]
    scores = [float(score) for score in payload.get("rec_scores", [])]
    return {
        "text_box_count": len(texts),
        "character_count": sum(len(text.strip()) for text in texts),
        "mean_confidence": round(statistics.fmean(scores), 6) if scores else 0.0,
        "min_confidence": round(min(scores), 6) if scores else 0.0,
    }


def create_ocr(device: str) -> PaddleOCR:
    return PaddleOCR(
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="PP-OCRv5_mobile_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device=device,
        engine="paddle",
    )


def write_summary(rows: list[dict[str, Any]], output_dir: Path) -> None:
    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    failures_path = output_dir / "failures.jsonl"
    with failures_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            if row["status"] == "error":
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_progress(
    path: Path,
    *,
    processed: int,
    total: int,
    rows: list[dict[str, Any]],
) -> None:
    payload = {
        "phase": "ocr",
        "processed_pages": processed,
        "total_pages": total,
        "progress_percent": round(processed / total * 100, 2),
        "success_pages": sum(
            row["status"] in {"success", "cached"} for row in rows
        ),
        "error_pages": sum(row["status"] == "error" for row in rows),
        "updated_at_epoch": round(time.time(), 3),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    manifest_path = project_path(args.manifest).resolve()
    output_dir = project_path(args.output).resolve()
    json_dir = output_dir / "json"
    visualization_dir = output_dir / "visualizations"

    items = load_manifest(manifest_path)
    progress_path = (
        project_path(args.progress_json).resolve()
        if args.progress_json is not None
        else None
    )
    json_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading OCR models on {args.device} ...")
    ocr = create_ocr(args.device)
    rows: list[dict[str, Any]] = []

    for index, item in enumerate(items, start=1):
        item_id = item["item_id"]
        source_path = project_path(Path(item["source_path"])).resolve()
        json_path = json_dir / f"{item_id}.json"
        visualization_path = visualization_dir / f"{item_id}{source_path.suffix.lower()}"
        started = time.perf_counter()
        status = "cached"
        error = ""

        try:
            if not source_path.is_file():
                raise FileNotFoundError(f"Input image not found: {source_path}")

            if args.force or not json_path.exists():
                results = list(ocr.predict(str(source_path)))
                if len(results) != 1:
                    raise RuntimeError(
                        f"Expected one result for one image, got {len(results)}."
                    )
                result = results[0]
                result.save_to_json(str(json_path))
                result.save_to_img(str(visualization_path))
                status = "success"

            metrics = load_result_metrics(json_path)
        except Exception as exc:  # Continue so one bad sample cannot stop the batch.
            status = "error"
            error = f"{type(exc).__name__}: {exc}"
            metrics = {
                "text_box_count": 0,
                "character_count": 0,
                "mean_confidence": 0.0,
                "min_confidence": 0.0,
            }

        elapsed = round(time.perf_counter() - started, 3)
        row = {
            "item_id": item_id,
            "display_name_zh": item.get("display_name_zh", ""),
            "category": item.get("category", ""),
            "source_path": source_path.relative_to(PROJECT_ROOT).as_posix(),
            "status": status,
            **metrics,
            "elapsed_seconds": elapsed,
            "json_path": json_path.relative_to(PROJECT_ROOT).as_posix(),
            "visualization_path": visualization_path.relative_to(
                PROJECT_ROOT
            ).as_posix(),
            "error": error,
        }
        rows.append(row)
        if progress_path is not None and (
            index % 10 == 0 or index == len(items)
        ):
            write_progress(
                progress_path,
                processed=index,
                total=len(items),
                rows=rows,
            )
        print(
            f"[{index:02d}/{len(items):02d}] {item_id}: "
            f"{status}, boxes={row['text_box_count']}, "
            f"mean_conf={row['mean_confidence']:.4f}, time={elapsed:.3f}s"
        )

    write_summary(rows, output_dir)
    success_count = sum(row["status"] != "error" for row in rows)
    print(
        f"Batch completed: {success_count}/{len(rows)} processed without errors. "
        f"Summary: {output_dir / 'summary.csv'}"
    )


if __name__ == "__main__":
    main()
