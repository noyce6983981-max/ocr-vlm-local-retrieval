"""Retry uncertain text-bearing pages at four rotations with one OCR model load."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--review-queue",
        type=Path,
        default=Path(
            "data/evaluation/public_dataset_200_review_queue_30.csv"
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/user_library/manifest.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/evaluation/public_dataset_200_rotation_retry.csv"
        ),
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path("work/rotation_retry_001"),
    )
    parser.add_argument("--device", default="gpu:0")
    parser.add_argument("--max-items", type=int, default=15)
    return parser.parse_args()


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def result_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    texts = [str(value).strip() for value in payload.get("rec_texts", [])]
    scores = [float(value) for value in payload.get("rec_scores", [])]
    character_count = sum(len(value) for value in texts)
    mean_confidence = sum(scores) / len(scores) if scores else 0.0
    low_fraction = (
        sum(value < 0.6 for value in scores) / len(scores)
        if scores
        else 1.0
    )
    return {
        "text_box_count": len(texts),
        "character_count": character_count,
        "mean_confidence": mean_confidence,
        "low_confidence_fraction": low_fraction,
        "text_preview": " ".join(value for value in texts if value)[:160],
    }


def rotation_utility(metrics: dict[str, Any]) -> float:
    return (
        math.log1p(int(metrics["character_count"]))
        * float(metrics["mean_confidence"])
        * (1.0 - 0.5 * float(metrics["low_confidence_fraction"]))
    )


def should_accept_rotation(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> bool:
    if int(candidate["angle"]) == 0:
        return False
    if int(candidate["character_count"]) < 8:
        return False
    baseline_utility = max(rotation_utility(baseline), 1e-6)
    candidate_utility = rotation_utility(candidate)
    confidence_gain = (
        float(candidate["mean_confidence"])
        - float(baseline["mean_confidence"])
    )
    character_gain = int(candidate["character_count"]) / max(
        int(baseline["character_count"]), 1
    )
    material_gain = (
        character_gain >= 1.25 and confidence_gain >= 0.05
    ) or (
        character_gain >= 0.90 and confidence_gain >= 0.15
    )
    return candidate_utility >= baseline_utility * 1.15 and material_gain


def main() -> None:
    args = parse_args()
    queue_path = project_path(args.review_queue)
    manifest_path = project_path(args.manifest)
    output_path = project_path(args.output)
    artifacts_dir = project_path(args.artifacts)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    with queue_path.open("r", encoding="utf-8-sig", newline="") as handle:
        queue = list(csv.DictReader(handle))
    candidates = [
        row
        for row in queue
        if parse_bool(row["has_text_expected"])
        and float(row["mean_confidence"]) < 0.75
    ]
    candidates.sort(
        key=lambda row: (-float(row["risk_score"]), row["filename"])
    )
    candidates = candidates[: args.max_items]
    manifest = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest_by_filename = {
        row["source_file_name"]: row for row in manifest
    }

    os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "BOS")
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    from scripts.batch_ocr import create_ocr

    print(f"Loading one OCR model for {len(candidates)} pages ...")
    ocr = create_ocr(args.device)
    result_rows: list[dict[str, Any]] = []
    started_all = time.perf_counter()
    for item_index, queue_row in enumerate(candidates, start=1):
        item = manifest_by_filename[queue_row["filename"]]
        source_path = project_path(Path(item["source_path"]))
        page_rows: list[dict[str, Any]] = []
        with Image.open(source_path) as source:
            normalized = ImageOps.exif_transpose(source).convert("RGB")
            for angle in (0, 90, 180, 270):
                rotated = normalized.rotate(angle, expand=True)
                rotated_path = (
                    artifacts_dir
                    / f"{item['item_id']}_rotation_{angle}.jpg"
                )
                rotated.save(rotated_path, quality=95)
                started = time.perf_counter()
                results = list(ocr.predict(str(rotated_path)))
                elapsed = time.perf_counter() - started
                if len(results) != 1:
                    raise RuntimeError(
                        f"Expected one OCR result, got {len(results)}."
                    )
                json_path = rotated_path.with_suffix(".json")
                results[0].save_to_json(str(json_path))
                payload = json.loads(json_path.read_text(encoding="utf-8"))
                metrics = result_metrics(payload)
                page_rows.append(
                    {
                        "filename": queue_row["filename"],
                        "item_id": item["item_id"],
                        "category": item["category"],
                        "angle": angle,
                        **metrics,
                        "utility": rotation_utility(metrics),
                        "elapsed_seconds": elapsed,
                    }
                )
        baseline = page_rows[0]
        best = max(
            page_rows,
            key=lambda row: (float(row["utility"]), -int(row["angle"])),
        )
        accepted = should_accept_rotation(baseline, best)
        for row in page_rows:
            row["is_best"] = row["angle"] == best["angle"]
            row["recommended_rotation"] = (
                int(best["angle"]) if accepted else 0
            )
            row["rotation_accepted"] = accepted
            for key in (
                "mean_confidence",
                "low_confidence_fraction",
                "utility",
                "elapsed_seconds",
            ):
                row[key] = round(float(row[key]), 6)
            result_rows.append(row)
        print(
            f"[{item_index:02d}/{len(candidates):02d}] "
            f"{queue_row['filename']}: best={best['angle']}°, "
            f"accepted={accepted}, chars "
            f"{baseline['character_count']}->{best['character_count']}, "
            f"conf {baseline['mean_confidence']:.3f}"
            f"->{best['mean_confidence']:.3f}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result_rows[0]))
        writer.writeheader()
        writer.writerows(result_rows)
    accepted_pages = {
        row["item_id"]
        for row in result_rows
        if parse_bool(row["rotation_accepted"])
    }
    print(
        json.dumps(
            {
                "status": "success",
                "candidate_pages": len(candidates),
                "accepted_rotation_pages": len(accepted_pages),
                "ocr_inferences": len(result_rows),
                "elapsed_seconds": round(
                    time.perf_counter() - started_all, 3
                ),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
