"""Extract OCR quality features and create a compact human-review queue."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--library-dir",
        type=Path,
        default=Path("outputs/user_library"),
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=Path("data/evaluation/public_dataset_200_splits.csv"),
    )
    parser.add_argument(
        "--quality-output",
        type=Path,
        default=Path(
            "data/evaluation/public_dataset_200_ocr_quality.csv"
        ),
    )
    parser.add_argument(
        "--review-output",
        type=Path,
        default=Path(
            "data/evaluation/public_dataset_200_review_queue_30.csv"
        ),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(
            "data/evaluation/public_dataset_200_quality_summary.json"
        ),
    )
    parser.add_argument("--review-count", type=int, default=30)
    return parser.parse_args()


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def quality_features(
    scores: list[float],
    texts: list[str],
    has_text_expected: bool,
) -> dict[str, Any]:
    clean_scores = [float(value) for value in scores]
    character_count = sum(len(str(value).strip()) for value in texts)
    box_count = len(texts)
    mean_confidence = (
        statistics.fmean(clean_scores) if clean_scores else 0.0
    )
    low_confidence_fraction = (
        sum(value < 0.6 for value in clean_scores) / len(clean_scores)
        if clean_scores
        else 1.0
    )
    zero_confidence_fraction = (
        sum(value <= 0.01 for value in clean_scores) / len(clean_scores)
        if clean_scores
        else 1.0
    )
    sparse_risk = max(0.0, (50 - character_count) / 50)
    if has_text_expected:
        risk_score = (
            0.55 * (1.0 - mean_confidence)
            + 0.25 * low_confidence_fraction
            + 0.20 * sparse_risk
        )
    else:
        false_positive_strength = (
            min(character_count / 30, 1.0) * mean_confidence
        )
        risk_score = 0.5 + 0.5 * false_positive_strength
    proposed_ocr_useful = (
        has_text_expected
        and character_count >= 8
        and mean_confidence >= 0.60
    )
    return {
        "text_box_count": box_count,
        "character_count": character_count,
        "mean_confidence": round(mean_confidence, 6),
        "p10_confidence": round(percentile(clean_scores, 0.10), 6),
        "low_confidence_fraction": round(
            low_confidence_fraction, 6
        ),
        "zero_confidence_fraction": round(
            zero_confidence_fraction, 6
        ),
        "chars_per_box": round(
            character_count / box_count if box_count else 0.0, 6
        ),
        "risk_score": round(risk_score, 6),
        "proposed_ocr_useful": proposed_ocr_useful,
    }


def review_reasons(row: dict[str, Any]) -> str:
    reasons = []
    if row["privacy_review_required"]:
        reasons.append("privacy")
    if row["category_review_required"]:
        reasons.append("category")
    if not row["has_text_expected"] and row["character_count"] >= 5:
        reasons.append("no_text_false_positive")
    if row["has_text_expected"] and row["mean_confidence"] < 0.75:
        reasons.append("low_confidence")
    if row["has_text_expected"] and row["character_count"] < 20:
        reasons.append("sparse_ocr")
    return ",".join(reasons) or "high_uncertainty"


def select_review_queue(
    rows: list[dict[str, Any]],
    count: int,
) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    selected: dict[str, dict[str, Any]] = {}
    ranked = sorted(
        rows,
        key=lambda row: (
            not row["privacy_review_required"],
            -float(row["risk_score"]),
            row["filename"],
        ),
    )
    for row in ranked:
        if row["privacy_review_required"]:
            selected[row["filename"]] = row

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)
    for category_rows in by_category.values():
        category_rows.sort(
            key=lambda row: (-float(row["risk_score"]), row["filename"])
        )
        for row in category_rows[:2]:
            selected.setdefault(row["filename"], row)

    for row in sorted(
        rows,
        key=lambda value: (
            -float(value["risk_score"]),
            value["filename"],
        ),
    ):
        selected.setdefault(row["filename"], row)
        if len(selected) >= count:
            break
    return sorted(
        list(selected.values())[:count],
        key=lambda row: (-float(row["risk_score"]), row["filename"]),
    )


def write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp_path, path)


def main() -> None:
    args = parse_args()
    library_dir = project_path(args.library_dir)
    split_path = project_path(args.splits)
    quality_path = project_path(args.quality_output)
    review_path = project_path(args.review_output)
    summary_path = project_path(args.summary)

    manifest = read_jsonl(library_dir / "manifest.jsonl")
    manifest_by_id = {
        row["item_id"]: row
        for row in manifest
        if row.get("dataset_name") == "public_dataset_200_candidate"
    }
    with split_path.open("r", encoding="utf-8-sig", newline="") as handle:
        split_by_id = {
            row["library_item_id"]: row["split"]
            for row in csv.DictReader(handle)
        }

    quality_rows: list[dict[str, Any]] = []
    for item_id, item in manifest_by_id.items():
        ocr_path = library_dir / "ocr/json" / f"{item_id}.json"
        payload = json.loads(ocr_path.read_text(encoding="utf-8"))
        features = quality_features(
            list(payload.get("rec_scores", [])),
            [str(value) for value in payload.get("rec_texts", [])],
            bool(item["has_text_expected"]),
        )
        preview = " ".join(
            str(value).strip()
            for value in payload.get("rec_texts", [])
            if str(value).strip()
        )[:160]
        row = {
            "filename": item["source_file_name"],
            "item_id": item_id,
            "category": item["category"],
            "split": split_by_id[item_id],
            "has_text_expected": item["has_text_expected"],
            **features,
            "privacy_review_required": item[
                "privacy_review_required"
            ],
            "category_review_required": item[
                "category_review_required"
            ],
            "ocr_text_preview": preview,
        }
        row["review_reason"] = review_reasons(row)
        row["human_ocr_useful"] = ""
        row["human_error_type"] = ""
        row["human_notes"] = ""
        quality_rows.append(row)
    quality_rows.sort(key=lambda row: row["filename"])
    review_rows = select_review_queue(
        quality_rows, min(args.review_count, len(quality_rows))
    )

    quality_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv_atomic(quality_path, quality_rows)
    write_csv_atomic(review_path, review_rows)

    category_summary = {}
    for category in sorted({row["category"] for row in quality_rows}):
        category_rows = [
            row for row in quality_rows if row["category"] == category
        ]
        category_summary[category] = {
            "pages": len(category_rows),
            "mean_ocr_confidence": round(
                statistics.fmean(
                    float(row["mean_confidence"])
                    for row in category_rows
                ),
                6,
            ),
            "mean_character_count": round(
                statistics.fmean(
                    int(row["character_count"]) for row in category_rows
                ),
                3,
            ),
            "proposed_ocr_useful_pages": sum(
                parse_bool(row["proposed_ocr_useful"])
                for row in category_rows
            ),
        }
    summary = {
        "status": "success",
        "total_pages": len(quality_rows),
        "review_queue_pages": len(review_rows),
        "review_queue_categories": dict(
            Counter(row["category"] for row in review_rows)
        ),
        "proposed_ocr_useful_pages": sum(
            parse_bool(row["proposed_ocr_useful"])
            for row in quality_rows
        ),
        "zero_ocr_pages": sum(
            int(row["text_box_count"]) == 0 for row in quality_rows
        ),
        "category_summary": category_summary,
        "label_warning": (
            "proposed_ocr_useful is an automatic proposal, not human truth"
        ),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
