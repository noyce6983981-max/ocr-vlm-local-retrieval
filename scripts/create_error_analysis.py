"""Create a human-reviewable error table from retrieval evaluation results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create retrieval error analysis CSV.")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("outputs/evaluation/text_retrieval_baseline.json"),
    )
    parser.add_argument(
        "--ocr-summary",
        type=Path,
        default=Path("outputs/ocr_batch/summary.csv"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifest/pilot_manifest.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/evaluation/error_analysis.csv"),
    )
    return parser.parse_args()


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def infer_cause(result: dict[str, Any], mean_confidence: float) -> tuple[str, str]:
    if not result["expected_item_indexed"]:
        return (
            "visual_semantics_missing",
            "目标图片没有OCR文字，因此未进入文本索引。",
        )
    if mean_confidence < 0.7:
        return (
            "ocr_noise",
            f"目标图片平均OCR置信度仅{mean_confidence:.4f}，识别文本存在乱码。",
        )
    return (
        "semantic_ranking_ambiguity",
        "目标已进入文本索引，但另一图片含有相近主题词，排序更靠前。",
    )


def main() -> None:
    args = parse_args()
    report_path = project_path(args.report)
    ocr_summary_path = project_path(args.ocr_summary)
    manifest_path = project_path(args.manifest)
    output_path = project_path(args.output)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest = {
        row["item_id"]: row
        for row in (
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    with ocr_summary_path.open("r", encoding="utf-8-sig", newline="") as handle:
        confidence_by_item = {
            row["item_id"]: float(row["mean_confidence"])
            for row in csv.DictReader(handle)
        }

    rows: list[dict[str, Any]] = []
    for result in report["results"]:
        if result["expected_rank"] == 1:
            continue
        expected_item = result["expected_item_id"]
        confidence = confidence_by_item.get(expected_item, 0.0)
        cause, evidence = infer_cause(result, confidence)
        top_1 = result["top_3"][0] if result["top_3"] else {}
        item = manifest[expected_item]
        source_path = Path(item["source_path"])
        rows.append(
            {
                "query_id": result["query_id"],
                "query": result["query"],
                "expected_display_name": item["display_name_zh"],
                "expected_item_id": expected_item,
                "original_filename": source_path.name,
                "visualization_path": (
                    "outputs/ocr_batch/visualizations/"
                    f"{expected_item}{source_path.suffix.lower()}"
                ),
                "expected_rank": result["expected_rank"] or "未召回",
                "top_1_display_name": top_1.get("display_name_zh", ""),
                "top_1_item_id": top_1.get("item_id", ""),
                "expected_mean_ocr_confidence": round(confidence, 6),
                "proposed_cause": cause,
                "evidence": evidence,
                "human_judgement": "待确认",
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    print(f"Error cases: {len(rows)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
