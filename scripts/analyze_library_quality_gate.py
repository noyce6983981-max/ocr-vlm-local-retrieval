"""Route OCR pages to pass, retry, category review, or privacy review queues."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
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
        "--output-csv",
        type=Path,
        default=Path(
            "data/evaluation/public_dataset_1500_quality_gate.csv"
        ),
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path(
            "data/evaluation/public_dataset_1500_quality_gate_summary.json"
        ),
    )
    return parser.parse_args()


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def classify_quality(
    *,
    has_text_expected: bool,
    character_count: int,
    text_box_count: int,
    mean_confidence: float,
    privacy_review_required: bool,
    sensitive_text_detected: bool = True,
) -> tuple[str, str]:
    if privacy_review_required and sensitive_text_detected:
        return "privacy_review", "人工检查人脸、身份或敏感字段"
    if not has_text_expected:
        if character_count >= 6 and mean_confidence >= 0.70:
            return "category_text_leak", "复核无文字负样本并考虑重分类"
        if character_count >= 3 and mean_confidence >= 0.50:
            return "category_review", "检查是否存在少量场景文字或商标"
        return "pass", "保留为视觉负样本"
    if character_count == 0 or text_box_count == 0:
        return "ocr_retry", "触发旋转、增强或视觉模型兜底"
    if mean_confidence < 0.60:
        return "ocr_low_confidence", "进入增强OCR和视觉重排队列"
    return "pass", "无需额外处理"


def contains_sensitive_text(text: str) -> bool:
    normalized = " ".join(text.split())
    keyword_pattern = re.compile(
        r"身份证|公民身份|住址|出生|姓名|手机号|电话|邮箱|"
        r"\baddress\b|\bphone\b|\be-?mail\b|\bdate of birth\b|"
        r"\bssn\b|\bsocial security\b",
        re.IGNORECASE,
    )
    email_pattern = re.compile(
        r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        re.IGNORECASE,
    )
    long_number_pattern = re.compile(r"(?<!\d)\d[\d -]{7,}\d(?!\d)")
    return bool(
        keyword_pattern.search(normalized)
        or email_pattern.search(normalized)
        or long_number_pattern.search(normalized)
    )


def suggest_semantic_category(text: str) -> str:
    """Suggest a coarse document category from high-precision OCR cues."""
    normalized = " ".join(text.split())
    identity_pattern = re.compile(
        r"身份证|公民身份号码|居民身份证|"
        r"\bidentity card\b|\bid card\b",
        re.IGNORECASE,
    )
    if identity_pattern.search(normalized):
        return "table_form_ticket"
    return ""


def apply_category_suggestion(
    *,
    route: str,
    action: str,
    current_category: str,
    suggested_category: str,
) -> tuple[str, str]:
    """Route high-confidence semantic category mismatches to review."""
    if (
        route == "pass"
        and suggested_category
        and suggested_category != current_category
    ):
        return (
            "category_review",
            "OCR内容与当前类别不一致，建议人工确认分类",
        )
    return route, action


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    args = parse_args()
    library_dir = project_path(args.library_dir)
    manifest = read_jsonl(library_dir / "manifest.jsonl")
    by_id = {row["item_id"]: row for row in manifest}
    with (library_dir / "ocr/summary.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        ocr_rows = list(csv.DictReader(handle))

    output_rows: list[dict[str, Any]] = []
    for ocr in ocr_rows:
        item = by_id.get(ocr["item_id"])
        if item is None or not item.get("public_source_name"):
            continue
        ocr_json_path = library_dir / "ocr/json" / f"{ocr['item_id']}.json"
        ocr_text = ""
        if ocr_json_path.is_file():
            payload = json.loads(ocr_json_path.read_text(encoding="utf-8"))
            ocr_text = " ".join(
                str(value) for value in payload.get("rec_texts", [])
            )
        sensitive_text_detected = contains_sensitive_text(ocr_text)
        suggested_category = suggest_semantic_category(ocr_text)
        route, action = classify_quality(
            has_text_expected=parse_bool(
                item.get("has_text_expected", True)
            ),
            character_count=int(ocr["character_count"]),
            text_box_count=int(ocr["text_box_count"]),
            mean_confidence=float(ocr["mean_confidence"]),
            privacy_review_required=parse_bool(
                item.get("privacy_review_required", False)
            ),
            sensitive_text_detected=sensitive_text_detected,
        )
        route, action = apply_category_suggestion(
            route=route,
            action=action,
            current_category=item.get("category", ""),
            suggested_category=suggested_category,
        )
        output_rows.append(
            {
                "item_id": ocr["item_id"],
                "filename": item.get("source_file_name", ""),
                "category": item.get("category", ""),
                "suggested_category": suggested_category,
                "source_name": item.get("public_source_name", ""),
                "has_text_expected": str(
                    parse_bool(item.get("has_text_expected", True))
                ).lower(),
                "text_box_count": ocr["text_box_count"],
                "character_count": ocr["character_count"],
                "mean_confidence": ocr["mean_confidence"],
                "sensitive_text_detected": str(
                    sensitive_text_detected
                ).lower(),
                "quality_route": route,
                "recommended_action": action,
                "human_judgement": "",
                "human_notes": "",
            }
        )

    output_csv = project_path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    routes = Counter(row["quality_route"] for row in output_rows)
    category_review_counts = Counter(
        row["category"]
        for row in output_rows
        if row["quality_route"]
        in {"category_text_leak", "category_review"}
    )
    summary = {
        "status": "success",
        "page_count": len(output_rows),
        "route_counts": dict(routes),
        "category_review_counts": dict(category_review_counts),
        "automatic_decisions_are_candidates_not_ground_truth": True,
        "privacy_source_flags": sum(
            parse_bool(row.get("privacy_review_required", False))
            for row in manifest
            if row.get("public_source_name")
        ),
        "sensitive_text_candidates": sum(
            row["sensitive_text_detected"] == "true"
            for row in output_rows
        ),
        "human_review_required": sum(
            route != "pass" for route in routes.elements()
        ),
    }
    output_summary = project_path(args.output_summary)
    output_summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
