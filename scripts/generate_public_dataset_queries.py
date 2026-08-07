"""Generate auditable retrieval-query candidates without calling an API."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NATURAL_DESCRIPTIONS = {
    "natural_no_text_001.jpg": "喀斯特群山、绿色稻田与乡间土路",
    "natural_no_text_002.jpg": "群山间蜿蜒的长城与山脚村落",
    "natural_no_text_003.jpg": "长城山脉下方的传统村落全景",
    "natural_no_text_004.jpg": "强烈阳光照射下的连绵山谷",
    "natural_no_text_005.jpg": "岩石山脚下的农田与村庄",
    "natural_no_text_006.jpg": "阳朔翠屏的喀斯特山水全景",
    "natural_no_text_007.jpg": "山地树林中的历史飞机坠落地点",
    "natural_no_text_008.jpg": "跨越大溪河桥梁的高速列车",
    "natural_no_text_009.jpg": "牵引客车跨越大溪河铁路桥的机车",
    "natural_no_text_010.jpg": "列车驶过水库上的高架桥",
    "natural_no_text_011.jpg": "积雪山峰与松林构成的黄山景观",
    "natural_no_text_012.jpg": "城市公园一角的步道与绿化",
    "natural_no_text_013.jpg": "从长城俯瞰八达岭山地景观",
    "natural_no_text_014.jpg": "黑白立体老照片中的中式临水楼阁",
    "natural_no_text_015.jpg": "货运列车穿行在山谷铁路线上",
    "natural_no_text_016.jpg": "滨江绿化带的城市夜景",
    "natural_no_text_017.jpg": "宝山滨江公园的树木与步道",
    "natural_no_text_018.jpg": "黑白老照片中的三水乡村与农田景观",
    "natural_no_text_019.jpg": "冬季田地中成排种植的豌豆",
    "natural_no_text_020.jpg": "群山峭壁与树林构成的彩色风景",
}
CATEGORY_PROMPTS = {
    "clear_document": "哪份清晰文档中出现",
    "complex_academic": "哪份复杂页面中出现",
    "degraded_document": "哪份扫描文档中出现",
    "table_form_ticket": "哪张表格、表单或票据中出现",
    "ppt_poster_slide": "哪张幻灯片或海报中出现",
    "software_web_code": "哪张界面或技术图片中出现",
    "scene_text": "哪张场景照片中出现",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/user_library/manifest.jsonl"),
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=Path("data/evaluation/public_dataset_200_splits.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/evaluation/public_dataset_200_query_candidates.csv"
        ),
    )
    parser.add_argument(
        "--review-output",
        type=Path,
        default=Path(
            "data/evaluation/public_dataset_200_query_review_priority.csv"
        ),
    )
    parser.add_argument(
        "--diagnostic-output",
        type=Path,
        default=Path(
            "data/evaluation/"
            "public_dataset_200_query_diagnostic_pending.csv"
        ),
    )
    parser.add_argument(
        "--all-nonempty-output",
        type=Path,
        default=Path(
            "data/evaluation/"
            "public_dataset_200_query_all_nonempty_pending.csv"
        ),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(
            "data/evaluation/public_dataset_200_query_summary.json"
        ),
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


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def normalize_line(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip(" \t\r\n,，。；;：:")


def valid_evidence_line(text: str, confidence: float) -> bool:
    if confidence < 0.65 or not 4 <= len(text) <= 36:
        return False
    if re.search(r"\d{6,}", text):
        return False
    sensitive_pattern = re.compile(
        r"地址|住址|电话|手机|邮箱|邮件|身份证|证件号码|"
        r"出生日期|姓名|学号|账号|银行卡|邮编|微信|"
        r"\baddress\b|\btelephone\b|\bphone\b|\be-?mail\b|"
        r"\bpassport\b|\bdate of birth\b",
        re.IGNORECASE,
    )
    if "@" in text or sensitive_pattern.search(text):
        return False
    location_or_url_pattern = re.compile(
        r"https?://|www\.|\.com\b|\.cn\b|"
        r"省.{0,12}市|市.{0,12}(?:区|县|镇|路|街|号)|"
        r"\d+\s+(?:avenue|street|road|boulevard)|"
        r"\bWashington,\s*D\.?C\.?\b|\bN\.?W\.?\b|"
        r"\b\d{5}(?:-\d{4})?\b",
        re.IGNORECASE,
    )
    if location_or_url_pattern.search(text):
        return False
    meaningful = sum(
        character.isalnum() or "\u4e00" <= character <= "\u9fff"
        for character in text
    )
    if meaningful / max(len(text), 1) < 0.65:
        return False
    digit_count = sum(character.isdigit() for character in text)
    if digit_count / max(meaningful, 1) > 0.5:
        return False
    contains_cjk = any("\u4e00" <= character <= "\u9fff" for character in text)
    if not contains_cjk and len(re.findall(r"[A-Za-z]+", text)) < 2:
        return False
    return len(set(text)) >= 3


def extract_lines(payload: dict[str, Any]) -> list[tuple[str, float]]:
    lines = []
    for text, score in zip(
        payload.get("rec_texts", []), payload.get("rec_scores", [])
    ):
        normalized = normalize_line(str(text))
        confidence = float(score)
        if valid_evidence_line(normalized, confidence):
            lines.append((normalized, confidence))
    return lines


def choose_evidence(
    lines: list[tuple[str, float]],
    document_frequency: Counter[str],
) -> list[str]:
    ranked = sorted(
        lines,
        key=lambda row: (
            -(
                row[1]
                * min(len(row[0]), 24)
                / max(document_frequency[row[0]], 1)
            ),
            row[0],
        ),
    )
    selected = []
    for text, _ in ranked:
        if any(text in existing or existing in text for existing in selected):
            continue
        selected.append(text)
        if len(selected) == 2:
            break
    return selected


def query_from_evidence(category: str, evidence: list[str]) -> str:
    prefix = CATEGORY_PROMPTS.get(category, "哪份资料中出现")
    quoted = "和".join(f"“{value}”" for value in evidence)
    return f"{prefix}{quoted}？"


def main() -> None:
    args = parse_args()
    manifest_path = project_path(args.manifest)
    split_path = project_path(args.splits)
    library_dir = manifest_path.parent
    output_path = project_path(args.output)
    review_path = project_path(args.review_output)
    diagnostic_path = project_path(args.diagnostic_output)
    all_nonempty_path = project_path(args.all_nonempty_output)
    summary_path = project_path(args.summary)

    manifest = [
        row
        for row in read_jsonl(manifest_path)
        if row.get("dataset_name") == "public_dataset_200_candidate"
    ]
    with split_path.open("r", encoding="utf-8-sig", newline="") as handle:
        split_rows = list(csv.DictReader(handle))
    split_by_id = {
        row["library_item_id"]: row["split"] for row in split_rows
    }
    group_members: dict[str, list[str]] = {}
    group_by_id: dict[str, str] = {}
    for row in split_rows:
        group = row["perceptual_group"]
        group_by_id[row["library_item_id"]] = group
        group_members.setdefault(group, []).append(row["library_item_id"])

    lines_by_id: dict[str, list[tuple[str, float]]] = {}
    document_frequency: Counter[str] = Counter()
    for item in manifest:
        item_id = item["item_id"]
        override_path = (
            library_dir / "ocr/overrides" / f"{item_id}.json"
        )
        baseline_path = library_dir / "ocr/json" / f"{item_id}.json"
        ocr_path = (
            override_path if override_path.is_file() else baseline_path
        )
        lines = extract_lines(
            json.loads(ocr_path.read_text(encoding="utf-8"))
        )
        lines_by_id[item_id] = lines
        document_frequency.update({text for text, _ in lines})

    query_rows: list[dict[str, Any]] = []
    for serial, item in enumerate(
        sorted(manifest, key=lambda row: row["source_file_name"]),
        start=1,
    ):
        item_id = item["item_id"]
        filename = item["source_file_name"]
        split = split_by_id[item_id]
        relevant_item_ids = sorted(
            group_members[group_by_id[item_id]]
        )
        flags = []
        evidence: list[str] = []
        query = ""
        generation_method = ""
        query_type = ""

        if item["privacy_review_required"]:
            flags.append("privacy_query_requires_manual")
            generation_method = "withheld_for_privacy"
            query_type = "manual_required"
        elif item["category"] == "natural_no_text":
            description = NATURAL_DESCRIPTIONS.get(filename, "")
            if description:
                query = f"查找一张关于“{description}”的自然图像。"
                evidence = [description]
                generation_method = "human_visual_description_seed"
                query_type = "visual_only"
                flags.append("visual_description_needs_confirmation")
            else:
                flags.append("visual_description_missing")
                generation_method = "manual_visual_description_required"
                query_type = "visual_only"
        else:
            evidence = choose_evidence(
                lines_by_id[item_id], document_frequency
            )
            if evidence:
                query = query_from_evidence(item["category"], evidence)
                generation_method = "distinctive_ocr_lines"
                query_type = (
                    "scene_text"
                    if item["category"] == "scene_text"
                    else "text_explicit"
                )
                maximum_df = max(document_frequency[value] for value in evidence)
                if maximum_df > 1:
                    flags.append("evidence_not_unique")
            else:
                flags.append("no_safe_ocr_evidence")
                generation_method = "manual_query_required"
                query_type = "manual_required"
        if item["category_review_required"]:
            flags.append("category_needs_confirmation")

        review_status = (
            "自动生成_训练候选"
            if split == "train"
            else "待人工审核"
        )
        if item["privacy_review_required"]:
            review_status = "隐私待处理"
        query_rows.append(
            {
                "query_id": f"p200_{serial:03d}",
                "query": query,
                "expected_item_id": item_id,
                "relevant_item_ids": ";".join(relevant_item_ids),
                "filename": filename,
                "split": split,
                "category": item["category"],
                "query_type": query_type,
                "generation_method": generation_method,
                "evidence": " | ".join(evidence),
                "evidence_document_frequency": (
                    max(
                        (document_frequency[value] for value in evidence),
                        default=0,
                    )
                ),
                "ambiguity_flags": ",".join(flags),
                "review_status": review_status,
                "usable_for_formal_metrics": False,
                "human_judgment": "",
                "human_query_revision": "",
                "human_notes": "",
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(query_rows[0]))
        writer.writeheader()
        writer.writerows(query_rows)

    priority_rows = [
        row
        for row in query_rows
        if row["split"] in {"validation", "test"}
        and (
            not row["query"]
            or bool(row["ambiguity_flags"])
            or int(row["evidence_document_frequency"]) > 1
        )
    ]
    priority_rows.sort(
        key=lambda row: (
            not (
                "privacy_query_requires_manual"
                in row["ambiguity_flags"]
                or "no_safe_ocr_evidence" in row["ambiguity_flags"]
                or "visual_description_missing" in row["ambiguity_flags"]
            ),
            row["split"],
            row["filename"],
        )
    )
    with review_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(query_rows[0]))
        writer.writeheader()
        writer.writerows(priority_rows)

    diagnostic_rows = [
        row
        for row in query_rows
        if row["split"] in {"validation", "test"} and bool(row["query"])
    ]
    with diagnostic_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(query_rows[0]))
        writer.writeheader()
        writer.writerows(diagnostic_rows)

    all_nonempty_rows = [row for row in query_rows if bool(row["query"])]
    with all_nonempty_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(query_rows[0]))
        writer.writeheader()
        writer.writerows(all_nonempty_rows)

    summary = {
        "status": "candidate_queries_only",
        "total_queries": len(query_rows),
        "nonempty_queries": sum(bool(row["query"]) for row in query_rows),
        "split_counts": dict(Counter(row["split"] for row in query_rows)),
        "query_type_counts": dict(
            Counter(row["query_type"] for row in query_rows)
        ),
        "validation_test_queries": sum(
            row["split"] in {"validation", "test"} for row in query_rows
        ),
        "diagnostic_nonempty_queries": len(diagnostic_rows),
        "all_nonempty_queries": len(all_nonempty_rows),
        "priority_review_queries": len(priority_rows),
        "formal_metric_queries": 0,
        "warning": (
            "All validation/test candidates require human confirmation "
            "before formal evaluation."
        ),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
