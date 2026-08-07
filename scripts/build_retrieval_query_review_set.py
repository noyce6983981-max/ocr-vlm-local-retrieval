"""Build a balanced 100-query human-review set for the 1500-page library."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_public_dataset_queries import (  # noqa: E402
    choose_evidence,
    extract_lines,
)
from scripts.quality_review import read_quality_reviews  # noqa: E402
from scripts.taxonomy import CATEGORY_LABELS, normalize_category  # noqa: E402


CATEGORY_SPLIT_QUOTAS = {
    "general_text_document": {"train": 9, "validation": 3, "test": 3},
    "complex_academic": {"train": 8, "validation": 3, "test": 3},
    "table_form_ticket": {"train": 10, "validation": 4, "test": 4},
    "ppt_poster_slide": {"train": 8, "validation": 2, "test": 2},
    "software_web_code": {"train": 7, "validation": 2, "test": 2},
    "scene_text": {"train": 6, "validation": 2, "test": 2},
    "natural_no_text": {"train": 6, "validation": 2, "test": 2},
}
NO_ANSWER_QUERIES = {
    "train": [
        "查找紫色热气球飞越雪山峡谷的照片。",
        "哪张图片展示宇航员在月球表面维修机器人？",
        "查找夜晚绿色极光下的木屋和结冰湖面。",
        "哪份文档介绍量子计算机的低温制冷维护流程？",
        "查找咖啡拉花步骤和心形图案的教学海报。",
        "哪张图片展示博物馆中的完整恐龙骨架和参观人群？",
    ],
    "validation": [
        "查找红色双层公交车穿过雨夜霓虹街道的照片。",
        "哪份表格记录火星探测器每日能源消耗和通信时延？",
    ],
    "test": [
        "查找钢琴旁摆放手写乐谱和小提琴的室内照片。",
        "哪张技术图片展示显微镜下进行电路板焊接？",
    ],
}
TEXT_PREFIXES = {
    "general_text_document": "哪份普通文档中提到了",
    "complex_academic": "哪份学术或技术文档中提到了",
    "table_form_ticket": "哪张表格、表单或票据中包含",
    "ppt_poster_slide": "哪张幻灯片或海报中出现",
    "software_web_code": "哪张软件、网页或代码截图中出现",
    "scene_text": "哪张场景照片中出现",
}
QUERY_TYPES = {
    "general_text_document": "text_explicit",
    "complex_academic": "academic_semantic",
    "table_form_ticket": "table_field",
    "ppt_poster_slide": "slide_semantic",
    "software_web_code": "interface_semantic",
    "scene_text": "scene_text",
    "natural_no_text": "visual_only",
}
QUEUE_FIELDS = [
    "query_id",
    "query",
    "expected_item_id",
    "relevant_item_ids",
    "candidate_item_ids",
    "system_acceptance",
    "system_acceptance_reason",
    "system_top1_score",
    "filename",
    "split",
    "category",
    "query_type",
    "generation_method",
    "evidence",
    "evidence_document_frequency",
    "ambiguity_flags",
    "review_status",
    "usable_for_formal_metrics",
    "human_judgment",
    "human_query_revision",
    "human_notes",
    "diagnostic_priority_score",
    "diagnostic_review_reasons",
    "relevant_group_size",
    "source_query_ids_for_group",
]
PRIVATE_OR_LOCATION_PATTERN = re.compile(
    r"姓名|姓\s*名|住址|地址|身份证|公民身份|手机号|电话|邮箱|"
    r"出生|户籍|邮编|联系人|家长|本人|个人信息|"
    r"\bname\b|\baddress\b|\bphone\b|\btelephone\b|\be-?mail\b|"
    r"\bfax\b|\bdate of birth\b|"
    r"[\u4e00-\u9fff]{2,10}(?:省|市|区|县|镇|乡|村|路|街|巷|号)",
    re.IGNORECASE,
)


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
        default=Path(
            "data/evaluation/public_dataset_1500_category_review_splits.csv"
        ),
    )
    parser.add_argument(
        "--reviews",
        type=Path,
        default=Path(
            "data/evaluation/public_dataset_1500_quality_human_reviews.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/evaluation/"
            "public_dataset_1500_retrieval_query_queue_100.csv"
        ),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "data/evaluation/"
            "public_dataset_1500_retrieval_query_protocol.json"
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def effective_category(item: dict[str, Any], review: dict[str, str] | None) -> str:
    if (
        review
        and review.get("decision") == "reclassified"
        and review.get("revised_category") in CATEGORY_LABELS
    ):
        return review["revised_category"]
    return normalize_category(str(item.get("category", "")))


def ocr_payload(library_dir: Path, item_id: str) -> dict[str, Any]:
    for path in (
        library_dir / "ocr/overrides" / f"{item_id}.json",
        library_dir / "ocr/json" / f"{item_id}.json",
    ):
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    return {}


def source_description(item: dict[str, Any]) -> str:
    value = str(item.get("public_source_file", "")).strip()
    value = re.sub(r"^File:", "", value, flags=re.IGNORECASE)
    value = Path(value).stem
    value = re.sub(r"\([^)]*\)", " ", value)
    value = re.sub(r"[_-]+", " ", value)
    value = re.sub(r"(?:IMG|DSC|image|photo)\s*\d+", " ", value, flags=re.I)
    value = re.sub(r"\b\d{4,}\b", " ", value)
    value = re.sub(r"^\d+\s+", "", value)
    value = " ".join(value.split()).strip(" .-_")
    words = re.findall(r"[A-Za-z\u4e00-\u9fff]+", value)
    if len(words) < 3 or len(value) < 12:
        return ""
    return value[:90]


def safe_retrieval_lines(
    lines: list[tuple[str, float]],
) -> list[tuple[str, float]]:
    return [
        (text, confidence)
        for text, confidence in lines
        if not PRIVATE_OR_LOCATION_PATTERN.search(text)
        and not re.search(r"\d{6,}", text)
    ]


def build_query(
    item: dict[str, Any],
    category: str,
    evidence: list[str],
) -> tuple[str, str]:
    if category == "natural_no_text":
        description = source_description(item)
        if not description:
            return "", ""
        return f"查找一张展示“{description}”的自然图像。", description
    if category == "scene_text" and not evidence:
        description = source_description(item)
        if not description:
            return "", ""
        return f"哪张场景照片与“{description}”相符？", description
    if category == "software_web_code" and not evidence:
        description = source_description(item)
        if not description:
            return "", ""
        return f"哪张软件或网页截图与“{description}”相关？", description
    if not evidence:
        return "", ""
    quoted = "和".join(f"“{value}”" for value in evidence)
    return f"{TEXT_PREFIXES[category]}{quoted}？", " | ".join(evidence)


def main() -> None:
    args = parse_args()
    manifest_path = project_path(args.manifest)
    split_path = project_path(args.splits)
    reviews_path = project_path(args.reviews)
    output_path = project_path(args.output)
    protocol_path = project_path(args.protocol)
    library_dir = manifest_path.parent

    manifest = [
        item
        for item in read_jsonl(manifest_path)
        if bool(item.get("search_enabled", True))
    ]
    manifest_by_id = {item["item_id"]: item for item in manifest}
    split_rows = read_csv(split_path)
    group_by_id = {row["item_id"]: row["group_id"] for row in split_rows}
    group_members: dict[str, list[str]] = defaultdict(list)
    for row in split_rows:
        if row["item_id"] in manifest_by_id:
            group_members[row["group_id"]].append(row["item_id"])
    reviews = read_quality_reviews(reviews_path)

    lines_by_id: dict[str, list[tuple[str, float]]] = {}
    document_frequency: Counter[str] = Counter()
    for item in manifest:
        lines = safe_retrieval_lines(
            extract_lines(ocr_payload(library_dir, item["item_id"]))
        )
        lines_by_id[item["item_id"]] = lines
        document_frequency.update({text for text, _ in lines})

    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in manifest:
        item_id = item["item_id"]
        if item_id not in group_by_id:
            continue
        if bool(item.get("privacy_review_required", False)):
            continue
        category = effective_category(item, reviews.get(item_id))
        if category not in CATEGORY_SPLIT_QUOTAS:
            continue
        evidence = choose_evidence(lines_by_id[item_id], document_frequency)
        query, evidence_text = build_query(item, category, evidence)
        if not query:
            continue
        max_df = max((document_frequency[value] for value in evidence), default=1)
        mean_confidence = (
            sum(score for _, score in lines_by_id[item_id])
            / max(len(lines_by_id[item_id]), 1)
        )
        description_bonus = 1.0 if category in {"natural_no_text", "scene_text"} else 0.0
        quality_score = (
            4.0 / max(max_df, 1)
            + 1.5 * min(len(evidence), 2)
            + mean_confidence
            + description_bonus
        )
        candidates[category].append(
            {
                "item": item,
                "category": category,
                "query": query,
                "evidence": evidence_text,
                "max_df": max_df,
                "quality_score": quality_score,
            }
        )

    selected: list[dict[str, Any]] = []
    used_groups: set[str] = set()
    for category, split_quotas in CATEGORY_SPLIT_QUOTAS.items():
        quota = sum(split_quotas.values())
        bucket = sorted(
            candidates[category],
            key=lambda row: (-row["quality_score"], row["item"]["item_id"]),
        )
        chosen = []
        for candidate in bucket:
            group_id = group_by_id[candidate["item"]["item_id"]]
            if group_id in used_groups:
                continue
            chosen.append(candidate)
            used_groups.add(group_id)
            if len(chosen) == quota:
                break
        if len(chosen) != quota:
            raise ValueError(
                f"insufficient {category} candidates: {len(chosen)}/{quota}"
            )
        chosen.sort(
            key=lambda row: hashlib.sha256(
                row["item"]["item_id"].encode("utf-8")
            ).hexdigest()
        )
        offset = 0
        for split in ("train", "validation", "test"):
            split_count = split_quotas[split]
            for candidate in chosen[offset : offset + split_count]:
                selected.append({**candidate, "split": split})
            offset += split_count

    rows: list[dict[str, Any]] = []
    sequence = 0
    for split in ("train", "validation", "test"):
        for candidate in [row for row in selected if row["split"] == split]:
            sequence += 1
            item = candidate["item"]
            group_id = group_by_id[item["item_id"]]
            relevant_ids = sorted(group_members[group_id])
            rows.append(
                {
                    "query_id": f"rq100_{sequence:03d}",
                    "query": candidate["query"],
                    "expected_item_id": item["item_id"],
                    "relevant_item_ids": ";".join(relevant_ids),
                    "candidate_item_ids": "",
                    "system_acceptance": "",
                    "system_acceptance_reason": "",
                    "system_top1_score": "",
                    "filename": item.get("source_file_name", ""),
                    "split": split,
                    "category": candidate["category"],
                    "query_type": QUERY_TYPES[candidate["category"]],
                    "generation_method": (
                        "public_source_visual_description"
                        if candidate["category"] == "natural_no_text"
                        or (
                            candidate["category"] == "scene_text"
                            and not candidate["evidence"]
                        )
                        else "distinctive_safe_ocr_evidence"
                    ),
                    "evidence": candidate["evidence"],
                    "evidence_document_frequency": candidate["max_df"],
                    "ambiguity_flags": "human_confirmation_required",
                    "review_status": "待人工审核",
                    "usable_for_formal_metrics": "False",
                    "human_judgment": "",
                    "human_query_revision": "",
                    "human_notes": "",
                    "diagnostic_priority_score": round(
                        candidate["quality_score"], 4
                    ),
                    "diagnostic_review_reasons": (
                        "visual_description,auto_generated"
                        if candidate["category"] == "natural_no_text"
                        else "auto_generated"
                    ),
                    "relevant_group_size": len(relevant_ids),
                    "source_query_ids_for_group": f"rq100_{sequence:03d}",
                }
            )
        for query in NO_ANSWER_QUERIES[split]:
            sequence += 1
            rows.append(
                {
                    "query_id": f"rq100_{sequence:03d}",
                    "query": query,
                    "expected_item_id": "",
                    "relevant_item_ids": "",
                    "candidate_item_ids": "",
                    "system_acceptance": "",
                    "system_acceptance_reason": "",
                    "system_top1_score": "",
                    "filename": "",
                    "split": split,
                    "category": "open_set_no_answer",
                    "query_type": "no_answer",
                    "generation_method": "human_designed_absent_combination",
                    "evidence": "",
                    "evidence_document_frequency": 0,
                    "ambiguity_flags": "open_set_requires_confirmation",
                    "review_status": "待人工审核",
                    "usable_for_formal_metrics": "False",
                    "human_judgment": "",
                    "human_query_revision": "",
                    "human_notes": "",
                    "diagnostic_priority_score": 10.0,
                    "diagnostic_review_reasons": "no_answer",
                    "relevant_group_size": 0,
                    "source_query_ids_for_group": f"rq100_{sequence:03d}",
                }
            )

    if len(rows) != 100 or sequence != 100:
        raise AssertionError(f"expected 100 rows, got {len(rows)}")
    if len({row["query"] for row in rows}) != 100:
        raise ValueError("generated queries are not unique")
    if len({row["query_id"] for row in rows}) != 100:
        raise ValueError("query IDs are not unique")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=QUEUE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    fingerprint = hashlib.sha256(output_path.read_bytes()).hexdigest()
    protocol = {
        "status": "human_review_required",
        "query_count": 100,
        "answerable_count": 90,
        "no_answer_count": 10,
        "split_counts": dict(Counter(row["split"] for row in rows)),
        "category_counts": dict(Counter(row["category"] for row in rows)),
        "query_type_counts": dict(Counter(row["query_type"] for row in rows)),
        "target_group_count": len(
            {
                group_by_id[row["expected_item_id"]]
                for row in rows
                if row["expected_item_id"]
            }
        ),
        "group_overlap_between_targets": False,
        "privacy_targets_included": False,
        "queue_sha256": fingerprint,
        "selection_policy": (
            "independent balanced query split; one target per isolated group; "
            "safe OCR evidence or public source visual description"
        ),
        "metric_policy": (
            "train/development may tune retrieval; validation selects; "
            "test is reported once after method freeze"
        ),
    }
    protocol_path.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
