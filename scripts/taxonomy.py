"""Shared category and visual-quality taxonomy for document pages."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Iterable


CATEGORY_LABELS = OrderedDict(
    [
        ("general_text_document", "普通文本型文档"),
        ("complex_academic", "学术/技术文档"),
        ("table_form_ticket", "证件/表格/表单/票据"),
        ("ppt_poster_slide", "PPT/海报/课件"),
        ("software_web_code", "软件/网页/代码"),
        ("scene_text", "场景文字"),
        ("natural_no_text", "自然图像"),
    ]
)

CATEGORY_DESCRIPTIONS = {
    "general_text_document": "通知、说明、报告、纪要、简历、书籍正文等普通连续文本页面",
    "complex_academic": "论文、教材、公式、实验图表、技术报告等学术或技术页面",
    "table_form_ticket": "证件、成绩单、申请表、统计表、票据、参数表等结构化页面",
    "ppt_poster_slide": "演示文稿、课程幻灯片、宣传海报和信息图",
    "software_web_code": "软件界面、网页、代码、终端、仪表盘和聊天界面",
    "scene_text": "路牌、店招、横幅、公告栏和屏幕实拍等自然场景文字",
    "natural_no_text": "以人物、建筑、车辆、动物或自然景观为主的图像",
}

QUALITY_LABELS = OrderedDict(
    [
        ("clear", "无明显质量问题"),
        ("blur", "模糊"),
        ("tilt", "倾斜/透视"),
        ("shadow", "阴影"),
        ("reflection", "反光"),
        ("low_resolution", "低分辨率"),
        ("occlusion", "遮挡"),
        ("low_contrast", "低对比度"),
        ("scan_noise", "扫描噪声/折痕"),
        ("compression_artifact", "压缩失真"),
        ("unspecified_degradation", "其他退化"),
    ]
)

LEGACY_CATEGORY_MAP = {
    "clear_document": "general_text_document",
    "degraded_document": "general_text_document",
    "document_page": "general_text_document",
    **{category: category for category in CATEGORY_LABELS},
}

AMBIGUOUS_LEGACY_CATEGORIES = {
    "clear_document",
    "degraded_document",
    "document_page",
}

TABLE_FORM_SOURCE_TAGS = {
    "budget",
    "form",
    "form_layout",
    "invoice",
    "real_form",
    "real_receipt",
    "receipt",
    "table",
    "table_layout",
    "ticket",
}

QUALITY_TAG_ALIASES = {
    "clear": "clear",
    "clear_scan": "clear",
    "blur": "blur",
    "blurry": "blur",
    "tilt": "tilt",
    "perspective": "tilt",
    "shadow": "shadow",
    "reflection": "reflection",
    "glare": "reflection",
    "low_resolution": "low_resolution",
    "occlusion": "occlusion",
    "low_contrast": "low_contrast",
    "scan_noise": "scan_noise",
    "compression_artifact": "compression_artifact",
    "jpeg_artifact": "compression_artifact",
    "unspecified_degradation": "unspecified_degradation",
}


def parse_quality_tags(value: str | Iterable[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_tags = value.split(";")
    else:
        raw_tags = list(value)
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_tag in raw_tags:
        tag = str(raw_tag).strip()
        if not tag:
            continue
        mapped = QUALITY_TAG_ALIASES.get(tag, tag)
        if mapped in QUALITY_LABELS and mapped not in seen:
            seen.add(mapped)
            normalized.append(mapped)
    if "clear" in seen and len(seen) > 1:
        normalized = [tag for tag in normalized if tag != "clear"]
    return normalized


def serialize_quality_tags(value: str | Iterable[str] | None) -> str:
    return ";".join(parse_quality_tags(value))


def normalize_category(category: str) -> str:
    return LEGACY_CATEGORY_MAP.get(category, "general_text_document")


def source_tag_category_suggestion(
    row: dict[str, Any],
) -> str | None:
    """Return only high-precision category suggestions from provenance tags."""
    source_tags = {
        str(tag).strip().lower()
        for tag in row.get("source_tags", [])
        if str(tag).strip()
    }
    if source_tags & TABLE_FORM_SOURCE_TAGS:
        return "table_form_ticket"
    provenance = " ".join(
        str(row.get(field, "")).strip().lower()
        for field in ("source", "public_source_name")
    )
    if "funsd" in provenance or "xfund" in provenance:
        return "table_form_ticket"
    return None


def migrate_manifest_row(
    source_row: dict[str, Any],
    review: dict[str, str] | None = None,
) -> dict[str, Any]:
    row = dict(source_row)
    legacy_category = str(row.get("category", ""))
    migrated_category = normalize_category(legacy_category)
    if review and review.get("decision") == "reclassified":
        migrated_category = normalize_category(
            review.get("revised_category", "")
        )

    legacy_tags = list(row.get("quality_tags") or [])
    quality_tags = parse_quality_tags(legacy_tags)
    if legacy_category == "degraded_document" and not quality_tags:
        quality_tags = ["unspecified_degradation"]
    source_tags = [
        str(tag)
        for tag in legacy_tags
        if str(tag) not in QUALITY_TAG_ALIASES
    ]

    row["taxonomy_v1_category"] = legacy_category
    row["category"] = migrated_category
    row["taxonomy_version"] = 2
    row["quality_tags"] = quality_tags
    row["source_tags"] = source_tags

    if review and review.get("decision") in {
        "accepted",
        "reclassified",
        "ocr_retry",
        "quarantined",
    }:
        row["taxonomy_review_status"] = "reviewed"
    elif legacy_category in AMBIGUOUS_LEGACY_CATEGORIES:
        row["taxonomy_review_status"] = "pending"
    else:
        row["taxonomy_review_status"] = "preserved"
    return row
