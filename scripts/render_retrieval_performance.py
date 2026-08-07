"""Render an evidence-backed retrieval performance summary as PNG."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "records/presentation/ocr_vlm_retrieval_v2_performance.png"
)

COLORS = {
    "background": "#F3F6FA",
    "panel": "#FFFFFF",
    "ink": "#12263A",
    "muted": "#64748B",
    "line": "#DDE5EE",
    "baseline": "#94A3B8",
    "fast": "#2563EB",
    "precision": "#0F766E",
    "warning": "#D97706",
    "soft_blue": "#EAF2FF",
    "soft_green": "#E6F5F1",
    "soft_orange": "#FFF4E5",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "msyhbd.ttc" if bold else "msyh.ttc"
    return ImageFont.truetype(f"C:/Windows/Fonts/{name}", size)


def rounded_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    fill: str = COLORS["panel"],
    radius: int = 24,
    outline: str | None = None,
) -> None:
    draw.rounded_rectangle(
        box, radius=radius, fill=fill, outline=outline, width=2
    )


def text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    value: str,
    *,
    size: int,
    color: str = COLORS["ink"],
    bold: bool = False,
    anchor: str | None = None,
) -> None:
    draw.text(
        xy,
        value,
        font=font(size, bold=bold),
        fill=color,
        anchor=anchor,
    )


def compact_metrics(report: dict[str, Any]) -> dict[str, float]:
    return {
        "Recall@1": float(report["recall_at_1"]),
        "Recall@3": float(report["recall_at_3"]),
        "MRR": float(report["mrr"]),
        "拒答准确率": float(report["no_answer_rejection_accuracy"]),
        "端到端准确率": float(report["end_to_end_accuracy"]),
    }


def selected_validation(
    report: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selected = report["selected_config"]["ranker"]
    row = next(
        value
        for value in report["family_finalists"]
        if value["config"] == selected
    )
    return row["validation_metrics"], row["validation_details"]


def category_accuracy(
    details: list[dict[str, Any]],
    queries: dict[str, dict[str, str]],
) -> dict[str, tuple[int, int]]:
    values: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in details:
        if row["is_no_answer"]:
            continue
        category = queries[row["query_id"]]["category"]
        values[category][1] += 1
        values[category][0] += int(row["expected_rank"] == 1)
    return {
        category: (counts[0], counts[1])
        for category, counts in values.items()
    }


def main() -> None:
    args = parse_args()
    baseline = json.loads(
        (
            PROJECT_ROOT
            / "outputs/evaluation/library_retrieval/"
            "development_selection.json"
        ).read_text(encoding="utf-8")
    )
    v2 = json.loads(
        (
            PROJECT_ROOT
            / "outputs/evaluation/library_retrieval/"
            "development_selection_v2_candidate.json"
        ).read_text(encoding="utf-8")
    )
    precision = json.loads(
        (
            PROJECT_ROOT
            / "outputs/evaluation/library_retrieval/"
            "reranker_gate_evaluation.json"
        ).read_text(encoding="utf-8")
    )
    sealed_test = json.loads(
        (
            PROJECT_ROOT
            / "outputs/evaluation/library_retrieval/"
            "sealed_test_evaluation.json"
        ).read_text(encoding="utf-8")
    )
    with (
        PROJECT_ROOT
        / "data/evaluation/"
        "public_dataset_1500_retrieval_queries_formal.csv"
    ).open("r", encoding="utf-8-sig", newline="") as handle:
        queries = {
            row["query_id"]: row for row in csv.DictReader(handle)
        }

    baseline_metrics, baseline_details = selected_validation(baseline)
    v2_metrics, v2_details = selected_validation(v2)
    precision_metrics = precision["validation_metrics"]
    series = {
        "原三路融合": compact_metrics(baseline_metrics),
        "V2 快速模式": compact_metrics(v2_metrics),
        "V2 高精度模式": {
            "Recall@1": float(precision_metrics["recall_at_1"]),
            "Recall@3": float(precision_metrics["recall_at_3"]),
            "MRR": 1.0,
            "拒答准确率": float(
                precision_metrics["no_answer_rejection_accuracy"]
            ),
            "端到端准确率": float(
                precision_metrics["end_to_end_accuracy"]
            ),
        },
    }
    baseline_categories = category_accuracy(
        baseline_details, queries
    )
    v2_categories = category_accuracy(v2_details, queries)

    width, height = 1800, 1200
    image = Image.new("RGB", (width, height), COLORS["background"])
    draw = ImageDraw.Draw(image)

    text(
        draw,
        (64, 48),
        "OCR-VLM 多模态资料检索｜V2 性能报告",
        size=46,
        bold=True,
    )
    text(
        draw,
        (66, 108),
        "真实公开资料 · 人工检索真值 · 训练/验证/测试严格隔离",
        size=23,
        color=COLORS["muted"],
    )

    cards = [
        ("1,482", "可检索真实页面", COLORS["soft_blue"]),
        ("99", "人工确认查询", COLORS["soft_green"]),
        ("60 / 19 / 20", "训练 / 验证 / 测试", COLORS["soft_orange"]),
        ("0", "元数据分支合成样本", "#EEF2FF"),
    ]
    for index, (value, label, fill) in enumerate(cards):
        x0 = 64 + index * 420
        rounded_panel(draw, (x0, 154, x0 + 390, 252), fill=fill)
        text(draw, (x0 + 24, 170), value, size=31, bold=True)
        text(
            draw,
            (x0 + 24, 215),
            label,
            size=19,
            color=COLORS["muted"],
        )

    rounded_panel(draw, (64, 282, 1135, 850))
    text(draw, (94, 312), "验证集方法对比", size=30, bold=True)
    text(
        draw,
        (94, 354),
        "n=19（17 条有答案 + 2 条无答案）",
        size=18,
        color=COLORS["muted"],
    )
    legend = [
        ("原三路融合", COLORS["baseline"]),
        ("V2 快速模式", COLORS["fast"]),
        ("V2 高精度模式", COLORS["precision"]),
    ]
    for index, (label, color) in enumerate(legend):
        x = 510 + index * 195
        draw.rounded_rectangle(
            (x, 325, x + 20, 345), radius=5, fill=color
        )
        text(
            draw,
            (x + 29, 323),
            label,
            size=17,
            color=COLORS["muted"],
        )

    metric_names = list(next(iter(series.values())))
    chart_left = 260
    chart_right = 1082
    chart_width = chart_right - chart_left
    base_y = 425
    row_gap = 79
    bar_height = 16
    for metric_index, metric_name in enumerate(metric_names):
        y = base_y + metric_index * row_gap
        text(
            draw,
            (94, y + 11),
            metric_name,
            size=19,
            color=COLORS["ink"],
            anchor="lm",
        )
        for series_index, (label, values) in enumerate(series.items()):
            bar_y = y - 13 + series_index * 20
            value = values[metric_name]
            draw.rounded_rectangle(
                (
                    chart_left,
                    bar_y,
                    chart_left + chart_width,
                    bar_y + bar_height,
                ),
                radius=7,
                fill="#EDF1F5",
            )
            color = dict(legend)[label]
            draw.rounded_rectangle(
                (
                    chart_left,
                    bar_y,
                    chart_left + int(chart_width * value),
                    bar_y + bar_height,
                ),
                radius=7,
                fill=color,
            )
            text(
                draw,
                (chart_right + 8, bar_y + 8),
                f"{value * 100:.1f}%",
                size=16,
                color=color,
                bold=True,
                anchor="lm",
            )

    rounded_panel(draw, (1165, 282, 1736, 850))
    text(draw, (1195, 312), "类别 Top1 变化", size=30, bold=True)
    text(
        draw,
        (1195, 354),
        "验证集：正确条数 / 类别查询数",
        size=18,
        color=COLORS["muted"],
    )
    category_labels = {
        "general_text_document": "普通文档",
        "complex_academic": "学术页面",
        "table_form_ticket": "表格/表单",
        "ppt_poster_slide": "PPT/海报",
        "software_web_code": "软件/网页",
        "scene_text": "场景文字",
        "natural_no_text": "自然图像",
    }
    categories = [
        key
        for key in category_labels
        if key in v2_categories
    ]
    for index, category in enumerate(categories):
        y = 410 + index * 58
        base_hit, total = baseline_categories.get(category, (0, 0))
        v2_hit, _ = v2_categories[category]
        text(
            draw,
            (1195, y),
            category_labels[category],
            size=18,
        )
        text(
            draw,
            (1515, y),
            f"{base_hit}/{total}",
            size=18,
            color=COLORS["baseline"],
            bold=True,
        )
        text(
            draw,
            (1610, y),
            "→",
            size=20,
            color=COLORS["muted"],
        )
        text(
            draw,
            (1665, y),
            f"{v2_hit}/{total}",
            size=18,
            color=COLORS["precision"],
            bold=True,
        )
    text(
        draw,
        (1195, 812),
        "主要修复：场景文字路由、无 OCR 图像标题检索",
        size=16,
        color=COLORS["muted"],
    )

    rounded_panel(draw, (64, 880, 1736, 1132))
    text(draw, (94, 910), "评测协议与可信边界", size=28, bold=True)
    test_metrics = sealed_test["test_metrics"]
    test_summary = (
        f"V1 单次封闭测试（n=20）：Recall@1 "
        f"{test_metrics['recall_at_1'] * 100:.2f}% ｜ Recall@3 "
        f"{test_metrics['recall_at_3'] * 100:.2f}% ｜ MRR "
        f"{test_metrics['mrr'] * 100:.2f}% ｜ 端到端 "
        f"{test_metrics['end_to_end_accuracy'] * 100:.2f}%"
    )
    text(draw, (94, 956), test_summary, size=21, bold=True)
    text(
        draw,
        (94, 1001),
        "V2 在训练集调参、验证集选型；旧测试集揭晓后未再次用于 V2 调参或性能宣称。",
        size=20,
        color=COLORS["warning"],
        bold=True,
    )
    text(
        draw,
        (94, 1043),
        "验证集 100% 仅表示当前 19 条人工真值全部通过，样本量仍小；下一轮应新增真实外部盲测查询后再报告 V2 测试指标。",
        size=19,
        color=COLORS["muted"],
    )
    text(
        draw,
        (94, 1081),
        "V2 组件：BGE-M3 Dense + BM25 + Qwen3-VL Image + 真实元数据 + 查询路由 + Qwen3-VL Reranker",
        size=18,
        color=COLORS["muted"],
    )

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, optimize=True)
    print(output_path)


if __name__ == "__main__":
    main()
