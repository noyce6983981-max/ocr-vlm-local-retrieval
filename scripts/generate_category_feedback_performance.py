"""Generate the final frozen category-feedback performance figure."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_performance_overview import font


WIDTH = 1800
HEIGHT = 1050
COLORS = {
    "background": "#F2F6F8",
    "card": "#FFFFFF",
    "navy": "#102A43",
    "blue": "#2F6FAD",
    "teal": "#159A8C",
    "amber": "#DB923A",
    "red": "#C75B5B",
    "ink": "#172B3A",
    "muted": "#65778A",
    "line": "#D7E1E8",
    "soft_teal": "#E7F6F3",
    "soft_amber": "#FFF3DF",
}


def card(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    fill: str = COLORS["card"],
) -> None:
    draw.rounded_rectangle(
        box,
        radius=22,
        fill=fill,
        outline=COLORS["line"],
        width=2,
    )


def metric_card(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    value: str,
    label: str,
    accent: str,
) -> None:
    card(draw, box)
    left, top, _, _ = box
    draw.rounded_rectangle(
        (left + 18, top + 18, left + 26, top + 102),
        radius=4,
        fill=accent,
    )
    draw.text(
        (left + 45, top + 18),
        value,
        fill=COLORS["ink"],
        font=font(37, bold=True),
    )
    draw.text(
        (left + 45, top + 70),
        label,
        fill=COLORS["muted"],
        font=font(18),
    )


def draw_accuracy_panel(
    draw: ImageDraw.ImageDraw,
    validation: dict,
    test: dict,
) -> None:
    box = (48, 188, 1112, 646)
    card(draw, box)
    draw.text(
        (80, 218),
        "冻结策略：验证集与测试集 Accuracy",
        fill=COLORS["ink"],
        font=font(28, bold=True),
    )
    draw.text(
        (80, 258),
        "阈值0.80只由验证集选择；测试集仅评估一次",
        fill=COLORS["muted"],
        font=font(17),
    )
    methods = [
        (
            "旧版混合类别",
            "legacy_baseline_accuracy",
            COLORS["red"],
        ),
        (
            "当前规则/来源标签",
            "current_baseline_accuracy",
            COLORS["blue"],
        ),
        (
            "反馈模型单独分类",
            "standalone_model_accuracy",
            COLORS["amber"],
        ),
        ("规则 + 高置信纠错", "accuracy", COLORS["teal"]),
    ]
    metrics_validation = validation["metrics"]
    metrics_test = test["metrics"]
    plot_left = 382
    plot_right = 1055
    plot_top = 326
    plot_bottom = 590
    for tick in range(6):
        value = tick / 5
        x = int(plot_left + value * (plot_right - plot_left))
        draw.line(
            (x, plot_top - 15, x, plot_bottom),
            fill=COLORS["line"],
            width=2,
        )
        draw.text(
            (x - 12, plot_bottom + 8),
            f"{value:.1f}",
            fill=COLORS["muted"],
            font=font(14),
        )

    for index, (label, key, color) in enumerate(methods):
        y = plot_top + index * 65
        draw.text(
            (80, y + 8),
            label,
            fill=COLORS["ink"],
            font=font(19, bold=key == "accuracy"),
        )
        values = [
            ("验证", float(metrics_validation[key]), color),
            ("测试", float(metrics_test[key]), COLORS["navy"]),
        ]
        for bar_index, (split_label, value, bar_color) in enumerate(
            values
        ):
            bar_y = y + bar_index * 22
            bar_end = int(
                plot_left + value * (plot_right - plot_left)
            )
            draw.rounded_rectangle(
                (plot_left, bar_y, bar_end, bar_y + 15),
                radius=7,
                fill=bar_color,
            )
            draw.text(
                (bar_end + 7, bar_y - 4),
                f"{split_label} {value:.1%}",
                fill=COLORS["ink"],
                font=font(14, bold=True),
            )


def draw_class_panel(draw: ImageDraw.ImageDraw, test: dict) -> None:
    box = (48, 676, 1752, 980)
    card(draw, box)
    draw.text(
        (80, 706),
        "最终测试：各类别 F1 与样本量",
        fill=COLORS["ink"],
        font=font(27, bold=True),
    )
    draw.text(
        (80, 746),
        "Macro-F1受少样本类别影响；学术/技术文档测试集仅1页",
        fill=COLORS["muted"],
        font=font(16),
    )
    items = test["per_class"]
    chart_left = 80
    chart_top = 810
    item_width = 226
    max_height = 116
    for index, item in enumerate(items):
        left = chart_left + index * item_width
        value = float(item["f1"])
        bar_height = int(value * max_height)
        bar_color = (
            COLORS["teal"]
            if value >= 0.8
            else COLORS["amber"]
            if value >= 0.5
            else COLORS["red"]
        )
        draw.rounded_rectangle(
            (
                left + 66,
                chart_top + max_height - bar_height,
                left + 118,
                chart_top + max_height,
            ),
            radius=10,
            fill=bar_color,
        )
        draw.text(
            (left + 55, chart_top + max_height - bar_height - 28),
            f"{value:.3f}",
            fill=COLORS["ink"],
            font=font(15, bold=True),
        )
        label = str(item["label"]).replace("文档", "")
        draw.text(
            (left, chart_top + max_height + 12),
            label,
            fill=COLORS["ink"],
            font=font(15, bold=True),
        )
        draw.text(
            (left, chart_top + max_height + 38),
            f"n={int(item['support'])}",
            fill=COLORS["muted"],
            font=font(14),
        )


def generate(
    validation_report: Path,
    test_report: Path,
    output_path: Path,
) -> None:
    validation = json.loads(
        validation_report.read_text(encoding="utf-8")
    )
    test = json.loads(test_report.read_text(encoding="utf-8"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (WIDTH, HEIGHT), COLORS["background"])
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, WIDTH, 158), fill=COLORS["navy"])
    draw.text(
        (48, 30),
        "OCR-VLM 类别反馈学习｜最终冻结测试",
        fill="#FFFFFF",
        font=font(44, bold=True),
    )
    draw.text(
        (50, 93),
        "Qwen3-VL视觉 + BGE-M3文本 + OCR质量特征 · 规则基线 + 高置信模型纠错",
        fill="#CBE3EC",
        font=font(21),
    )

    draw_accuracy_panel(draw, validation, test)
    test_metrics = test["metrics"]
    metric_card(
        draw,
        (1140, 188, 1429, 310),
        "95",
        "独立测试页",
        COLORS["blue"],
    )
    metric_card(
        draw,
        (1460, 188, 1749, 310),
        f"{float(test_metrics['accuracy']):.1%}",
        "最终 Accuracy",
        COLORS["teal"],
    )
    metric_card(
        draw,
        (1140, 330, 1429, 452),
        f"{float(test_metrics['macro_f1']):.3f}",
        "最终 Macro-F1",
        COLORS["amber"],
    )
    metric_card(
        draw,
        (1460, 330, 1749, 452),
        str(test_metrics["override_count"]),
        "测试集模型覆盖",
        COLORS["teal"],
    )
    card(
        draw,
        (1140, 474, 1749, 646),
        fill=COLORS["soft_amber"],
    )
    draw.text(
        (1172, 500),
        "结论边界",
        fill=COLORS["ink"],
        font=font(23, bold=True),
    )
    notes = [
        "• 规则基线 81/95，门控方案 82/95",
        "• 实际只多纠正1页，不宣称统计显著",
        "• 独立模型仅71/95，不能替代规则",
        "• 测试完成后不再调整模型或阈值",
    ]
    for index, line in enumerate(notes):
        draw.text(
            (1172, 543 + index * 24),
            line,
            fill=COLORS["ink"],
            font=font(16),
        )
    draw_class_panel(draw, test)
    draw.text(
        (48, 1010),
        "分组隔离：训练410页｜验证95页｜测试95页　·　84项分歧已完成人工裁决　·　2026-07-30",
        fill=COLORS["muted"],
        font=font(16),
    )
    image.save(output_path, format="PNG", optimize=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validation-report",
        type=Path,
        default=Path(
            "outputs/user_library/feedback_model/evaluation/"
            "validation_report.json"
        ),
    )
    parser.add_argument(
        "--test-report",
        type=Path,
        default=Path(
            "outputs/user_library/feedback_model/evaluation/"
            "test_report.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "records/presentation/"
            "category_feedback_final_performance.png"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate(
        args.validation_report,
        args.test_report,
        args.output,
    )
    print(args.output.resolve())
