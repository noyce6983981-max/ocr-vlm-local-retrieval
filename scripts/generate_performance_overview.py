"""Generate a shareable, evidence-bounded performance overview image."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


WIDTH = 1920
HEIGHT = 1080

METHODS = [
    ("OCR文本", 0.7083, 0.7917, 0.7490),
    ("Qwen3-VL视觉", 0.9167, 0.9583, 0.9435),
    ("固定融合", 0.7917, 0.7917, 0.8090),
    ("OCR质量自适应", 0.9583, 0.9583, 0.9643),
    ("查询感知三路混合", 0.9583, 1.0000, 0.9722),
    ("三路混合 + VLM精排", 1.0000, 1.0000, 1.0000),
]

COLORS = {
    "background": "#F2F6F8",
    "card": "#FFFFFF",
    "navy": "#0B2239",
    "blue": "#3478C7",
    "teal": "#16A897",
    "amber": "#E5A13D",
    "ink": "#17293A",
    "muted": "#65768A",
    "line": "#D9E3EA",
    "soft_blue": "#EAF2FB",
    "soft_teal": "#E6F6F3",
    "soft_amber": "#FFF3DF",
}


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    windows_dir = Path(os.environ.get("WINDIR", "C:/Windows"))
    candidates = [
        windows_dir / "Fonts" / ("msyhbd.ttc" if bold else "msyh.ttc"),
        windows_dir / "Fonts" / ("simhei.ttf" if bold else "simsun.ttc"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def rounded_card(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    fill: str = COLORS["card"],
    radius: int = 24,
) -> None:
    draw.rounded_rectangle(
        box,
        radius=radius,
        fill=fill,
        outline=COLORS["line"],
        width=2,
    )


def draw_gradient_header(image: Image.Image) -> None:
    start = (11, 34, 57)
    end = (22, 91, 102)
    pixels = image.load()
    for y in range(170):
        ratio_y = y / 169
        for x in range(WIDTH):
            ratio = min(1.0, 0.75 * (x / WIDTH) + 0.25 * ratio_y)
            pixels[x, y] = tuple(
                int(start[channel] * (1 - ratio) + end[channel] * ratio)
                for channel in range(3)
            )


def draw_metric_card(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    value: str,
    label: str,
    accent: str,
) -> None:
    rounded_card(draw, box)
    left, top, _, _ = box
    draw.rounded_rectangle(
        (left + 18, top + 20, left + 25, top + 94),
        radius=4,
        fill=accent,
    )
    draw.text(
        (left + 42, top + 18),
        value,
        fill=COLORS["ink"],
        font=font(35, bold=True),
    )
    draw.text(
        (left + 42, top + 66),
        label,
        fill=COLORS["muted"],
        font=font(19),
    )


def draw_performance_chart(draw: ImageDraw.ImageDraw) -> None:
    card = (54, 202, 1314, 886)
    rounded_card(draw, card)
    draw.text(
        (88, 231),
        "24条人工真值查询：检索性能对比",
        fill=COLORS["ink"],
        font=font(29, bold=True),
    )
    draw.text(
        (88, 273),
        "数值越高越好；横轴为0–1完整尺度",
        fill=COLORS["muted"],
        font=font(18),
    )

    legend_x = 760
    legend = [
        ("Recall@1", COLORS["blue"]),
        ("Recall@3", COLORS["teal"]),
        ("MRR", COLORS["amber"]),
    ]
    for label, color in legend:
        draw.rounded_rectangle(
            (legend_x, 246, legend_x + 22, 268),
            radius=5,
            fill=color,
        )
        draw.text(
            (legend_x + 31, 244),
            label,
            fill=COLORS["muted"],
            font=font(17),
        )
        legend_x += 150

    plot_left = 380
    plot_right = 1258
    plot_top = 330
    plot_bottom = 824
    for tick in range(6):
        score = tick / 5
        x = int(plot_left + score * (plot_right - plot_left))
        draw.line(
            (x, plot_top - 10, x, plot_bottom),
            fill=COLORS["line"],
            width=2,
        )
        label = f"{score:.1f}"
        text_box = draw.textbbox((0, 0), label, font=font(15))
        draw.text(
            (x - (text_box[2] - text_box[0]) / 2, plot_bottom + 10),
            label,
            fill=COLORS["muted"],
            font=font(15),
        )

    group_height = 78
    bar_height = 14
    series = [
        (1, COLORS["blue"]),
        (2, COLORS["teal"]),
        (3, COLORS["amber"]),
    ]
    for index, method in enumerate(METHODS):
        y = plot_top + index * group_height
        draw.text(
            (88, y + 18),
            method[0],
            fill=COLORS["ink"],
            font=font(20, bold=index >= 3),
        )
        for series_index, color in series:
            value = method[series_index]
            bar_y = y + (series_index - 1) * 19
            bar_end = int(
                plot_left + value * (plot_right - plot_left)
            )
            draw.rounded_rectangle(
                (plot_left, bar_y, bar_end, bar_y + bar_height),
                radius=7,
                fill=color,
            )
            value_text = f"{value:.3f}"
            text_width = draw.textbbox(
                (0, 0), value_text, font=font(14, bold=True)
            )[2]
            text_x = min(bar_end + 8, plot_right - text_width)
            draw.text(
                (text_x, bar_y - 3),
                value_text,
                fill=COLORS["ink"],
                font=font(14, bold=True),
            )


def draw_right_panel(draw: ImageDraw.ImageDraw) -> None:
    draw.text(
        (1350, 211),
        "复现与工程证据",
        fill=COLORS["ink"],
        font=font(27, bold=True),
    )
    draw_metric_card(
        draw,
        (1344, 258, 1587, 374),
        "24",
        "人工审核查询",
        COLORS["blue"],
    )
    draw_metric_card(
        draw,
        (1604, 258, 1847, 374),
        "1,482",
        "当前有效页面",
        COLORS["teal"],
    )
    draw_metric_card(
        draw,
        (1344, 391, 1587, 507),
        "4.609 GiB",
        "重排峰值显存",
        COLORS["amber"],
    )
    draw_metric_card(
        draw,
        (1604, 391, 1847, 507),
        "112",
        "自动化测试通过",
        COLORS["blue"],
    )

    rounded_card(
        draw,
        (1344, 532, 1847, 700),
        fill=COLORS["soft_teal"],
    )
    draw.text(
        (1372, 555),
        "关键观察",
        fill=COLORS["ink"],
        font=font(23, bold=True),
    )
    observations = [
        "• 视觉分支明显补偿纯OCR的检索盲区",
        "• 查询感知融合的Recall@3达到1.000",
        "• VLM精排在当前24条查询上达到24/24 Top 1",
        "• 等权RRF Recall@1仅0.750，作为负结果保留",
    ]
    for index, line in enumerate(observations):
        draw.text(
            (1372, 598 + index * 25),
            line,
            fill=COLORS["ink"],
            font=font(16),
        )

    rounded_card(
        draw,
        (1344, 718, 1847, 886),
        fill=COLORS["soft_amber"],
    )
    draw.text(
        (1372, 741),
        "结论边界",
        fill=COLORS["ink"],
        font=font(23, bold=True),
    )
    boundaries = [
        "检索指标来自当前24条人工真值查询，",
        "只证明本机链路复现与当前样本有效，",
        "不能外推为大规模公开基准性能。",
        "1500页资料库仅用于工程压力与质量治理。",
    ]
    for index, line in enumerate(boundaries):
        draw.text(
            (1372, 783 + index * 24),
            line,
            fill=COLORS["ink"],
            font=font(16),
        )


def draw_footer_pipeline(draw: ImageDraw.ImageDraw) -> None:
    draw.text(
        (58, 927),
        "系统主链路",
        fill=COLORS["muted"],
        font=font(18, bold=True),
    )
    nodes = [
        ("OCR → BGE-M3", COLORS["soft_blue"]),
        ("BM25关键词", COLORS["soft_blue"]),
        ("Qwen3-VL视觉", COLORS["soft_teal"]),
        ("查询感知融合", COLORS["soft_teal"]),
        ("Qwen3-VL精排", COLORS["soft_amber"]),
    ]
    x = 200
    for index, (label, color) in enumerate(nodes):
        box = (x, 908, x + 270, 970)
        draw.rounded_rectangle(
            box,
            radius=15,
            fill=color,
            outline=COLORS["line"],
            width=2,
        )
        text_box = draw.textbbox((0, 0), label, font=font(19, bold=True))
        draw.text(
            (
                x + (270 - (text_box[2] - text_box[0])) / 2,
                924,
            ),
            label,
            fill=COLORS["ink"],
            font=font(19, bold=True),
        )
        if index < len(nodes) - 1:
            if index < 2:
                draw.text(
                    (x + 288, 920),
                    "+",
                    fill=COLORS["muted"],
                    font=font(27, bold=True),
                )
            else:
                draw.line(
                    (x + 280, 939, x + 316, 939),
                    fill=COLORS["muted"],
                    width=3,
                )
                draw.polygon(
                    [
                        (x + 316, 939),
                        (x + 304, 932),
                        (x + 304, 946),
                    ],
                    fill=COLORS["muted"],
                )
        x += 330

    draw.text(
        (58, 1024),
        "实验环境：Windows 11 · RTX 4060 Laptop GPU 8GB · 本地运行 · 不调用商业API",
        fill=COLORS["muted"],
        font=font(17),
    )
    source_text = "数据记录：README与records/experiments · 生成日期 2026-07-30"
    source_width = draw.textbbox(
        (0, 0), source_text, font=font(15)
    )[2]
    draw.text(
        (WIDTH - 58 - source_width, 1027),
        source_text,
        fill=COLORS["muted"],
        font=font(15),
    )


def generate(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (WIDTH, HEIGHT), COLORS["background"])
    draw_gradient_header(image)
    draw = ImageDraw.Draw(image)
    draw.text(
        (56, 35),
        "OCR-VLM 多模态资料检索",
        fill="#FFFFFF",
        font=font(49, bold=True),
    )
    draw.text(
        (58, 101),
        "性能概览｜OCR文本 + BM25 + Qwen3-VL视觉 + 查询感知融合与精排",
        fill="#C9E3E3",
        font=font(23),
    )
    draw.text(
        (1650, 55),
        "REPRODUCED LOCALLY",
        fill="#B6DED7",
        font=font(15, bold=True),
    )

    draw_performance_chart(draw)
    draw_right_panel(draw)
    draw_footer_pipeline(draw)
    image.save(output_path, format="PNG", optimize=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "records/presentation/ocr_vlm_model_performance.png"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate(args.output)
    print(args.output.resolve())
