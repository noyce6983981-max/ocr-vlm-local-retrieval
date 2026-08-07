from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


SCALE = 2


def scaled(value: int) -> int:
    return value * SCALE


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), scaled(size))
    return ImageFont.load_default()


def text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    value: str,
    size: int,
    *,
    color: str,
    bold: bool = False,
    anchor: str | None = None,
) -> None:
    draw.text(
        (scaled(xy[0]), scaled(xy[1])),
        value,
        fill=color,
        font=font(size, bold),
        anchor=anchor,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = json.loads(args.summary.read_text(encoding="utf-8-sig"))
    round4 = data["model_metrics"]["round4"]
    round5 = data["model_metrics"]["round5"]
    groups = ["Validation Accuracy", "Validation Macro-F1", "Test Accuracy", "Test Macro-F1"]
    values4 = [
        round4["validation"]["accuracy"],
        round4["validation"]["macro_f1"],
        round4["test"]["accuracy"],
        round4["test"]["macro_f1"],
    ]
    values5 = [
        round5["validation"]["accuracy"],
        round5["validation"]["macro_f1"],
        round5["test"]["accuracy"],
        round5["test"]["macro_f1"],
    ]

    canvas = Image.new("RGB", (scaled(1500), scaled(900)), "#F4F7FA")
    draw = ImageDraw.Draw(canvas)
    text(
        draw,
        (70, 48),
        "OCR-VLM Category Feedback Model — Two Full Blind Review Passes",
        27,
        color="#10253B",
        bold=True,
    )
    text(
        draw,
        (70, 92),
        "Same 600-page cohort; validation selects the candidate; test labels never enter fitting.",
        15,
        color="#607286",
    )
    cards = [
        ("Blind-label agreement", f"{data['overall']['agreement']:.1%}", "#176B87"),
        ("Cohen's kappa", f"{data['overall']['cohen_kappa']:.3f}", "#2A9D8F"),
        ("Protocol", "410 / 95 / 95", "#D97706"),
    ]
    card_y = 135
    card_w = 430
    for index, (title, value, color) in enumerate(cards):
        left = 70 + index * 470
        draw.rounded_rectangle(
            (scaled(left), scaled(card_y), scaled(left + card_w), scaled(card_y + 150)),
            radius=scaled(16),
            fill="white",
            outline="#DCE4EA",
            width=scaled(1),
        )
        text(draw, (left + 28, card_y + 30), title, 15, color="#526273", bold=True)
        text(draw, (left + 28, card_y + 74), value, 31, color=color, bold=True)

    chart = (70, 325, 1430, 800)
    draw.rounded_rectangle(
        tuple(scaled(value) for value in chart),
        radius=scaled(16),
        fill="white",
        outline="#DCE4EA",
        width=scaled(1),
    )
    left, top, right, bottom = 145, 380, 1380, 725
    minimum, maximum = 0.60, 0.86
    for tick in (0.60, 0.65, 0.70, 0.75, 0.80, 0.85):
        y = bottom - (tick - minimum) / (maximum - minimum) * (bottom - top)
        draw.line(
            (scaled(left), scaled(int(y)), scaled(right), scaled(int(y))),
            fill="#E1E7EC",
            width=scaled(1),
        )
        text(draw, (left - 18, int(y)), f"{tick:.2f}", 12, color="#607286", anchor="rm")

    group_width = (right - left) / len(groups)
    bar_width = 76
    for index, group in enumerate(groups):
        center = left + group_width * (index + 0.5)
        for offset, value, color in (
            (-bar_width * 0.56, values4[index], "#176B87"),
            (bar_width * 0.56, values5[index], "#E9A23B"),
        ):
            x_center = center + offset
            y = bottom - (value - minimum) / (maximum - minimum) * (bottom - top)
            draw.rounded_rectangle(
                (
                    scaled(int(x_center - bar_width / 2)),
                    scaled(int(y)),
                    scaled(int(x_center + bar_width / 2)),
                    scaled(bottom),
                ),
                radius=scaled(6),
                fill=color,
            )
            text(
                draw,
                (int(x_center), int(y - 11)),
                f"{value:.3f}",
                12,
                color="#243746",
                bold=True,
                anchor="mb",
            )
        text(draw, (int(center), bottom + 22), group, 12, color="#334A5E", anchor="ma")

    draw.rounded_rectangle(
        (scaled(1135), scaled(345), scaled(1155), scaled(365)),
        radius=scaled(3),
        fill="#176B87",
    )
    text(draw, (1165, 343), "Round 4", 13, color="#334A5E")
    draw.rounded_rectangle(
        (scaled(1255), scaled(345), scaled(1275), scaled(365)),
        radius=scaled(3),
        fill="#E9A23B",
    )
    text(draw, (1285, 343), "Round 5", 13, color="#334A5E")
    text(
        draw,
        (70, 842),
        "Two-pass review complete: all 84 disagreements were human-adjudicated before final model deployment.",
        14,
        color="#334A5E",
        bold=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.resize((1500, 900), Image.Resampling.LANCZOS).save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
