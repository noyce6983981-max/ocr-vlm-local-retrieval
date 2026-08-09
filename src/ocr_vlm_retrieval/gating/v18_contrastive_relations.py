"""Natural-language directional counterfactuals used by the V18 method."""

from __future__ import annotations

import re

from ocr_vlm_retrieval.gating.contrastive_relations import (
    RelationCounterfactual,
    build_relation_counterfactual,
)

V18_INVERSE_RELATION_PAIRS: tuple[tuple[str, str], ...] = (
    ("前方", "后方"),
    ("前面", "后面"),
    ("上方", "下方"),
    ("上面", "下面"),
    ("左侧", "右侧"),
    ("左边", "右边"),
    ("内部", "外部"),
    ("里面", "外面"),
    ("之上", "之下"),
    ("above", "below"),
    ("in front of", "behind"),
    ("to the left of", "to the right of"),
    ("inside", "outside"),
)


def _pattern(marker: str) -> re.Pattern[str]:
    if re.search(r"[a-z0-9]", marker, flags=re.IGNORECASE):
        return re.compile(
            rf"(?<![a-z0-9]){re.escape(marker)}(?![a-z0-9])",
            flags=re.IGNORECASE,
        )
    return re.compile(re.escape(marker))


def build_v18_relation_counterfactual(
    value: str,
) -> RelationCounterfactual | None:
    """Invert one directional condition, preferring valid natural Chinese."""

    candidates = sorted(
        (
            (positive, negative)
            for left, right in V18_INVERSE_RELATION_PAIRS
            for positive, negative in ((left, right), (right, left))
        ),
        key=lambda pair: len(pair[0]),
        reverse=True,
    )
    for positive, negative in candidates:
        match = _pattern(positive).search(value)
        if match is None:
            continue
        negative_value = f"{value[:match.start()]}{negative}{value[match.end():]}"
        return RelationCounterfactual(
            positive_value=value,
            negative_value=negative_value,
            positive_marker=match.group(0),
            negative_marker=negative,
        )
    return build_relation_counterfactual(value)


def v18_counterfactual_prompt(counterfactual: RelationCounterfactual) -> str:
    return f"候选图像清楚显示这一动作或空间关系：{counterfactual.negative_value}。"
