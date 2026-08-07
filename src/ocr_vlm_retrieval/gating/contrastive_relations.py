"""Directional counterfactuals for compositional candidate verification.

The verifier can assign a high score when both relation endpoints are visible
even if their direction is reversed.  This module builds an auditable inverse
relation and requires the positive score to beat that counterfactual by a
calibrated margin.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class RelationCounterfactual:
    """One deterministic positive/inverse relation pair."""

    positive_value: str
    negative_value: str
    positive_marker: str
    negative_marker: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Longest markers are tried first so that ``on`` cannot consume
# ``on the back of`` or ``on the left``.
INVERSE_RELATION_PAIRS: tuple[tuple[str, str], ...] = (
    ("on the back of", "on the front of"),
    ("in front of", "behind"),
    ("to the left of", "to the right of"),
    ("on the left of", "on the right of"),
    ("on the left", "on the right"),
    ("at the bottom of", "at the top of"),
    ("at the bottom", "at the top"),
    ("above", "below"),
    ("under", "over"),
    ("inside", "outside"),
    ("front", "rear"),
    ("on", "under"),
    ("上方", "下方"),
    ("前面", "后面"),
    ("左边", "右边"),
)


def _marker_pattern(marker: str) -> re.Pattern[str]:
    if re.search(r"[a-z0-9]", marker, flags=re.IGNORECASE):
        return re.compile(
            rf"(?<![a-z0-9]){re.escape(marker)}(?![a-z0-9])",
            flags=re.IGNORECASE,
        )
    return re.compile(re.escape(marker))


def build_relation_counterfactual(value: str) -> RelationCounterfactual | None:
    """Replace one directional marker with its registered inverse."""

    candidates = sorted(
        (
            (positive, negative)
            for left, right in INVERSE_RELATION_PAIRS
            for positive, negative in ((left, right), (right, left))
        ),
        key=lambda pair: len(pair[0]),
        reverse=True,
    )
    for positive, negative in candidates:
        match = _marker_pattern(positive).search(value)
        if match is None:
            continue
        negative_value = f"{value[: match.start()]}{negative}{value[match.end() :]}"
        return RelationCounterfactual(
            positive_value=value,
            negative_value=negative_value,
            positive_marker=match.group(0),
            negative_marker=negative,
        )
    return None


def counterfactual_prompt(counterfactual: RelationCounterfactual) -> str:
    """Mirror the positive relation template with only direction changed."""

    return (
        f"画面中清楚显示{counterfactual.negative_value}"
        "这一动作或空间关系。"
    )


def relation_margin(positive_score: float, negative_score: float) -> float:
    return float(positive_score) - float(negative_score)


def relation_evidence_passes(
    *,
    positive_score: float,
    negative_score: float,
    absolute_threshold: float,
    margin_threshold: float,
) -> bool:
    """Require both absolute positive evidence and positive-vs-inverse margin."""

    return bool(
        float(positive_score) >= float(absolute_threshold)
        and relation_margin(positive_score, negative_score)
        >= float(margin_threshold)
    )
