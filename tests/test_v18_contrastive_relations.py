from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from ocr_vlm_retrieval.gating.v18_contrastive_relations import (
    build_v18_relation_counterfactual,
    v18_counterfactual_prompt,
)


def test_natural_chinese_front_relation_is_inverted() -> None:
    row = build_v18_relation_counterfactual("农田位于山脉的前方")
    assert row is not None
    assert row.negative_value == "农田位于山脉的后方"
    assert row.positive_marker == "前方"


def test_natural_chinese_vertical_relation_is_inverted() -> None:
    row = build_v18_relation_counterfactual("列车在水面上方")
    assert row is not None
    assert row.negative_value == "列车在水面下方"
    assert "下方" in v18_counterfactual_prompt(row)
