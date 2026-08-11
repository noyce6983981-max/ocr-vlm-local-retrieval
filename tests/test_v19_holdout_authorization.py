from __future__ import annotations

from pathlib import Path

import pytest

from scripts.run_intent_routing_study import validate_holdout_claim


def test_formal_holdout_requires_claim(tmp_path: Path) -> None:
    query_path = tmp_path / "holdout.csv"
    query_path.write_text("x", encoding="utf-8")
    with pytest.raises(PermissionError, match="claim"):
        validate_holdout_claim(
            query_path=query_path,
            split="adjudicated_formal_holdout_30pct_double_review",
            claim_path=None,
            mode="hybrid",
        )


def test_non_holdout_does_not_require_claim(tmp_path: Path) -> None:
    validate_holdout_claim(
        query_path=tmp_path / "missing.csv",
        split="adjudicated_formal_calibration_30pct_double_review",
        claim_path=None,
        mode="hybrid",
    )
