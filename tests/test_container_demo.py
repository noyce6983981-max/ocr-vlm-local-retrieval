from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_container_demo import DemoVerificationError, run_demo

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_RESULT = PROJECT_ROOT / "repro/expected_results.json"


def test_container_demo_matches_frozen_result(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    result = run_demo(output_dir, EXPECTED_RESULT)

    assert result["status"] == "passed"
    assert result["page_count"] == 40
    assert result["query_count"] == 16
    assert result["visual_index_built"] is False
    verification = json.loads(
        (output_dir / "verification.json").read_text(encoding="utf-8")
    )
    assert verification == result


def test_container_demo_rejects_changed_expectation(tmp_path: Path) -> None:
    expected = json.loads(EXPECTED_RESULT.read_text(encoding="utf-8"))
    expected["metrics"]["end_to_end_top1_accuracy"] = 0.5
    changed_expected = tmp_path / "changed_expected.json"
    changed_expected.write_text(
        json.dumps(expected, ensure_ascii=False), encoding="utf-8"
    )

    output_dir = tmp_path / "output"
    with pytest.raises(DemoVerificationError):
        run_demo(output_dir, changed_expected)

    verification = json.loads(
        (output_dir / "verification.json").read_text(encoding="utf-8")
    )
    assert verification["status"] == "failed"
    assert verification["expected"] != verification["observed"]
