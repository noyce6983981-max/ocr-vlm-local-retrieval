from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.minimal_repro import EXPECTED_CPU_METRICS, evaluate, ingest


def test_minimal_cpu_reproduction_is_complete_and_deterministic(
    tmp_path: Path,
) -> None:
    receipt = ingest(tmp_path)
    assert receipt["page_count"] == 40
    assert receipt["query_count"] == 16
    assert len(list((tmp_path / "images").glob("*.png"))) == 40
    manifest = (tmp_path / "manifest.jsonl").read_text(encoding="utf-8")
    assert manifest.count('"license": "CC0-1.0"') == 40

    report = evaluate(tmp_path, backend="cpu")
    assert report["metrics"] == EXPECTED_CPU_METRICS
    assert report["matches_expected_cpu_result"]
    assert all(
        row["missing_reference"]
        for row in report["details"]
        if row["is_no_answer"]
    )
