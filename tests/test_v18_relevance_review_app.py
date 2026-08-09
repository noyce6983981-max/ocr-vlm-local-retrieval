from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts/v18_relevance_review_app.py"
    spec = importlib.util.spec_from_file_location("v18_relevance_review_app", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _packets(count: int = 80) -> list[dict[str, str]]:
    return [
        {
            "query_id": f"q{index:03d}",
            "study_fingerprint": "study",
        }
        for index in range(count)
    ]


def test_secondary_assignment_is_stable_thirty_percent() -> None:
    module = _load_module()
    packets = _packets()
    first = module.secondary_packets(packets)
    second = module.secondary_packets(list(reversed(packets)))
    assert len(first) == 24
    assert [row["query_id"] for row in first] == [
        row["query_id"] for row in second
    ]


def test_primary_assignment_keeps_all_packets() -> None:
    module = _load_module()
    packets = _packets(3)
    assert module.assigned_packets(packets, "primary") == packets
