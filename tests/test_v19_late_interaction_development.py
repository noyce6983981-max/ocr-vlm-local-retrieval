from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_v19_late_interaction_development import build_tasks


def write_payload(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"rankings": {"quality_hybrid": rows}}), encoding="utf-8"
    )


def test_build_tasks_uses_only_guarded_changes_and_unions_candidates(
    tmp_path: Path,
) -> None:
    library = tmp_path / "library"
    image = tmp_path / "image.png"
    image.write_bytes(b"image")
    baseline = tmp_path / "baseline"
    guarded = tmp_path / "guarded"
    rows = [{"item_id": "a", "source_path": str(image)}]
    write_payload(baseline / "q1_v18_frozen.json", rows)
    write_payload(guarded / "q1_b21.json", rows)
    payload = {
        "split": "v19_reviewed_development_only",
        "assignments": [
            {
                "query_id": "q1",
                "query": "查询",
                "guarded_route_changed": True,
                "legacy_route": "text_evidence",
                "guarded_route": "mixed",
            },
            {"query_id": "q2", "guarded_route_changed": False},
        ],
    }
    tasks = build_tasks(
        payload,
        baseline_dir=baseline,
        guarded_dir=guarded,
        library_dir=library,
        top_k=3,
    )
    assert len(tasks) == 1
    assert [row["item_id"] for row in tasks[0]["candidates"]] == ["a"]


def test_build_tasks_rejects_non_development_payload(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="development"):
        build_tasks(
            {"split": "holdout", "assignments": []},
            baseline_dir=tmp_path,
            guarded_dir=tmp_path,
            library_dir=tmp_path,
            top_k=3,
        )
