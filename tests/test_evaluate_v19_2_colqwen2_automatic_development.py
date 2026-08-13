from __future__ import annotations

import argparse
import json

import pytest

from scripts import evaluate_v19_2_colqwen2_automatic_development as evaluator


def test_run_rejects_any_nonautomatic_split(tmp_path) -> None:  # type: ignore[no-untyped-def]
    assignments = tmp_path / "assignments.json"
    assignments.write_text(
        json.dumps({"split": "v19_1_human_reviewed_development_only"}),
        encoding="utf-8",
    )
    args = argparse.Namespace(assignments=assignments)
    with pytest.raises(ValueError, match="automatic development only"):
        evaluator.run(args)


def test_run_rejects_missing_machine_only_marker(tmp_path) -> None:  # type: ignore[no-untyped-def]
    assignments = tmp_path / "assignments.json"
    assignments.write_text(
        json.dumps({"split": evaluator.SPLIT, "human_review_used": True}),
        encoding="utf-8",
    )
    args = argparse.Namespace(assignments=assignments)
    with pytest.raises(ValueError, match="must not use human review"):
        evaluator.run(args)
