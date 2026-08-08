from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts/study_query_authoring_app.py"
    spec = importlib.util.spec_from_file_location("study_query_authoring_app", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_submission_save_is_atomic_and_persistent(tmp_path: Path) -> None:
    module = _load_module()
    module.SUBMISSIONS_PATH = tmp_path / "submissions.jsonl"
    module.save_submission({"authoring_id": "source_002", "review_action": "approve"})
    module.save_submission({"authoring_id": "source_001", "review_action": "approve"})
    module.save_submission({"authoring_id": "source_002", "review_action": "reject"})
    assert module.read_jsonl(module.SUBMISSIONS_PATH) == [
        {"authoring_id": "source_001", "review_action": "approve"},
        {"authoring_id": "source_002", "review_action": "reject"},
    ]


def test_translation_guard_and_split_blinding_are_present() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "scripts/study_query_authoring_app.py"
    ).read_text(encoding="utf-8")
    assert 'setAttribute("translate", "no")' in source
    assert "row['split']" not in source
