from __future__ import annotations

from pathlib import Path

from scripts.freeze_v19_retrieval_method import (
    LOCKED_PATHS,
    locked_hashes,
    verify_lock,
)


def test_retrieval_method_lock_detects_changed_file(tmp_path: Path) -> None:
    for relative in LOCKED_PATHS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("frozen", encoding="utf-8")
    payload = {"locked_files": locked_hashes(tmp_path)}
    assert verify_lock(payload, tmp_path) == []
    (tmp_path / LOCKED_PATHS[0]).write_text("changed", encoding="utf-8")
    assert verify_lock(payload, tmp_path) == [f"SHA-256 mismatch: {LOCKED_PATHS[0]}"]


def test_retrieval_method_lock_reports_missing_file(tmp_path: Path) -> None:
    payload = {"locked_files": {}}
    errors = verify_lock(payload, tmp_path)
    assert len(errors) == 1
    assert errors[0].startswith("locked file is missing:")
