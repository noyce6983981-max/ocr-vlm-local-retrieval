from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from ocr_vlm_retrieval.evaluation.protocol_lock import (
    canonical_json_sha256,
    create_one_shot_receipt,
    directory_manifest_sha256,
    file_sha256,
    read_jsonl,
    resolve_relative_path,
    validate_method_lock,
)
from ocr_vlm_retrieval.evaluation.query_collection import query_fingerprint


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def git(project: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=project,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def make_lock(tmp_path: Path) -> tuple[Path, Path, dict]:
    project = tmp_path / "project"
    runtime = tmp_path / "runtime"
    project.mkdir()
    runtime.mkdir()
    method = project / "method.py"
    governance = project / "runner.py"
    method.write_bytes(b"METHOD = 'v17'\n")
    governance.write_bytes(b"LOCKED = True\n")
    git(project, "init")
    git(project, "config", "user.email", "ci@example.invalid")
    git(project, "config", "user.name", "CI")
    git(project, "add", "method.py", "runner.py")
    git(project, "commit", "-m", "locked method")
    commit = git(project, "rev-parse", "HEAD")

    rows = [
        {
            "query_id": "q1",
            "query": " Red   sign ",
            "split": "holdout",
            "group_id": "g1",
        },
        {
            "query_id": "q2",
            "query": "blue car",
            "split": "calibration",
            "group_id": "g2",
        },
    ]
    frozen = runtime / "queries.jsonl"
    write_jsonl(frozen, rows)
    calibration = runtime / "calibration.json"
    calibration.write_text("{}\n", encoding="utf-8")
    model = runtime / "model"
    model.mkdir()
    (model / "weights.bin").write_bytes(b"frozen-weights")
    (model / ".git").mkdir()
    (model / ".git" / "ignored").write_text("ignore", encoding="utf-8")
    manifest_hash, manifest = directory_manifest_sha256(model)
    prompt = {"full_query_instruction": "verify all conditions"}
    lock = {
        "schema_version": 1,
        "git_commit_sha": commit,
        "locked_files": {"method.py": file_sha256(method)},
        "governance_files": {"runner.py": file_sha256(governance)},
        "runtime_files": {
            "queries.jsonl": file_sha256(frozen),
            "calibration.json": file_sha256(calibration),
        },
        "query_set_path": "queries.jsonl",
        "query_set_sha256": query_fingerprint(rows),
        "verification_prompt_text": prompt,
        "verification_prompt_sha256": canonical_json_sha256(prompt),
        "model": {
            "path": "model",
            "weight_manifest_sha256": manifest_hash,
            "weight_file_count": manifest["file_count"],
            "weight_total_bytes": manifest["total_bytes"],
        },
    }
    return project, runtime, lock


def test_validate_method_lock_accepts_exact_snapshot(tmp_path: Path) -> None:
    project, runtime, lock = make_lock(tmp_path)

    report = validate_method_lock(
        lock,
        project_root=project,
        runtime_root=runtime,
    )

    assert report["valid"], report
    assert report["git_tree_clean"]
    assert report["locked_file_count"] == 1
    assert report["governance_file_count"] == 1
    assert report["runtime_file_count"] == 2
    assert read_jsonl(runtime / "queries.jsonl")[0]["query_id"] == "q1"


def test_validate_method_lock_reports_tampering(tmp_path: Path) -> None:
    project, runtime, lock = make_lock(tmp_path)
    (project / "method.py").write_bytes(b"METHOD = 'changed'\n")
    (project / "runner.py").unlink()
    (runtime / "calibration.json").write_text('{"changed": true}\n', encoding="utf-8")
    (runtime / "queries.jsonl").write_text("{}\n", encoding="utf-8")
    (runtime / "model" / "weights.bin").write_bytes(b"changed")
    lock["verification_prompt_sha256"] = "0" * 64

    report = validate_method_lock(
        lock,
        project_root=project,
        runtime_root=runtime,
    )

    assert not report["valid"]
    joined = " | ".join(report["errors"])
    assert "working tree is not clean" in joined
    assert "locked file hash mismatch" in joined
    assert "governance file is missing" in joined
    assert "runtime file hash mismatch" in joined
    assert "frozen query set cannot be fingerprinted" in joined
    assert "verification prompt hash mismatch" in joined
    assert "model weight manifest hash mismatch" in joined


def test_invalid_lock_metadata_and_path_escape_are_rejected(tmp_path: Path) -> None:
    project, runtime, lock = make_lock(tmp_path)
    lock["schema_version"] = 0
    lock["git_commit_sha"] = "f" * 40
    lock["locked_files"] = {"../escape": "0" * 64}
    lock["governance_files"] = []
    lock["runtime_files"] = []
    lock["query_set_path"] = ""
    lock["verification_prompt_text"] = None
    lock["model"] = {"path": "", "weight_file_count": -1}

    report = validate_method_lock(
        lock,
        project_root=project,
        runtime_root=runtime,
        require_clean=False,
    )

    assert not report["valid"]
    assert any("escapes its root" in error for error in report["errors"])
    assert any(
        "locked Git commit is unavailable" in error for error in report["errors"]
    )
    with pytest.raises(ValueError, match="must be relative"):
        resolve_relative_path(runtime, runtime / "absolute.json")


def test_one_shot_receipt_cannot_be_overwritten(tmp_path: Path) -> None:
    receipt = tmp_path / "nested" / "receipt.json"
    create_one_shot_receipt(receipt, {"status": "claimed"})

    assert json.loads(receipt.read_text(encoding="utf-8"))["status"] == "claimed"
    with pytest.raises(FileExistsError):
        create_one_shot_receipt(receipt, {"status": "second"})
