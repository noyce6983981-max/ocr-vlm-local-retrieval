"""Hash validation and one-shot receipt primitives for the V17 holdout."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ocr_vlm_retrieval.evaluation.query_collection import query_fingerprint


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    material = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(material).hexdigest()


def directory_manifest(directory: Path) -> dict[str, Any]:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    files: list[dict[str, Any]] = []
    total_bytes = 0
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        relative = path.relative_to(directory).as_posix()
        if "/.git/" in f"/{relative}/" or relative.startswith(".git/"):
            continue
        size = path.stat().st_size
        total_bytes += size
        files.append(
            {
                "path": relative,
                "size": size,
                "sha256": file_sha256(path),
            }
        )
    return {
        "file_count": len(files),
        "total_bytes": total_bytes,
        "files": files,
    }


def directory_manifest_sha256(directory: Path) -> tuple[str, dict[str, Any]]:
    manifest = directory_manifest(directory)
    return canonical_json_sha256(manifest), manifest


def _git(
    project_root: Path,
    *args: str,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *args],
        cwd=project_root,
        check=check,
        capture_output=True,
    )


def git_tree_clean(project_root: Path) -> bool:
    return not _git(project_root, "status", "--porcelain").stdout.strip()


def resolve_relative_path(root: Path, relative: str) -> Path:
    """Resolve a lock-controlled path without allowing an absolute escape."""

    value = Path(relative)
    if value.is_absolute():
        raise ValueError(f"Locked path must be relative: {relative}")
    resolved_root = root.resolve()
    resolved = (resolved_root / value).resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(f"Locked path escapes its root: {relative}")
    return resolved


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def validate_method_lock(
    lock: Mapping[str, Any],
    *,
    project_root: Path,
    runtime_root: Path,
    require_clean: bool = True,
) -> dict[str, Any]:
    """Validate code snapshot, runtime artifacts, prompts, and model weights."""

    errors: list[str] = []
    if int(lock.get("schema_version", 0)) != 1:
        errors.append("method lock schema_version must be 1")
    commit = str(lock.get("git_commit_sha", ""))
    if not commit:
        errors.append("git_commit_sha is missing")
    elif _git(
        project_root,
        "cat-file",
        "-e",
        f"{commit}^{{commit}}",
        check=False,
    ).returncode:
        errors.append("locked Git commit is unavailable")
    clean = git_tree_clean(project_root)
    if require_clean and not clean:
        errors.append("Git working tree is not clean")

    locked_files = lock.get("locked_files", {})
    if not isinstance(locked_files, Mapping):
        errors.append("locked_files must be a mapping")
        locked_files = {}
    for relative, expected in locked_files.items():
        try:
            path = resolve_relative_path(project_root, str(relative))
        except ValueError as error:
            errors.append(str(error))
            continue
        if not path.is_file():
            errors.append(f"locked file is missing: {relative}")
            continue
        current_hash = file_sha256(path)
        if current_hash != str(expected):
            errors.append(f"locked file hash mismatch: {relative}")
        if commit:
            snapshot = _git(
                project_root,
                "show",
                f"{commit}:{relative}",
                check=False,
            )
            if snapshot.returncode:
                errors.append(f"locked file absent from Git snapshot: {relative}")
            elif hashlib.sha256(snapshot.stdout).hexdigest() != str(expected):
                errors.append(f"Git snapshot hash mismatch: {relative}")

    retrieval_dependencies = lock.get("retrieval_dependency_files", {})
    if not isinstance(retrieval_dependencies, Mapping):
        errors.append("retrieval_dependency_files must be a mapping")
        retrieval_dependencies = {}
    for relative, expected in retrieval_dependencies.items():
        try:
            path = resolve_relative_path(project_root, str(relative))
        except ValueError as error:
            errors.append(str(error))
            continue
        if not path.is_file():
            errors.append(f"retrieval dependency is missing: {relative}")
            continue
        if file_sha256(path) != str(expected):
            errors.append(f"retrieval dependency hash mismatch: {relative}")
        if commit:
            snapshot = _git(
                project_root,
                "show",
                f"{commit}:{relative}",
                check=False,
            )
            if snapshot.returncode:
                errors.append(
                    f"retrieval dependency absent from Git snapshot: {relative}"
                )
            elif hashlib.sha256(snapshot.stdout).hexdigest() != str(expected):
                errors.append(
                    f"retrieval dependency Git snapshot mismatch: {relative}"
                )

    governance_files = lock.get("governance_files", {})
    if not isinstance(governance_files, Mapping):
        errors.append("governance_files must be a mapping")
        governance_files = {}
    for relative, expected in governance_files.items():
        try:
            path = resolve_relative_path(project_root, str(relative))
        except ValueError as error:
            errors.append(str(error))
            continue
        if not path.is_file():
            errors.append(f"governance file is missing: {relative}")
        elif file_sha256(path) != str(expected):
            errors.append(f"governance file hash mismatch: {relative}")

    runtime_files = lock.get("runtime_files", {})
    if not isinstance(runtime_files, Mapping):
        errors.append("runtime_files must be a mapping")
        runtime_files = {}
    for relative, expected in runtime_files.items():
        try:
            path = resolve_relative_path(runtime_root, str(relative))
        except ValueError as error:
            errors.append(str(error))
            continue
        if not path.is_file():
            errors.append(f"runtime file is missing: {relative}")
        elif file_sha256(path) != str(expected):
            errors.append(f"runtime file hash mismatch: {relative}")

    ranking_policy = lock.get("candidate_ranking_policy")
    ranking_policy_hash = str(lock.get("candidate_ranking_policy_sha256", ""))
    if not ranking_policy_hash or canonical_json_sha256(
        ranking_policy
    ) != ranking_policy_hash:
        errors.append("candidate ranking policy hash mismatch")
    if isinstance(ranking_policy, Mapping) and ranking_policy.get(
        "retrieval_dependency_manifest_sha256"
    ) != canonical_json_sha256(dict(retrieval_dependencies)):
        errors.append("retrieval dependency manifest hash mismatch")

    holdout_inputs = lock.get("holdout_inputs", {})
    if not isinstance(holdout_inputs, Mapping):
        errors.append("holdout_inputs must be a mapping")
        holdout_inputs = {}
    holdout_artifacts = (
        "retrieval_receipt",
        "review_packets",
        "adjudicated_judgments",
        "adjudication_report",
        "v17_candidate_ranking",
        "v16_baseline_ranking",
    )
    for artifact in holdout_artifacts:
        relative = str(holdout_inputs.get(f"{artifact}_path", "")).strip()
        expected = str(holdout_inputs.get(f"{artifact}_sha256", "")).strip()
        if not relative or not expected:
            errors.append(f"holdout input binding is missing: {artifact}")
            continue
        try:
            path = resolve_relative_path(runtime_root, relative)
        except ValueError as error:
            errors.append(str(error))
            continue
        if not path.is_file():
            errors.append(f"holdout input is missing: {artifact}")
        elif file_sha256(path) != expected:
            errors.append(f"holdout input hash mismatch: {artifact}")

    execution = lock.get("execution", {})
    if isinstance(execution, Mapping):
        expected_judgments = str(
            holdout_inputs.get("adjudicated_judgments_path", "")
        )
        if execution.get("judgments") != expected_judgments:
            errors.append("execution judgments path differs from holdout binding")
    else:
        errors.append("execution plan must be a mapping")

    prompt_text = lock.get("verification_prompt_text")
    prompt_hash = str(lock.get("verification_prompt_sha256", ""))
    if not prompt_hash or canonical_json_sha256(prompt_text) != prompt_hash:
        errors.append("verification prompt hash mismatch")

    query_set_path = str(lock.get("query_set_path", ""))
    query_set_hash = str(lock.get("query_set_sha256", ""))
    if not query_set_path or not query_set_hash:
        errors.append("query set path or canonical hash is missing")
    else:
        try:
            frozen_path = resolve_relative_path(runtime_root, query_set_path)
            current_query_hash = query_fingerprint(read_jsonl(frozen_path))
        except (
            FileNotFoundError,
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ):
            errors.append("frozen query set cannot be fingerprinted")
        else:
            if current_query_hash != query_set_hash:
                errors.append("frozen query set canonical hash mismatch")

    model = lock.get("model", {})
    if not isinstance(model, Mapping):
        errors.append("model lock must be a mapping")
    else:
        model_relative = str(model.get("path", "")).strip()
        if not model_relative:
            errors.append("locked model path is missing")
            model_relative = "__missing_locked_model_path__"
        try:
            model_path = resolve_relative_path(
                runtime_root,
                model_relative,
            )
        except ValueError as error:
            errors.append(str(error))
            model_path = runtime_root / "__invalid_locked_model_path__"
        try:
            manifest_hash, manifest = directory_manifest_sha256(model_path)
        except FileNotFoundError:
            errors.append("locked model directory is missing")
        else:
            if manifest_hash != str(model.get("weight_manifest_sha256", "")):
                errors.append("model weight manifest hash mismatch")
            if int(model.get("weight_file_count", -1)) != manifest["file_count"]:
                errors.append("model weight file count mismatch")
            if int(model.get("weight_total_bytes", -1)) != manifest["total_bytes"]:
                errors.append("model weight total bytes mismatch")

    return {
        "valid": not errors,
        "errors": errors,
        "git_tree_clean": clean,
        "git_commit_sha": commit,
        "locked_file_count": len(locked_files),
        "retrieval_dependency_file_count": len(retrieval_dependencies),
        "governance_file_count": len(governance_files),
        "runtime_file_count": len(runtime_files),
        "holdout_input_count": len(holdout_artifacts),
    }


def create_one_shot_receipt(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically claim the one allowed run; existing receipts are immutable."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
