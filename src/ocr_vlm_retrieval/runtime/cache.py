"""Crash-safe JSON cache I/O with bounded retention."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_json_object(path: Path) -> dict[str, Any] | None:
    """Return a JSON object or treat a partial/corrupt cache as a miss."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def write_json_atomic(path: Path, payload: Any) -> None:
    """Replace a cache only after a complete JSON file reaches disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def prune_json_cache(
    directory: Path,
    *,
    protected: set[Path] | None = None,
    max_files: int,
    max_bytes: int,
) -> dict[str, int]:
    """Bound generated JSON files by count and bytes, newest first."""

    if max_files < 0 or max_bytes < 0:
        raise ValueError("cache limits must not be negative")
    if not directory.is_dir():
        return {"deleted_files": 0, "deleted_bytes": 0}
    protected_paths = {
        path.resolve() for path in (protected or set()) if path.exists()
    }
    candidates: list[tuple[Path, int, int]] = []
    kept_files = 0
    kept_bytes = 0
    for path in directory.glob("*.json"):
        try:
            stat = path.stat()
        except OSError:
            continue
        if path.resolve() in protected_paths:
            kept_files += 1
            kept_bytes += stat.st_size
            continue
        candidates.append((path, stat.st_size, stat.st_mtime_ns))
    candidates.sort(key=lambda row: row[2], reverse=True)
    deleted_files = 0
    deleted_bytes = 0
    for path, size, _ in candidates:
        if kept_files < max_files and kept_bytes + size <= max_bytes:
            kept_files += 1
            kept_bytes += size
            continue
        try:
            path.unlink()
        except OSError:
            continue
        deleted_files += 1
        deleted_bytes += size
    return {"deleted_files": deleted_files, "deleted_bytes": deleted_bytes}
