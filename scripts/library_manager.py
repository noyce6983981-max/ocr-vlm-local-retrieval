"""Manage isolated local document libraries and their registry."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = PROJECT_ROOT / "outputs/library_registry.json"
LEGACY_LIBRARY_PATH = Path("outputs/user_library")
LIBRARIES_ROOT = Path("outputs/libraries")
VALID_PRIVACY_LEVELS = {"public", "study", "private"}


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def resolve_library_dir(
    library: dict[str, Any],
    project_root: Path = PROJECT_ROOT,
) -> Path:
    """Resolve a registry path and reject paths outside approved roots."""
    path = (project_root / library["path"]).resolve()
    legacy = (project_root / LEGACY_LIBRARY_PATH).resolve()
    managed_root = (project_root / LIBRARIES_ROOT).resolve()
    if path != legacy and not _inside(path, managed_root):
        raise ValueError(f"Unsafe library path: {path}")
    return path


def initial_registry(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    legacy_dir = project_root / LEGACY_LIBRARY_PATH
    if legacy_dir.is_dir():
        libraries = [
            {
                "id": "public_research_200",
                "name": "公开研究库",
                "privacy": "public",
                "path": LEGACY_LIBRARY_PATH.as_posix(),
                "created_at": _timestamp(),
            }
        ]
    else:
        default_path = LIBRARIES_ROOT / "general"
        (project_root / default_path).mkdir(parents=True, exist_ok=True)
        libraries = [
            {
                "id": "general",
                "name": "通用资料库",
                "privacy": "study",
                "path": default_path.as_posix(),
                "created_at": _timestamp(),
            }
        ]
    return {
        "version": 1,
        "active_library_id": libraries[0]["id"],
        "libraries": libraries,
    }


def validate_registry(
    registry: dict[str, Any],
    project_root: Path = PROJECT_ROOT,
) -> None:
    libraries = registry.get("libraries", [])
    if not libraries:
        raise ValueError("Library registry cannot be empty.")
    ids = [row.get("id", "") for row in libraries]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("Library IDs must be non-empty and unique.")
    if registry.get("active_library_id") not in set(ids):
        raise ValueError("Active library is absent from registry.")
    for row in libraries:
        if row.get("privacy") not in VALID_PRIVACY_LEVELS:
            raise ValueError(f"Invalid privacy level: {row.get('privacy')}")
        resolve_library_dir(row, project_root)


def write_registry(path: Path, registry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_registry(
    path: Path = REGISTRY_PATH,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    if not path.is_file():
        registry = initial_registry(project_root)
        write_registry(path, registry)
    else:
        registry = json.loads(path.read_text(encoding="utf-8"))
    validate_registry(registry, project_root)
    return registry


def create_library(
    name: str,
    privacy: str,
    registry_path: Path = REGISTRY_PATH,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    normalized_name = " ".join(name.split())
    if not normalized_name:
        raise ValueError("资料库名称不能为空。")
    if len(normalized_name) > 40:
        raise ValueError("资料库名称不能超过40个字符。")
    if privacy not in VALID_PRIVACY_LEVELS:
        raise ValueError("不支持的资料库隐私级别。")

    registry = load_registry(registry_path, project_root)
    if any(row["name"] == normalized_name for row in registry["libraries"]):
        raise ValueError("已经存在同名资料库。")
    library_id = f"library_{uuid.uuid4().hex[:12]}"
    relative_path = LIBRARIES_ROOT / library_id
    library_dir = project_root / relative_path
    library_dir.mkdir(parents=True, exist_ok=False)
    library = {
        "id": library_id,
        "name": normalized_name,
        "privacy": privacy,
        "path": relative_path.as_posix(),
        "created_at": _timestamp(),
    }
    registry["libraries"].append(library)
    registry["active_library_id"] = library_id
    write_registry(registry_path, registry)
    return library


def set_active_library(
    library_id: str,
    registry_path: Path = REGISTRY_PATH,
    project_root: Path = PROJECT_ROOT,
) -> None:
    registry = load_registry(registry_path, project_root)
    if library_id not in {row["id"] for row in registry["libraries"]}:
        raise ValueError(f"Unknown library: {library_id}")
    registry["active_library_id"] = library_id
    write_registry(registry_path, registry)


def rename_library(
    library_id: str,
    name: str,
    registry_path: Path = REGISTRY_PATH,
    project_root: Path = PROJECT_ROOT,
) -> None:
    normalized_name = " ".join(name.split())
    if not normalized_name:
        raise ValueError("资料库名称不能为空。")
    if len(normalized_name) > 40:
        raise ValueError("资料库名称不能超过40个字符。")
    registry = load_registry(registry_path, project_root)
    target = library_by_id(registry, library_id)
    if any(
        row["id"] != library_id and row["name"] == normalized_name
        for row in registry["libraries"]
    ):
        raise ValueError("已经存在同名资料库。")
    target["name"] = normalized_name
    write_registry(registry_path, registry)


def library_by_id(
    registry: dict[str, Any], library_id: str
) -> dict[str, Any]:
    return next(
        row for row in registry["libraries"] if row["id"] == library_id
    )


def manifest_count(library_dir: Path) -> int:
    path = library_dir / "manifest.jsonl"
    if not path.is_file():
        return 0
    return sum(1 for line in path.open(encoding="utf-8") if line.strip())
