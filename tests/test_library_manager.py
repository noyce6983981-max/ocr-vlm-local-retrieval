from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.library_manager import (
    create_library,
    library_by_id,
    load_registry,
    rename_library,
    resolve_library_dir,
    set_active_library,
)
from scripts.live_search import combined_manifest, library_revision


class LibraryManagerTests(unittest.TestCase):
    def test_legacy_library_is_registered_without_moving_data(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = root / "outputs/user_library"
            legacy.mkdir(parents=True)
            (legacy / "manifest.jsonl").write_text(
                '{"item_id":"legacy"}\n', encoding="utf-8"
            )
            registry_path = root / "outputs/library_registry.json"
            registry = load_registry(registry_path, root)
            library = library_by_id(registry, "public_research_200")
            self.assertEqual(
                resolve_library_dir(library, root), legacy.resolve()
            )

    def test_created_libraries_are_strictly_isolated(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            registry_path = root / "outputs/library_registry.json"
            load_registry(registry_path, root)
            first = create_library(
                "课程资料", "study", registry_path, root
            )
            second = create_library(
                "私人资料", "private", registry_path, root
            )
            first_dir = resolve_library_dir(first, root)
            second_dir = resolve_library_dir(second, root)
            (first_dir / "manifest.jsonl").write_text(
                json.dumps({"item_id": "course"}) + "\n",
                encoding="utf-8",
            )
            (second_dir / "manifest.jsonl").write_text(
                json.dumps({"item_id": "private"}) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(
                [row["item_id"] for row in combined_manifest(first_dir)],
                ["course"],
            )
            self.assertEqual(
                [row["item_id"] for row in combined_manifest(second_dir)],
                ["private"],
            )
            self.assertNotEqual(
                library_revision(first_dir),
                library_revision(second_dir),
            )

    def test_search_manifest_excludes_disabled_pages(self) -> None:
        with TemporaryDirectory() as directory:
            library_dir = Path(directory)
            (library_dir / "manifest.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"item_id": "active"}),
                        json.dumps(
                            {
                                "item_id": "disabled",
                                "search_enabled": False,
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            self.assertEqual(
                [row["item_id"] for row in combined_manifest(library_dir)],
                ["active"],
            )

    def test_active_library_must_exist(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            registry_path = root / "outputs/library_registry.json"
            load_registry(registry_path, root)
            with self.assertRaisesRegex(ValueError, "Unknown library"):
                set_active_library("missing", registry_path, root)

    def test_library_can_be_renamed_without_changing_its_path(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            registry_path = root / "outputs/library_registry.json"
            registry = load_registry(registry_path, root)
            library_id = registry["active_library_id"]
            original_path = library_by_id(registry, library_id)["path"]

            rename_library(
                library_id,
                "公开研究库",
                registry_path,
                root,
            )

            updated = load_registry(registry_path, root)
            library = library_by_id(updated, library_id)
            self.assertEqual(library["name"], "公开研究库")
            self.assertEqual(library["path"], original_path)

    def test_registry_rejects_path_escape(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                resolve_library_dir(
                    {
                        "path": "../outside",
                    },
                    root,
                )


if __name__ == "__main__":
    unittest.main()
