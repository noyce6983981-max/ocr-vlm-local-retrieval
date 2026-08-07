"""Tests for safe mixed-document preparation and batch merging."""

from __future__ import annotations

import zipfile
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pymupdf

from scripts.batch_ingest import (
    merge_unique,
    validate_declared_ai_percent,
    write_job_status,
)
from scripts.prepare_documents import (
    PROJECT_ROOT,
    collect_inputs,
    prepare_pdf,
    safe_extract_zip,
    stable_item_id,
)


class DocumentIngestTests(unittest.TestCase):
    def test_stable_item_id_depends_on_content(self) -> None:
        self.assertEqual(stable_item_id(b"same"), stable_item_id(b"same"))
        self.assertNotEqual(stable_item_id(b"first"), stable_item_id(b"second"))

    def test_safe_zip_extracts_supported_files(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "batch.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("folder/page.png", b"image")
                archive.writestr("notes.txt", b"skip")
            extracted = safe_extract_zip(
                archive_path,
                root / "extracted",
                max_files=10,
                max_uncompressed_bytes=1024,
            )
            self.assertEqual(len(extracted), 1)
            self.assertEqual(extracted[0].name, "page.png")

    def test_safe_zip_blocks_path_traversal(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "unsafe.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("../escape.png", b"image")
            with self.assertRaises(ValueError):
                safe_extract_zip(
                    archive_path,
                    root / "extracted",
                    max_files=10,
                    max_uncompressed_bytes=1024,
                )

    def test_directory_input_is_discovered_recursively(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "batch/nested"
            nested.mkdir(parents=True)
            first = root / "batch/page.jpg"
            second = nested / "report.pdf"
            ignored = nested / "notes.txt"
            first.write_bytes(b"jpg")
            second.write_bytes(b"pdf")
            ignored.write_text("ignore", encoding="utf-8")

            collected = collect_inputs(
                [root / "batch"],
                root / "extracted",
                max_files=10,
                max_uncompressed_bytes=1024,
            )

            self.assertEqual(collected, [second.resolve(), first.resolve()])

    def test_pdf_is_rendered_into_page_units(self) -> None:
        outputs_dir = PROJECT_ROOT / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=outputs_dir) as directory:
            root = Path(directory)
            pdf_path = root / "sample.pdf"
            document = pymupdf.open()
            for page_number in range(2):
                page = document.new_page()
                page.insert_text((72, 72), f"page {page_number + 1}")
            document.save(pdf_path)
            document.close()

            rows = prepare_pdf(
                pdf_path,
                pdf_path,
                root / "pages",
                "source_test",
                dpi=100,
            )
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["page_number"], 1)
            self.assertEqual(rows[1]["page_number"], 2)
            self.assertTrue(
                (PROJECT_ROOT / rows[0]["source_path"]).is_file()
            )

    def test_merge_unique_returns_only_new_rows(self) -> None:
        combined, added = merge_unique(
            [{"item_id": "a"}],
            [{"item_id": "a"}, {"item_id": "b"}],
            "item_id",
        )
        self.assertEqual(combined, [{"item_id": "a"}, {"item_id": "b"}])
        self.assertEqual(added, [{"item_id": "b"}])

    def test_new_batch_rejects_more_than_five_percent_ai(self) -> None:
        self.assertEqual(validate_declared_ai_percent(5.0), 5.0)
        with self.assertRaisesRegex(ValueError, "below 5 percent"):
            validate_declared_ai_percent(5.01)

    def test_job_status_is_written_atomically(self) -> None:
        with TemporaryDirectory() as directory:
            job_dir = Path(directory)
            write_job_status(
                job_dir,
                phase="ocr",
                new_pages=12,
                total_pages=20,
            )
            payload = (job_dir / "status.json").read_text(encoding="utf-8")
            self.assertIn('"phase": "ocr"', payload)
            self.assertIn('"new_pages": 12', payload)


if __name__ == "__main__":
    unittest.main()
