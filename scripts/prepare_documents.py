"""Safely expand mixed files and convert them into image retrieval units."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
import subprocess
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import pymupdf
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
DOCUMENT_SUFFIXES = {".pdf", ".docx", ".pptx"}
SUPPORTED_SUFFIXES = IMAGE_SUFFIXES | DOCUMENT_SUFFIXES | {".zip"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-pages", type=int, default=1000)
    parser.add_argument("--max-files", type=int, default=1000)
    parser.add_argument("--max-uncompressed-mb", type=int, default=2048)
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def stable_item_id(image_bytes: bytes) -> str:
    return f"user_{sha256_bytes(image_bytes)[:12]}"


def is_zip_symlink(info: zipfile.ZipInfo) -> bool:
    mode = info.external_attr >> 16
    return stat.S_ISLNK(mode)


def safe_extract_zip(
    archive_path: Path,
    destination: Path,
    max_files: int,
    max_uncompressed_bytes: int,
) -> list[Path]:
    """Extract supported files while blocking traversal, links, and zip bombs."""
    extracted: list[Path] = []
    destination.mkdir(parents=True, exist_ok=True)
    destination_resolved = destination.resolve()

    with zipfile.ZipFile(archive_path) as archive:
        members = [info for info in archive.infolist() if not info.is_dir()]
        if len(members) > max_files:
            raise ValueError(
                f"ZIP contains {len(members)} files; limit is {max_files}."
            )
        total_size = sum(info.file_size for info in members)
        if total_size > max_uncompressed_bytes:
            raise ValueError("ZIP uncompressed size exceeds the safety limit.")

        for info in members:
            if info.flag_bits & 0x1:
                raise ValueError("Encrypted ZIP files are not supported.")
            if is_zip_symlink(info):
                raise ValueError(f"ZIP symbolic link is not allowed: {info.filename}")
            normalized = PurePosixPath(info.filename.replace("\\", "/"))
            if normalized.is_absolute() or ".." in normalized.parts:
                raise ValueError(f"Unsafe ZIP path: {info.filename}")
            if normalized.suffix.lower() not in SUPPORTED_SUFFIXES - {".zip"}:
                continue

            relative = Path(*normalized.parts)
            target = (destination / relative).resolve()
            if (
                target != destination_resolved
                and destination_resolved not in target.parents
            ):
                raise ValueError(f"ZIP path escaped destination: {info.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            extracted.append(target)
    return extracted


def convert_office_to_pdf(source: Path, output_pdf: Path) -> None:
    command = [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(PROJECT_ROOT / "scripts/convert_office_to_pdf.ps1"),
        "-InputPath",
        str(source),
        "-OutputPath",
        str(output_pdf),
    ]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout)[-1600:]
        raise RuntimeError(f"Office conversion failed for {source.name}:\n{detail}")


def display_name(source: Path, page_number: int, page_count: int) -> str:
    if page_count == 1:
        return source.stem
    return f"{source.stem} 第{page_number}页"


def category_for(source: Path) -> str:
    if source.suffix.lower() == ".pptx":
        return "slide"
    if source.suffix.lower() in {".docx", ".pdf"}:
        return "document_page"
    return "personal_image"


def save_page(
    image_bytes: bytes,
    suffix: str,
    pages_dir: Path,
    source: Path,
    page_number: int,
    page_count: int,
    source_group_id: str,
    source_format: str,
) -> dict[str, Any]:
    item_id = stable_item_id(image_bytes)
    normalized_suffix = ".jpg" if suffix.lower() == ".jpeg" else suffix.lower()
    pages_dir.mkdir(parents=True, exist_ok=True)
    target = pages_dir / f"{item_id}{normalized_suffix}"
    if not target.is_file():
        target.write_bytes(image_bytes)
    return {
        "item_id": item_id,
        "source_path": target.relative_to(PROJECT_ROOT).as_posix(),
        "display_name_zh": display_name(source, page_number, page_count),
        "category": category_for(source),
        "has_text_expected": source_format != "image_no_text",
        "review_status": "待人工审核",
        "source": "batch_upload",
        "source_file_name": source.name,
        "source_format": source_format,
        "source_group_id": source_group_id,
        "page_number": page_number,
        "page_count": page_count,
    }


def prepare_image(
    source: Path,
    pages_dir: Path,
    source_group_id: str,
) -> list[dict[str, Any]]:
    with Image.open(source) as opened:
        opened.verify()
    payload = source.read_bytes()
    return [
        save_page(
            payload,
            source.suffix,
            pages_dir,
            source,
            1,
            1,
            source_group_id,
            "image",
        )
    ]


def prepare_pdf(
    source_pdf: Path,
    logical_source: Path,
    pages_dir: Path,
    source_group_id: str,
    dpi: int,
) -> list[dict[str, Any]]:
    scale = dpi / 72.0
    matrix = pymupdf.Matrix(scale, scale)
    rows: list[dict[str, Any]] = []
    with pymupdf.open(source_pdf) as document:
        page_count = len(document)
        for page_index, page in enumerate(document):
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            payload = pixmap.tobytes("png")
            rows.append(
                save_page(
                    payload,
                    ".png",
                    pages_dir,
                    logical_source,
                    page_index + 1,
                    page_count,
                    source_group_id,
                    logical_source.suffix.lower().lstrip("."),
                )
            )
    return rows


def collect_inputs(
    inputs: list[Path],
    extraction_dir: Path,
    max_files: int,
    max_uncompressed_bytes: int,
) -> list[Path]:
    collected: list[Path] = []
    archive_index = 0

    def collect_file(path: Path) -> None:
        nonlocal archive_index
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            return
        if suffix == ".zip":
            collected.extend(
                safe_extract_zip(
                    path,
                    extraction_dir / f"archive_{archive_index:04d}",
                    max_files=max_files,
                    max_uncompressed_bytes=max_uncompressed_bytes,
                )
            )
            archive_index += 1
        else:
            collected.append(path)
        if len(collected) > max_files:
            raise ValueError(
                f"Prepared input count {len(collected)} exceeds {max_files}."
            )

    for raw_path in inputs:
        path = project_path(raw_path).resolve()
        if path.is_dir():
            for candidate in sorted(
                path.rglob("*"), key=lambda value: value.as_posix().lower()
            ):
                if candidate.is_symlink():
                    raise ValueError(
                        f"Directory symbolic link is not allowed: {candidate}"
                    )
                if candidate.is_file():
                    collect_file(candidate.resolve())
            continue
        if not path.is_file():
            raise FileNotFoundError(f"Input not found: {path}")
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported input type: {path.name}")
        collect_file(path)
    if not collected:
        raise ValueError("No supported files were found in the inputs.")
    if len(collected) > max_files:
        raise ValueError(
            f"Prepared input count {len(collected)} exceeds {max_files}."
        )
    return collected


def main() -> None:
    args = parse_args()
    if args.max_pages <= 0 or args.max_files <= 0:
        raise ValueError("Page and file limits must be positive.")
    if not 72 <= args.dpi <= 300:
        raise ValueError("--dpi must be between 72 and 300.")

    output_dir = project_path(args.output).resolve()
    pages_dir = output_dir / "pages"
    extraction_dir = output_dir / "extracted"
    converted_dir = output_dir / "converted"
    sources_dir = output_dir / "sources"
    for directory in (pages_dir, converted_dir, sources_dir):
        directory.mkdir(parents=True, exist_ok=True)

    files = collect_inputs(
        args.inputs,
        extraction_dir,
        max_files=args.max_files,
        max_uncompressed_bytes=args.max_uncompressed_mb * 1024 * 1024,
    )
    rows: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []

    for source in files:
        source_payload = source.read_bytes()
        source_hash = sha256_bytes(source_payload)
        source_group_id = f"source_{source_hash[:12]}"
        source_suffix = source.suffix.lower()
        stored_source = sources_dir / f"{source_group_id}{source_suffix}"
        if not stored_source.is_file():
            stored_source.write_bytes(source_payload)
        source_records.append(
            {
                "source_group_id": source_group_id,
                "original_name": source.name,
                "source_format": source_suffix.lstrip("."),
                "stored_path": stored_source.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": source_hash,
            }
        )

        if source_suffix in IMAGE_SUFFIXES:
            prepared = prepare_image(source, pages_dir, source_group_id)
        elif source_suffix == ".pdf":
            prepared = prepare_pdf(
                source,
                source,
                pages_dir,
                source_group_id,
                args.dpi,
            )
        else:
            converted_pdf = converted_dir / f"{source_group_id}.pdf"
            convert_office_to_pdf(source, converted_pdf)
            prepared = prepare_pdf(
                converted_pdf,
                source,
                pages_dir,
                source_group_id,
                args.dpi,
            )
        rows.extend(prepared)
        if len(rows) > args.max_pages:
            raise ValueError(
                f"Converted page count exceeds limit {args.max_pages}."
            )

    unique_rows: list[dict[str, Any]] = []
    seen_items: set[str] = set()
    for row in rows:
        if row["item_id"] in seen_items:
            continue
        seen_items.add(row["item_id"])
        unique_rows.append(row)

    manifest_path = output_dir / "prepared_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in unique_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    sources_path = output_dir / "prepared_sources.jsonl"
    with sources_path.open("w", encoding="utf-8") as handle:
        seen_sources: set[str] = set()
        for row in source_records:
            if row["source_group_id"] in seen_sources:
                continue
            seen_sources.add(row["source_group_id"])
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "input_count": len(args.inputs),
        "expanded_file_count": len(files),
        "source_count": len({row["source_group_id"] for row in source_records}),
        "page_count_before_deduplication": len(rows),
        "page_count": len(unique_rows),
        "duplicate_page_count": len(rows) - len(unique_rows),
        "manifest_path": manifest_path.relative_to(PROJECT_ROOT).as_posix(),
        "sources_path": sources_path.relative_to(PROJECT_ROOT).as_posix(),
        "dpi": args.dpi,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
