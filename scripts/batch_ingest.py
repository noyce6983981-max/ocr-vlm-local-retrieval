"""Batch-ingest ZIP, PDF, DOCX, PPTX, and image files into the user library."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PADDLE_PYTHON = PROJECT_ROOT / ".venv-paddle/Scripts/python.exe"
TEXT_PYTHON = PROJECT_ROOT / ".venv/Scripts/python.exe"
VISUAL_PYTHON = PROJECT_ROOT / ".venv-vl/Scripts/python.exe"
DEFAULT_LIBRARY_DIR = PROJECT_ROOT / "outputs/user_library"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--max-pages", type=int, default=1000)
    parser.add_argument("--max-files", type=int, default=1000)
    parser.add_argument("--max-uncompressed-mb", type=int, default=2048)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument(
        "--library-dir",
        type=Path,
        default=DEFAULT_LIBRARY_DIR,
        help="Isolated output directory for this document library.",
    )
    parser.add_argument(
        "--declared-ai-percent",
        type=float,
        default=0.0,
        help="User-declared percentage of AI-generated source content.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_job_status(
    job_dir: Path,
    *,
    phase: str,
    status: str = "running",
    **details: Any,
) -> None:
    path = job_dir / "status.json"
    previous: dict[str, Any] = {}
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
    payload = {
        **previous,
        "status": status,
        "phase": phase,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        **details,
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def merge_unique(
    existing: list[dict[str, Any]],
    incoming: list[dict[str, Any]],
    key: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seen = {row[key] for row in existing}
    added: list[dict[str, Any]] = []
    for row in incoming:
        value = row[key]
        if value in seen:
            continue
        seen.add(value)
        added.append(row)
    return existing + added, added


def validate_declared_ai_percent(value: float) -> float:
    if not 0.0 <= value <= 5.0:
        raise ValueError(
            "New data batches must keep declared AI-generated content "
            "at or below 5 percent."
        )
    return round(value, 2)


def run_checked(
    command: list[str],
    label: str,
    timeout: int,
) -> float:
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout)[-3000:]
        raise RuntimeError(f"{label} failed:\n{detail}")
    return round(time.perf_counter() - started, 3)


def commit_index_directories(
    replacements: list[tuple[Path, Path]],
    backup_root: Path,
) -> None:
    """Replace all index directories and roll back if any move fails."""
    backups: list[tuple[Path, Path]] = []
    installed: list[Path] = []
    backup_root.mkdir(parents=True, exist_ok=True)
    try:
        for _, target in replacements:
            if target.exists():
                backup = backup_root / target.name
                shutil.move(str(target), str(backup))
                backups.append((backup, target))
        for source, target in replacements:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(target))
            installed.append(target)
    except Exception:
        for target in installed:
            if target.exists():
                shutil.rmtree(target)
        for backup, target in backups:
            if backup.exists():
                shutil.move(str(backup), str(target))
        raise


def main() -> None:
    args = parse_args()
    declared_ai_percent = validate_declared_ai_percent(
        args.declared_ai_percent
    )
    library_dir = (
        args.library_dir
        if args.library_dir.is_absolute()
        else PROJECT_ROOT / args.library_dir
    ).resolve()
    allowed_legacy = (PROJECT_ROOT / "outputs/user_library").resolve()
    allowed_root = (PROJECT_ROOT / "outputs/libraries").resolve()
    if (
        library_dir != allowed_legacy
        and allowed_root not in library_dir.parents
    ):
        raise ValueError(f"Unsafe library directory: {library_dir}")
    manifest_path = library_dir / "manifest.jsonl"
    source_manifest_path = library_dir / "sources.jsonl"
    started = time.perf_counter()
    job_id = (
        datetime.now().strftime("%Y%m%d_%H%M%S")
        + "_"
        + uuid.uuid4().hex[:6]
    )
    job_dir = PROJECT_ROOT / "outputs/batch_jobs" / job_id
    prepared_dir = job_dir / "prepared"
    rebuild_dir = job_dir / "rebuild"
    job_dir.mkdir(parents=True, exist_ok=False)
    write_job_status(
        job_dir,
        phase="preparing_documents",
        job_id=job_id,
        library_dir=library_dir.relative_to(PROJECT_ROOT).as_posix(),
        input_count=len(args.inputs),
    )

    prepare_seconds = run_checked(
        [
            str(TEXT_PYTHON),
            str(PROJECT_ROOT / "scripts/prepare_documents.py"),
            *[str(path) for path in args.inputs],
            "--output",
            str(prepared_dir),
            "--max-pages",
            str(args.max_pages),
            "--max-files",
            str(args.max_files),
            "--max-uncompressed-mb",
            str(args.max_uncompressed_mb),
            "--dpi",
            str(args.dpi),
        ],
        "document preparation",
        timeout=900,
    )
    prepared_rows = read_jsonl(prepared_dir / "prepared_manifest.jsonl")
    prepared_sources = read_jsonl(prepared_dir / "prepared_sources.jsonl")
    existing_rows = read_jsonl(manifest_path)
    existing_sources = read_jsonl(source_manifest_path)
    candidate_rows, new_rows = merge_unique(
        existing_rows, prepared_rows, "item_id"
    )
    candidate_sources, new_sources = merge_unique(
        existing_sources, prepared_sources, "source_group_id"
    )
    write_job_status(
        job_dir,
        phase="deduplicating",
        prepared_pages=len(prepared_rows),
        new_pages=len(new_rows),
        total_pages=len(candidate_rows),
    )
    for row in new_sources:
        row["declared_ai_generated_percent"] = round(
            declared_ai_percent, 2
        )
        row["provenance_review_status"] = "user_declared"

    if not new_rows:
        write_job_status(
            job_dir,
            phase="completed",
            status="duplicate",
            prepared_pages=len(prepared_rows),
            new_pages=0,
            total_pages=len(existing_rows),
        )
        print(
            json.dumps(
                {
                    "status": "duplicate",
                    "job_id": job_id,
                    "prepared_pages": len(prepared_rows),
                    "new_pages": 0,
                    "total_pages": len(existing_rows),
                    "timings": {"prepare_seconds": prepare_seconds},
                },
                ensure_ascii=False,
            )
        )
        return

    images_dir = library_dir / "images"
    sources_dir = library_dir / "sources"
    images_dir.mkdir(parents=True, exist_ok=True)
    sources_dir.mkdir(parents=True, exist_ok=True)
    new_ids = {row["item_id"] for row in new_rows}
    for row in candidate_rows:
        if row["item_id"] not in new_ids:
            continue
        prepared_image = PROJECT_ROOT / row["source_path"]
        final_image = images_dir / prepared_image.name
        shutil.copy2(prepared_image, final_image)
        row["source_path"] = final_image.relative_to(PROJECT_ROOT).as_posix()

    new_source_ids = {row["source_group_id"] for row in new_sources}
    for row in candidate_sources:
        if row["source_group_id"] not in new_source_ids:
            continue
        prepared_source = PROJECT_ROOT / row["stored_path"]
        final_source = sources_dir / prepared_source.name
        shutil.copy2(prepared_source, final_source)
        row["stored_path"] = final_source.relative_to(PROJECT_ROOT).as_posix()

    candidate_manifest = job_dir / "candidate_manifest.jsonl"
    candidate_sources_path = job_dir / "candidate_sources.jsonl"
    write_jsonl(candidate_manifest, candidate_rows)
    write_jsonl(candidate_sources_path, candidate_sources)

    write_job_status(
        job_dir,
        phase="ocr",
        cached_pages=len(existing_rows),
        new_pages=len(new_rows),
        total_pages=len(candidate_rows),
    )
    ocr_seconds = run_checked(
        [
            str(PADDLE_PYTHON),
            str(PROJECT_ROOT / "scripts/batch_ocr.py"),
            "--manifest",
            str(candidate_manifest),
            "--output",
            str(library_dir / "ocr"),
            "--device",
            "gpu:0",
            "--progress-json",
            str(job_dir / "ocr_progress.json"),
        ],
        "batch OCR",
        timeout=7200,
    )
    write_job_status(
        job_dir,
        phase="building_text_corpus",
        total_pages=len(candidate_rows),
    )
    corpus_path = rebuild_dir / "corpus.jsonl"
    corpus_command = [
        str(TEXT_PYTHON),
        str(PROJECT_ROOT / "scripts/build_text_corpus.py"),
        "--manifest",
        str(candidate_manifest),
        "--ocr-dir",
        str(library_dir / "ocr/json"),
        "--output",
        str(corpus_path),
        "--summary",
        str(rebuild_dir / "corpus_summary.json"),
    ]
    overrides_dir = library_dir / "ocr/overrides"
    if overrides_dir.is_dir():
        corpus_command.extend(
            ["--ocr-overrides-dir", str(overrides_dir)]
        )
    corpus_seconds = run_checked(
        corpus_command,
        "OCR corpus construction",
        timeout=300,
    )
    write_job_status(
        job_dir,
        phase="building_text_index",
        total_pages=len(candidate_rows),
    )
    text_seconds = run_checked(
        [
            str(TEXT_PYTHON),
            str(PROJECT_ROOT / "scripts/build_text_index.py"),
            "--corpus",
            str(corpus_path),
            "--output",
            str(rebuild_dir / "text_index"),
            "--device",
            "cuda:0",
            "--batch-size",
            "8",
        ],
        "batch text indexing",
        timeout=3600,
    )
    write_job_status(
        job_dir,
        phase="building_sparse_index",
        total_pages=len(candidate_rows),
    )
    bm25_seconds = run_checked(
        [
            str(TEXT_PYTHON),
            str(PROJECT_ROOT / "scripts/build_bm25_index.py"),
            "--corpus",
            str(corpus_path),
            "--output",
            str(rebuild_dir / "bm25_index"),
        ],
        "batch BM25 indexing",
        timeout=600,
    )
    write_job_status(
        job_dir,
        phase="building_visual_index",
        total_pages=len(candidate_rows),
    )
    visual_seconds = run_checked(
        [
            str(VISUAL_PYTHON),
            str(PROJECT_ROOT / "scripts/build_visual_index.py"),
            "--manifest",
            str(candidate_manifest),
            "--output",
            str(rebuild_dir / "visual_index"),
        ],
        "batch visual indexing",
        timeout=7200,
    )

    write_job_status(
        job_dir,
        phase="committing_indexes",
        total_pages=len(candidate_rows),
    )
    commit_index_directories(
        [
            (
                rebuild_dir / "text_index",
                library_dir / "text_index",
            ),
            (
                rebuild_dir / "visual_index",
                library_dir / "visual_index",
            ),
            (
                rebuild_dir / "bm25_index",
                library_dir / "bm25_index",
            ),
        ],
        job_dir / "index_backups",
    )
    manifest_temp = library_dir / "manifest.jsonl.tmp"
    sources_temp = library_dir / "sources.jsonl.tmp"
    write_jsonl(manifest_temp, candidate_rows)
    write_jsonl(sources_temp, candidate_sources)
    os.replace(manifest_temp, manifest_path)
    os.replace(sources_temp, source_manifest_path)

    payload = {
        "status": "success",
        "job_id": job_id,
        "library_dir": library_dir.relative_to(PROJECT_ROOT).as_posix(),
        "input_count": len(args.inputs),
        "prepared_pages": len(prepared_rows),
        "new_pages": len(new_rows),
        "duplicate_pages": len(prepared_rows) - len(new_rows),
        "total_pages": len(candidate_rows),
        "new_sources": len(new_sources),
        "declared_ai_generated_percent": round(
            declared_ai_percent, 2
        ),
        "pending_review": len(new_rows),
        "timings": {
            "prepare_seconds": prepare_seconds,
            "ocr_seconds": ocr_seconds,
            "corpus_seconds": corpus_seconds,
            "text_index_seconds": text_seconds,
            "bm25_index_seconds": bm25_seconds,
            "visual_index_seconds": visual_seconds,
            "total_seconds": round(time.perf_counter() - started, 3),
        },
    }
    (job_dir / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_job_status(
        job_dir,
        phase="completed",
        status="success",
        new_pages=len(new_rows),
        total_pages=len(candidate_rows),
        total_seconds=payload["timings"]["total_seconds"],
    )
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
