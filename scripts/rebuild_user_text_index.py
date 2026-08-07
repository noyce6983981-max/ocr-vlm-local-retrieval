"""Rebuild the user BGE-M3 index, including optional OCR overrides."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.batch_ingest import commit_index_directories  # noqa: E402


PYTHON = PROJECT_ROOT / ".venv/Scripts/python.exe"
LIBRARY_DIR = PROJECT_ROOT / "outputs/user_library"


def run(command: list[str], label: str, timeout: int) -> float:
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"{label} failed:\n"
            + (completed.stderr or completed.stdout)[-3000:]
        )
    return round(time.perf_counter() - started, 3)


def main() -> None:
    job_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_dir = (
        PROJECT_ROOT / "outputs/batch_jobs" / f"{job_id}_text_rebuild"
    )
    rebuild_dir = job_dir / "rebuild"
    corpus_path = rebuild_dir / "corpus.jsonl"
    overrides_dir = LIBRARY_DIR / "ocr/overrides"
    corpus_command = [
        str(PYTHON),
        str(PROJECT_ROOT / "scripts/build_text_corpus.py"),
        "--manifest",
        str(LIBRARY_DIR / "manifest.jsonl"),
        "--ocr-dir",
        str(LIBRARY_DIR / "ocr/json"),
        "--output",
        str(corpus_path),
        "--summary",
        str(rebuild_dir / "corpus_summary.json"),
    ]
    if overrides_dir.is_dir():
        corpus_command.extend(
            ["--ocr-overrides-dir", str(overrides_dir)]
        )
    corpus_seconds = run(
        corpus_command, "OCR corpus rebuild", timeout=300
    )
    index_seconds = run(
        [
            str(PYTHON),
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
        "BGE-M3 index rebuild",
        timeout=3600,
    )
    commit_index_directories(
        [(rebuild_dir / "text_index", LIBRARY_DIR / "text_index")],
        job_dir / "index_backups",
    )
    payload = {
        "status": "success",
        "job_id": job_id,
        "ocr_override_count": (
            len(list(overrides_dir.glob("*.json")))
            if overrides_dir.is_dir()
            else 0
        ),
        "corpus_seconds": corpus_seconds,
        "text_index_seconds": index_seconds,
    }
    (job_dir / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
