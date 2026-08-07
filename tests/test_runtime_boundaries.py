from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from ocr_vlm_retrieval.runtime.backends import SubprocessBranchBackend


def test_subprocess_backend_runs_selected_python_environment() -> None:
    backend = SubprocessBranchBackend(PROJECT_ROOT, timeout_seconds=10)
    backend.run([sys.executable, "-c", "print('ok')"], label="text")


def test_subprocess_backend_exposes_concise_failure_tail() -> None:
    backend = SubprocessBranchBackend(
        PROJECT_ROOT,
        timeout_seconds=10,
        error_tail_characters=12,
    )
    with pytest.raises(RuntimeError, match="visual branch failed") as error:
        backend.run(
            [
                sys.executable,
                "-c",
                "import sys; print('prefix-unique-tail', file=sys.stderr); sys.exit(2)",
            ],
            label="visual",
        )
    assert "unique-tail" in str(error.value)
    assert "prefix-" not in str(error.value)


def test_subprocess_backend_rejects_empty_command() -> None:
    backend = SubprocessBranchBackend(PROJECT_ROOT)
    with pytest.raises(ValueError, match="must not be empty"):
        backend.run([], label="reranker")
