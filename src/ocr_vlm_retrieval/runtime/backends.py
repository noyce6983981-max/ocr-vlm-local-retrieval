"""Backend boundary for retrieval branches running in different environments."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class BranchBackend(Protocol):
    """Execute one model branch without exposing its environment to orchestration."""

    def run(self, command: list[str], *, label: str) -> None:
        """Run a branch or raise a concise runtime error."""


@dataclass(frozen=True)
class SubprocessBranchBackend:
    """Run text, visual, or reranker commands in their selected Python runtime."""

    project_root: Path
    timeout_seconds: int = 180
    error_tail_characters: int = 1600

    def run(self, command: list[str], *, label: str) -> None:
        if not command:
            raise ValueError("branch command must not be empty")
        completed = subprocess.run(
            command,
            cwd=self.project_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=self.timeout_seconds,
            check=False,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout)[
                -self.error_tail_characters :
            ]
            raise RuntimeError(f"{label} branch failed:\n{detail}")
