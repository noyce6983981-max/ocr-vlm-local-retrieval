"""Fail when a public release contains secrets, private paths, or PII-like data."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


TEXT_SUFFIXES = {
    ".csv",
    ".ini",
    ".json",
    ".jsonl",
    ".md",
    ".ps1",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
SKIP_DIRS = {
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    ".venv-paddle",
    ".venv-vl",
    "__pycache__",
    "artifacts",
    "logs",
    "models",
    "output",
    "outputs",
    "third_party",
    "tmp",
    "work",
}
FORBIDDEN_PREFIXES = (
    "data/evaluation/archive/",
    "data/evaluation/blind_study_v1/",
    "data/evaluation/v16/development/",
    "data/incoming/",
    "data/processed/",
    "data/raw/",
    "records/daily/",
    "records/private/",
)
FORBIDDEN_FILES = {
    "scripts/build_application_packets.py",
    "records/presentation/INTERVIEW_PACKAGE.md",
}
SECRET_PATTERNS = {
    "OpenAI-style key": re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    "GitHub token": re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    "AWS access key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "private key": re.compile(r"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY"),
}
PERSONAL_PATH_PATTERN = re.compile(
    r"(?i)(?:[A-Z]:[\\/]+Users[\\/]+(?!Public(?:[\\/]|$))[^\\/\s]+|/home/[^/\s]+)"
)
PII_PATTERNS = {
    "email": re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"),
    "Chinese mobile": re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    "Chinese ID": re.compile(r"(?<!\d)\d{17}[0-9Xx](?!\d)"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path.cwd())
    return parser.parse_args()


def iter_text_files(root: Path):
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        relative = path.relative_to(root)
        if any(part in SKIP_DIRS for part in relative.parts):
            continue
        yield path, relative.as_posix()


def audit(root: Path) -> list[str]:
    errors: list[str] = []
    for path, relative in iter_text_files(root):
        if relative in FORBIDDEN_FILES or relative.startswith(FORBIDDEN_PREFIXES):
            errors.append(f"forbidden public path: {relative}")
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            errors.append(f"non-UTF-8 text file: {relative}")
            continue

        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(content):
                errors.append(f"{label}: {relative}")
        if (
            relative != "scripts/check_public_release.py"
            and PERSONAL_PATH_PATTERN.search(content)
        ):
            errors.append(f"personal absolute path: {relative}")

        # Test fixtures intentionally contain fake identifiers to exercise the
        # privacy gate. Public data and prose must not contain such payloads.
        if relative.startswith(("data/", "records/")):
            for label, pattern in PII_PATTERNS.items():
                if pattern.search(content):
                    errors.append(f"PII-like {label}: {relative}")

    required = (
        ".dockerignore",
        ".github/workflows/tests.yml",
        ".github/workflows/publish-container.yml",
        ".gitattributes",
        ".gitignore",
        "CONTRIBUTING.md",
        "Dockerfile.demo",
        "README.md",
        "SECURITY.md",
        "THIRD_PARTY_DATA.md",
        "config/selected_retrieval_config_v16.json",
        "config/v17_attribute_coverage.json",
        "pyproject.toml",
        "repro/README.md",
        "repro/expected_results.json",
        "requirements-ci.txt",
        "requirements-demo.txt",
        "scripts/run_container_demo.py",
    )
    for relative in required:
        if not (root / relative).is_file():
            errors.append(f"missing release file: {relative}")
    return sorted(set(errors))


def main() -> int:
    root = parse_args().root.resolve()
    errors = audit(root)
    if errors:
        print("Public-release audit failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"Public-release audit passed: {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
