"""Export a clean GitHub-ready copy without private data or local history."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = {
    ".gitattributes",
    ".gitignore",
    "CONTRIBUTING.md",
    "README.md",
    "SECURITY.md",
    "THIRD_PARTY_DATA.md",
    "app.py",
    "correction_app.py",
    "pyproject.toml",
    "pytest.ini",
    "requirements-ci.txt",
    "requirements-ingest.txt",
    "requirements-retrieval.txt",
    "requirements-ui.txt",
    "requirements-vl.txt",
}
DATA_FILES = {
    "data/DATA_POLICY.md",
    "data/DATASET_200_SPEC.md",
    "data/README.md",
    "data/TAXONOMY_V2.md",
    "data/evaluation/dataset_v1_bm25_rrf_metrics.json",
    "data/evaluation/dataset_v1_queries.csv",
    "data/evaluation/error_analysis.csv",
    "data/evaluation/expansion_queries.csv",
    "data/evaluation/public_dataset_1500_retrieval_queries_formal.csv",
    "data/evaluation/public_dataset_1500_retrieval_query_protocol.json",
    "data/evaluation/public_dataset_1500_retrieval_query_queue_100.csv",
    "data/evaluation/retrieval_queries.csv",
    "data/evaluation/search_intent_regression_v12.csv",
    "data/evaluation/v16/blind_v1_regression_protocol.json",
    "data/evaluation/v16/calibration/dataset_protocol.json",
    "data/evaluation/v16/calibration/evaluation_protocol.json",
    "data/evaluation/v16/calibration/frozen_queries.csv",
    "data/evaluation/v16/holdout/dataset_protocol.json",
    "data/evaluation/v16/holdout/evaluation_protocol.json",
    "data/evaluation/v16/holdout/first_run_receipt.json",
    "data/evaluation/v16/holdout/frozen_queries.csv",
    "data/evaluation/v16/selected_config_lock.json",
    "data/evaluation/v17/amendments/001_scope_changed_to_compositional_80.json",
    "data/evaluation/v17/amendments/002_b3_failed_b5_selected.json",
    "data/evaluation/v17/amendments/"
    "003_separate_pooled_relevance_from_corpus_answerability.json",
    "data/evaluation/v17/protocol_frozen_v1.json",
    "data/evaluation/v17/study_status.json",
}
EXCLUDED_FILES = {
    "records/presentation/INTERVIEW_PACKAGE.md",
    "scripts/build_application_packets.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def git_candidates() -> list[str]:
    result = subprocess.run(
        [
            "git",
            "-c",
            "core.quotepath=false",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
    )
    return sorted(
        item for item in result.stdout.decode("utf-8").split("\0") if item
    )


def is_public_file(relative: str) -> bool:
    if relative in EXCLUDED_FILES:
        return False
    if "/__pycache__/" in f"/{relative}/" or relative.endswith((".pyc", ".pyo")):
        return False
    if "/" not in relative:
        return relative in ROOT_FILES or relative == "LICENSE"
    if relative.startswith(
        (".github/workflows/", "config/", "repro/", "scripts/", "src/", "tests/")
    ):
        return True
    if relative == ".streamlit/config.toml" or relative == "outputs/README.md":
        return True
    if relative.startswith("data/manifest/") or relative in DATA_FILES:
        return True
    if relative in {
        "records/COMMANDS.md",
        "records/CONTRIBUTION_LOG.md",
        "records/README.md",
    }:
        return True
    if relative.startswith(("records/experiments/", "records/research/")):
        return relative.endswith((".md", ".json"))
    if relative.startswith("records/screenshots/"):
        return relative.endswith((".png", ".jpg", ".jpeg", ".gitkeep"))
    if relative.startswith("records/presentation/"):
        return relative.endswith((".png", ".jpg", ".jpeg"))
    return False


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    output = parse_args().output.resolve()
    if output == PROJECT_ROOT or PROJECT_ROOT in output.parents:
        raise SystemExit("Output must be outside the source repository.")
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    selected = [path for path in git_candidates() if is_public_file(path)]
    for relative in selected:
        source = PROJECT_ROOT / relative
        if not source.is_file():
            continue
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    manifest_rows = [
        {"path": relative, "sha256": sha256(output / relative)}
        for relative in selected
        if (output / relative).is_file()
    ]
    manifest = {
        "schema_version": 1,
        "source_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "file_count": len(manifest_rows),
        "files": manifest_rows,
    }
    (output / "PUBLIC_RELEASE_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    audit = subprocess.run(
        [sys.executable, str(output / "scripts/check_public_release.py"), str(output)]
    )
    if audit.returncode:
        raise SystemExit("Export created but failed the public-release audit.")
    print(f"Exported {len(manifest_rows)} files to {output}")
    if not (output / "LICENSE").is_file():
        print("Warning: LICENSE is still waiting for the repository owner's choice.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
