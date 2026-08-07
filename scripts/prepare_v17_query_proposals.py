"""Prepare 80 V17 compositional query proposals without running retrieval."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from ocr_vlm_retrieval.evaluation.query_collection import (  # noqa: E402
    build_textvqa_compositional_proposals,
    proposal_summary,
)
from ocr_vlm_retrieval.gating.attribute_coverage import (  # noqa: E402
    load_attribute_policy,
)

DEFAULT_OUTPUT_DIR = Path("data/evaluation/v17/query_proposals")
REVIEW_FIELDS = (
    "query_id",
    "split",
    "query",
    "query_origin",
    "expected_answerability",
    "transformation_type",
    "source_item_id",
    "review_action",
    "reviewed_query",
    "reviewer_id",
    "human_notes",
)


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_excluded_queries(paths: list[Path]) -> set[str]:
    queries: set[str] = set()
    for path in paths:
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                query = " ".join(str(row.get("query", "")).split())
                if query:
                    queries.add(query)
    return queries


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_review_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in REVIEW_FIELDS})
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("outputs/user_library/manifest.jsonl")
    )
    parser.add_argument(
        "--policy", type=Path, default=Path("config/v17_attribute_coverage.json")
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = project_path(args.manifest)
    policy_path = project_path(args.policy)
    output_dir = project_path(args.output_dir)
    excluded_paths = [
        PROJECT_ROOT / "data/evaluation/v16/calibration/frozen_queries.csv",
        PROJECT_ROOT / "data/evaluation/v16/holdout/frozen_queries.csv",
        PROJECT_ROOT / "data/evaluation/blind_study_v1/formal_queries.csv",
    ]
    excluded = read_excluded_queries(excluded_paths)
    proposals = build_textvqa_compositional_proposals(
        read_jsonl(manifest_path),
        load_attribute_policy(policy_path),
        excluded_queries=excluded,
    )
    summary = {
        **proposal_summary(proposals),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "attribute_policy_sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
        "excluded_query_file_count": sum(path.is_file() for path in excluded_paths),
        "excluded_query_count": len(excluded),
        "retrieval_executed": False,
        "human_review_complete": False,
    }
    write_jsonl_atomic(output_dir / "compositional_visual_proposals.jsonl", proposals)
    write_review_csv(output_dir / "query_review_queue.csv", proposals)
    (output_dir / "proposal_receipt.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
