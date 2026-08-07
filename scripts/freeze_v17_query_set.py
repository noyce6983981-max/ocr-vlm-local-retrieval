"""Freeze the V17 query set only after complete human query review."""

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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from ocr_vlm_retrieval.evaluation.query_collection import (  # noqa: E402
    freeze_reviewed_queries,
    proposal_summary,
    query_fingerprint,
)
from ocr_vlm_retrieval.gating.attribute_coverage import (  # noqa: E402
    load_attribute_policy,
)
from scripts.prepare_v17_query_proposals import (  # noqa: E402
    read_excluded_queries,
    read_jsonl,
    write_jsonl_atomic,
)


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_reviews(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Human query review file not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path("data/evaluation/v17/query_proposals")
    parser.add_argument(
        "--proposals",
        type=Path,
        default=base / "compositional_visual_proposals.jsonl",
    )
    parser.add_argument(
        "--reviews",
        type=Path,
        default=base / "query_reviews.csv",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("config/v17_attribute_coverage.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/evaluation/v17/frozen_query_set"),
    )
    parser.add_argument("--expected-count", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    proposal_path = project_path(args.proposals)
    review_path = project_path(args.reviews)
    policy_path = project_path(args.policy)
    output_dir = project_path(args.output_dir)
    excluded_paths = [
        PROJECT_ROOT / "data/evaluation/v16/calibration/frozen_queries.csv",
        PROJECT_ROOT / "data/evaluation/v16/holdout/frozen_queries.csv",
        PROJECT_ROOT / "data/evaluation/blind_study_v1/formal_queries.csv",
    ]
    frozen = freeze_reviewed_queries(
        read_jsonl(proposal_path),
        read_reviews(review_path),
        load_attribute_policy(policy_path),
        expected_count=args.expected_count,
        excluded_queries=read_excluded_queries(excluded_paths),
    )
    summary = proposal_summary(frozen)
    receipt = {
        **summary,
        "status": "query_wording_frozen_relevance_not_yet_judged",
        "query_set_sha256": query_fingerprint(frozen),
        "proposal_file_sha256": hashlib.sha256(proposal_path.read_bytes()).hexdigest(),
        "review_file_sha256": hashlib.sha256(review_path.read_bytes()).hexdigest(),
        "attribute_policy_sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
        "retrieval_executed": False,
        "relevance_review_complete": False,
    }
    write_jsonl_atomic(output_dir / "frozen_queries.jsonl", frozen)
    write_json_atomic(output_dir / "freeze_receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
