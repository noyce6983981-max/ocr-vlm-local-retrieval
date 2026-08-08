"""Freeze a completed paired-query authoring queue exactly once."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from ocr_vlm_retrieval.studies.protocol import (  # noqa: E402
    load_study_protocol,
    protocol_fingerprint,
)
from ocr_vlm_retrieval.studies.query_split import (  # noqa: E402
    freeze_authored_queries,
    query_set_fingerprint,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def write_jsonl_new(path: Path, rows: list[dict[str, Any]]) -> None:
    """Create a frozen artifact without permitting accidental replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_ROOT / "config/studies/v18.json",
    )
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--submissions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = load_study_protocol(args.protocol.resolve())
    queue_path = args.queue.resolve()
    submissions_path = args.submissions.resolve()
    output_path = args.output.resolve()
    receipt_path = args.receipt.resolve()
    if output_path.exists() or receipt_path.exists():
        raise FileExistsError("frozen query output or receipt already exists")
    frozen = freeze_authored_queries(
        read_jsonl(queue_path),
        read_jsonl(submissions_path),
        study_id=protocol.study_id,
    )
    if len(frozen) != protocol.query_design.total_queries:
        raise ValueError("frozen query count does not match the study protocol")
    actual_counts = {
        "splits": Counter(str(row["split"]) for row in frozen),
        "roles": Counter(str(row["query_role"]) for row in frozen),
        "strata": Counter(str(row["stratum"]) for row in frozen),
        "languages": Counter(str(row["language"]) for row in frozen),
    }
    expected_counts = {
        "splits": protocol.query_design.splits,
        "roles": protocol.query_design.roles,
        "strata": protocol.query_design.strata,
        "languages": protocol.query_design.languages,
    }
    for name, actual in actual_counts.items():
        if dict(actual) != expected_counts[name]:
            raise ValueError(f"frozen {name} counts do not match the protocol")
    write_jsonl_new(output_path, frozen)
    receipt = {
        "study_id": protocol.study_id,
        "protocol_sha256": protocol_fingerprint(protocol),
        "authoring_queue_sha256": sha256(queue_path),
        "authoring_submissions_sha256": sha256(submissions_path),
        "query_set_sha256": query_set_fingerprint(frozen),
        "query_count": len(frozen),
        "split_counts": dict(sorted(actual_counts["splits"].items())),
        "role_counts": dict(sorted(actual_counts["roles"].items())),
        "stratum_counts": dict(sorted(actual_counts["strata"].items())),
        "language_counts": dict(sorted(actual_counts["languages"].items())),
        "retrieval_executed": False,
        "holdout_results_opened": False,
        "v17_artifacts_modified": False,
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with receipt_path.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
