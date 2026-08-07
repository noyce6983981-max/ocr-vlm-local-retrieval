"""Prepare a blind third-reviewer queue for V17 holdout conflicts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from ocr_vlm_retrieval.evaluation.human_evaluation import (
    agreement_report,
    validate_annotation_coverage,
)
from ocr_vlm_retrieval.evaluation.protocol_lock import file_sha256

UNRESOLVED_LABELS = {"uncertain", "excluded"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def write_jsonl_new(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def keyed(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        query_id = str(row.get("query_id", "")).strip()
        if not query_id or query_id in result:
            raise ValueError(f"Invalid or duplicate {label} query_id: {query_id!r}")
        result[query_id] = row
    return result


def prepare_queue(
    packets: list[dict[str, Any]],
    reviewer_1: list[dict[str, Any]],
    reviewer_2: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    packet_by_id = keyed(packets, "packet")
    first = keyed(reviewer_1, "reviewer_1")
    second = keyed(reviewer_2, "reviewer_2")
    if set(packet_by_id) != set(first) or set(packet_by_id) != set(second):
        raise ValueError("Both reviewers must cover the identical packet set")

    candidates = {
        query_id: [str(row["item_id"]) for row in packet["candidates"]]
        for query_id, packet in packet_by_id.items()
    }
    coverage = validate_annotation_coverage(
        [
            {"query_id": query_id, "split": "holdout"}
            for query_id in packet_by_id
        ],
        [*reviewer_1, *reviewer_2],
        candidate_ids_by_query=candidates,
    )
    if not coverage["valid"]:
        raise ValueError(f"Incomplete primary review: {coverage['failures']}")
    agreement = agreement_report([*reviewer_1, *reviewer_2])
    unresolved = {
        query_id
        for query_id in packet_by_id
        if str(first[query_id]["pool_relevance"]) in UNRESOLVED_LABELS
        or str(second[query_id]["pool_relevance"]) in UNRESOLVED_LABELS
    }
    conflict_ids = set(agreement["conflict_query_ids"]) | unresolved
    queue = [packet_by_id[query_id] for query_id in sorted(conflict_ids)]
    return queue, {
        "schema_version": 1,
        "status": "third_reviewer_queue_prepared",
        "primary_review_query_count": len(packet_by_id),
        "adjudication_query_count": len(queue),
        "conflict_query_count": len(agreement["conflict_query_ids"]),
        "unresolved_label_query_count": len(unresolved),
        "adjudication_query_ids": sorted(conflict_ids),
        "coverage": coverage,
        "agreement": agreement,
        "model_assisted_labels_used": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets", type=Path, required=True)
    parser.add_argument("--reviewer-1", type=Path, required=True)
    parser.add_argument("--reviewer-2", type=Path, required=True)
    parser.add_argument("--output-packets", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    packets = read_jsonl(args.packets)
    reviewer_1 = read_jsonl(args.reviewer_1)
    reviewer_2 = read_jsonl(args.reviewer_2)
    queue, report = prepare_queue(packets, reviewer_1, reviewer_2)
    report["input_hashes"] = {
        "packets": file_sha256(args.packets),
        "reviewer_1": file_sha256(args.reviewer_1),
        "reviewer_2": file_sha256(args.reviewer_2),
    }
    write_jsonl_new(args.output_packets, queue)
    report["output_packets_sha256"] = file_sha256(args.output_packets)
    write_json_new(args.output_report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
