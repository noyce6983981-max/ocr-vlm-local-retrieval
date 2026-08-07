"""Materialize final V17 holdout labels from independent human reviews."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from ocr_vlm_retrieval.evaluation.judgments import (
    NO_RELEVANT_CANDIDATE_IN_POOL,
    POOLED_RELEVANCE_TASK,
    RELEVANT_CANDIDATE_IN_POOL,
    normalize_pool_judgment,
)
from ocr_vlm_retrieval.evaluation.protocol_lock import file_sha256

FINAL_POOL_LABELS = {
    RELEVANT_CANDIDATE_IN_POOL,
    NO_RELEVANT_CANDIDATE_IN_POOL,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def keyed(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for source in rows:
        row = normalize_pool_judgment(source)
        query_id = str(row.get("query_id", "")).strip()
        if not query_id or query_id in result:
            raise ValueError(f"Invalid or duplicate {label} query_id: {query_id!r}")
        result[query_id] = row
    return result


def packet_candidates(packet: Mapping[str, Any]) -> list[str]:
    sources = packet.get("candidates", [])
    if not isinstance(sources, list):
        raise ValueError("Packet candidates must be a list")
    candidates = [str(row.get("item_id", "")).strip() for row in sources]
    if not candidates or any(not item_id for item_id in candidates):
        raise ValueError("Packet candidate IDs cannot be empty")
    if len(candidates) != len(set(candidates)):
        raise ValueError("Packet candidate IDs must be unique")
    return candidates


def validate_review(
    row: Mapping[str, Any],
    *,
    packet: Mapping[str, Any],
    expected_reviewer_id: str | None = None,
) -> None:
    if (
        expected_reviewer_id is not None
        and row.get("reviewer_id") != expected_reviewer_id
    ):
        raise ValueError(
            f"Unexpected reviewer for {row.get('query_id')}: {row.get('reviewer_id')}"
        )
    if row.get("study_fingerprint") != packet.get("study_fingerprint"):
        raise ValueError(f"Study fingerprint mismatch for {row.get('query_id')}")
    if row.get("pool_sha256") != packet.get("pool_sha256"):
        raise ValueError(f"Pool hash mismatch for {row.get('query_id')}")
    relevance = row.get("candidate_relevance", {})
    if not isinstance(relevance, Mapping):
        raise ValueError(f"Invalid candidate labels for {row.get('query_id')}")
    expected_candidates = set(packet_candidates(packet))
    if set(relevance) != expected_candidates:
        raise ValueError(f"Candidate coverage mismatch for {row.get('query_id')}")


def decisions_match(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    return (
        first["pool_relevance"] == second["pool_relevance"]
        and first["candidate_relevance"] == second["candidate_relevance"]
    )


def adjudicate_holdout(
    packets: list[dict[str, Any]],
    reviewer_1: list[dict[str, Any]],
    reviewer_2: list[dict[str, Any]],
    third_queue: list[dict[str, Any]],
    reviewer_3: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    packet_by_id = {str(row.get("query_id", "")): row for row in packets}
    if "" in packet_by_id or len(packet_by_id) != len(packets):
        raise ValueError("Review packets contain an empty or duplicate query_id")
    first = keyed(reviewer_1, "reviewer_1")
    second = keyed(reviewer_2, "reviewer_2")
    third = keyed(reviewer_3, "reviewer_3")
    queue_ids = [str(row.get("query_id", "")).strip() for row in third_queue]
    if not queue_ids or any(not query_id for query_id in queue_ids):
        raise ValueError("Third-review queue cannot be empty")
    if len(queue_ids) != len(set(queue_ids)):
        raise ValueError("Third-review queue contains duplicate query IDs")
    if set(first) != set(packet_by_id) or set(second) != set(packet_by_id):
        raise ValueError("Both primary reviewers must cover every holdout packet")
    if set(third) != set(queue_ids):
        raise ValueError("Third reviewer must cover exactly the adjudication queue")
    if not set(queue_ids) <= set(packet_by_id):
        raise ValueError("Third-review queue contains an unknown query")

    primary_ids = {
        str(row.get("reviewer_id", "")).strip()
        for row in [*reviewer_1, *reviewer_2]
    }
    third_ids = {
        str(row.get("reviewer_id", "")).strip() for row in reviewer_3
    }
    if len(primary_ids) != 2 or len(third_ids) != 1:
        raise ValueError("Expected two primary reviewers and one adjudicator")
    if primary_ids & third_ids:
        raise ValueError("The adjudicator must be independent of primary reviewers")

    queue_set = set(queue_ids)
    final_rows: list[dict[str, Any]] = []
    pool_counts: Counter[str] = Counter()
    conflict_count = 0
    unresolved_primary_count = 0
    for query_id in sorted(packet_by_id):
        packet = packet_by_id[query_id]
        for row in (first[query_id], second[query_id]):
            validate_review(row, packet=packet)
        primary_resolved = (
            first[query_id]["pool_relevance"] in FINAL_POOL_LABELS
            and second[query_id]["pool_relevance"] in FINAL_POOL_LABELS
        )
        primary_match = decisions_match(first[query_id], second[query_id])
        needs_adjudication = not primary_match or not primary_resolved
        if needs_adjudication != (query_id in queue_set):
            raise ValueError(
                f"Third-review queue does not match primary conflict state: {query_id}"
            )

        if query_id in queue_set:
            decision = third[query_id]
            validate_review(decision, packet=packet)
            method = "independent_third_reviewer_blind"
            conflict_count += int(not primary_match)
            unresolved_primary_count += int(not primary_resolved)
            source_reviewer_ids = sorted(primary_ids | third_ids)
            adjudicator_id = next(iter(third_ids))
            human_reviewer_count = 3
        else:
            decision = first[query_id]
            method = "independent_primary_consensus"
            source_reviewer_ids = sorted(primary_ids)
            adjudicator_id = None
            human_reviewer_count = 2

        pool_relevance = str(decision["pool_relevance"])
        if pool_relevance not in FINAL_POOL_LABELS:
            raise ValueError(f"Final decision remains unresolved: {query_id}")
        candidates = packet_candidates(packet)
        relevance = {
            item_id: bool(decision["candidate_relevance"][item_id])
            for item_id in candidates
        }
        any_relevant = any(relevance.values())
        if (pool_relevance == RELEVANT_CANDIDATE_IN_POOL) != any_relevant:
            raise ValueError(f"Inconsistent final pool decision: {query_id}")

        final_row = {
            "query_id": query_id,
            "reviewer_id": "final_human_adjudication",
            "reviewer_role": "final_human_adjudication",
            "reviewer_type": "human",
            "task_id": POOLED_RELEVANCE_TASK,
            "pool_relevance": pool_relevance,
            "pool_sha256": packet["pool_sha256"],
            "candidate_relevance": relevance,
            "study_fingerprint": packet["study_fingerprint"],
            "independent_reviewer_count": 2,
            "total_human_reviewer_count": human_reviewer_count,
            "adjudicated": True,
            "adjudication_method": method,
            "source_reviewer_ids": source_reviewer_ids,
            "model_assisted_labels_used": False,
        }
        if adjudicator_id is not None:
            final_row["adjudicator_id"] = adjudicator_id
        final_rows.append(final_row)
        pool_counts[pool_relevance] += 1

    return final_rows, {
        "schema_version": 1,
        "status": "holdout_human_adjudication_complete",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "query_count": len(final_rows),
        "primary_reviewer_count": 2,
        "independent_adjudicator_count": 1,
        "adjudication_query_count": len(queue_set),
        "primary_conflict_query_count": conflict_count,
        "unresolved_primary_query_count": unresolved_primary_count,
        "pool_relevance_counts": dict(sorted(pool_counts.items())),
        "independent_reviews": True,
        "full_candidate_coverage": True,
        "blinded_to_method_outputs": True,
        "model_assisted_labels_used": False,
        "conflict_adjudication_complete": True,
        "adjudicator_independent_of_primary_reviewers": True,
    }


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets", type=Path, required=True)
    parser.add_argument("--reviewer-1", type=Path, required=True)
    parser.add_argument("--reviewer-2", type=Path, required=True)
    parser.add_argument("--third-queue", type=Path, required=True)
    parser.add_argument("--reviewer-3", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    final_rows, report = adjudicate_holdout(
        read_jsonl(args.packets),
        read_jsonl(args.reviewer_1),
        read_jsonl(args.reviewer_2),
        read_jsonl(args.third_queue),
        read_jsonl(args.reviewer_3),
    )
    report["input_hashes"] = {
        "packets": file_sha256(args.packets),
        "reviewer_1": file_sha256(args.reviewer_1),
        "reviewer_2": file_sha256(args.reviewer_2),
        "third_queue": file_sha256(args.third_queue),
        "reviewer_3": file_sha256(args.reviewer_3),
    }
    write_jsonl_new(args.output, final_rows)
    report["output_sha256"] = file_sha256(args.output)
    write_json_new(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
