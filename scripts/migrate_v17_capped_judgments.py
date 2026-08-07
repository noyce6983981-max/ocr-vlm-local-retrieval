"""Migrate completed V17 judgments to a deterministic capped candidate pool."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-packets", type=Path, required=True)
    parser.add_argument("--new-packets", type=Path, required=True)
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def index_packets(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed = {str(row["query_id"]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("Packet query IDs must be unique")
    return indexed


def main() -> None:
    args = parse_args()
    old_packets = index_packets(read_jsonl(project_path(args.old_packets)))
    new_packets = index_packets(read_jsonl(project_path(args.new_packets)))
    judgments = read_jsonl(project_path(args.judgments))
    migrated: list[dict[str, Any]] = []
    dropped_relevant_count = 0
    for judgment in judgments:
        query_id = str(judgment["query_id"])
        if query_id not in old_packets or query_id not in new_packets:
            raise ValueError(f"Unknown migrated query: {query_id}")
        old_packet = old_packets[query_id]
        new_packet = new_packets[query_id]
        old_items = {str(row["item_id"]) for row in old_packet["candidates"]}
        new_items = {str(row["item_id"]) for row in new_packet["candidates"]}
        if not new_items.issubset(old_items):
            raise ValueError(f"New pool is not a subset for query {query_id}")
        old_relevance = {
            str(item_id): bool(value)
            for item_id, value in judgment["candidate_relevance"].items()
        }
        if not old_items.issubset(old_relevance):
            raise ValueError(f"Old judgment has incomplete coverage: {query_id}")
        dropped_relevant = sorted(
            item_id for item_id in old_items - new_items if old_relevance[item_id]
        )
        dropped_relevant_count += len(dropped_relevant)
        migrated.append(
            {
                **judgment,
                "candidate_relevance": {
                    item_id: old_relevance[item_id] for item_id in sorted(new_items)
                },
                "study_fingerprint": new_packet["study_fingerprint"],
                "migrated_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "migration": {
                    "reason": "user_requested_total_pool_cap_20",
                    "from_study_fingerprint": old_packet["study_fingerprint"],
                    "to_study_fingerprint": new_packet["study_fingerprint"],
                    "candidate_count_before": len(old_items),
                    "candidate_count_after": len(new_items),
                    "known_relevant_dropped_item_ids": dropped_relevant,
                },
            }
        )
    write_jsonl_atomic(project_path(args.output), migrated)
    print(
        json.dumps(
            {
                "migrated_judgment_count": len(migrated),
                "known_relevant_dropped_count": dropped_relevant_count,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
