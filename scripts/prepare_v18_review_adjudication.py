"""Prepare complete V18 query pairs for third-reviewer conflict adjudication."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from ocr_vlm_retrieval.evaluation.human_evaluation import (  # noqa: E402
    agreement_report,
)
from scripts.run_v17_calibration_retrieval import (  # noqa: E402
    file_sha256,
    read_jsonl,
    write_json_atomic,
    write_jsonl_atomic,
)


def _index_packets(
    packets: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for packet in packets:
        query_id = str(packet.get("query_id", "")).strip()
        if not query_id or query_id in indexed:
            raise ValueError("Adjudication packets need unique query IDs")
        indexed[query_id] = dict(packet)
    return indexed


def build_adjudication_packets(
    packets: Sequence[Mapping[str, Any]],
    primary_rows: Sequence[Mapping[str, Any]],
    secondary_rows: Sequence[Mapping[str, Any]],
    *,
    expected_split: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    packet_index = _index_packets(packets)
    if {str(row.get("split", "")) for row in packets} != {expected_split}:
        raise ValueError(f"Expected {expected_split} review packets only")
    agreement = agreement_report([*primary_rows, *secondary_rows])
    conflict_query_ids = [str(value) for value in agreement["conflict_query_ids"]]
    if not conflict_query_ids:
        raise ValueError("No reviewer conflicts need adjudication")
    if not set(conflict_query_ids) <= set(packet_index):
        raise ValueError("Reviewer conflict is outside the frozen packet set")
    conflict_group_ids = {
        str(packet_index[query_id].get("group_id", ""))
        for query_id in conflict_query_ids
    }
    if "" in conflict_group_ids:
        raise ValueError("Every conflict packet needs a group ID")
    selected = sorted(
        (
            dict(packet)
            for packet in packets
            if str(packet.get("group_id", "")) in conflict_group_ids
        ),
        key=lambda row: (str(row["group_id"]), str(row["query_id"])),
    )
    group_counts = Counter(str(row["group_id"]) for row in selected)
    if set(group_counts) != conflict_group_ids or any(
        count != 2 for count in group_counts.values()
    ):
        raise ValueError("Conflict adjudication must preserve complete query pairs")
    report = {
        "status": f"v18_{expected_split}_adjudication_ready",
        "executed_split": expected_split,
        "conflict_query_ids": sorted(conflict_query_ids),
        "conflict_query_count": len(conflict_query_ids),
        "adjudication_group_ids": sorted(conflict_group_ids),
        "adjudication_pair_count": len(conflict_group_ids),
        "adjudication_query_count": len(selected),
        "agreement_before_adjudication": agreement,
        "v17_artifacts_modified": False,
    }
    return selected, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--secondary", type=Path, required=True)
    parser.add_argument("--output-packets", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--split", choices=("calibration", "holdout"), required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = {
        "packets": args.packets.resolve(),
        "primary": args.primary.resolve(),
        "secondary": args.secondary.resolve(),
        "output_packets": args.output_packets.resolve(),
        "output_report": args.output_report.resolve(),
    }
    selected, report = build_adjudication_packets(
        read_jsonl(paths["packets"]),
        read_jsonl(paths["primary"]),
        read_jsonl(paths["secondary"]),
        expected_split=args.split,
    )
    write_jsonl_atomic(paths["output_packets"], selected)
    report.update(
        {
            "source_packets_sha256": file_sha256(paths["packets"]),
            "primary_judgments_sha256": file_sha256(paths["primary"]),
            "secondary_judgments_sha256": file_sha256(paths["secondary"]),
            "adjudication_packets_sha256": file_sha256(paths["output_packets"]),
        }
    )
    write_json_atomic(paths["output_report"], report)
    print(paths["output_report"])


if __name__ == "__main__":
    main()
