"""Freeze label-blind V16 decisions for the V17 holdout comparison."""

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

from ocr_vlm_retrieval.evaluation.protocol_lock import file_sha256

BASELINE_SCOPE = "v17_holdout_v16_locked_baseline"
BASELINE_METHOD = "quality_hybrid"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def keyed(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        query_id = str(row.get("query_id", "")).strip()
        if not query_id or query_id in result:
            raise ValueError(f"Invalid or duplicate {label} query_id: {query_id!r}")
        result[query_id] = row
    return result


def build_baseline_records(
    packets: list[dict[str, Any]],
    ranking_rows: list[dict[str, Any]],
    raw_payloads: Mapping[str, Mapping[str, Any]],
    *,
    ranking_sha256: str,
) -> list[dict[str, Any]]:
    packet_by_id = keyed(packets, "packet")
    ranking_by_id = keyed(ranking_rows, "ranking")
    if set(packet_by_id) != set(ranking_by_id) or set(packet_by_id) != set(
        raw_payloads
    ):
        raise ValueError("Packets, rankings, and raw V16 payloads must align")

    records: list[dict[str, Any]] = []
    for query_id in sorted(packet_by_id):
        packet = packet_by_id[query_id]
        ranking = ranking_by_id[query_id].get("ranking", [])
        if not isinstance(ranking, list) or not ranking:
            raise ValueError(f"V16 ranking is empty for {query_id}")
        top_item_id = str(ranking[0].get("item_id", "")).strip()
        if not top_item_id:
            raise ValueError(f"V16 top candidate is invalid for {query_id}")

        raw = raw_payloads[query_id]
        if int(raw.get("search_policy_version", 0)) != 16:
            raise ValueError(f"Raw payload is not V16 for {query_id}")
        if raw.get("query") != packet.get("query"):
            raise ValueError(f"Raw V16 query mismatch for {query_id}")
        raw_rankings = raw.get("rankings", {})
        if not isinstance(raw_rankings, Mapping):
            raise ValueError(f"Raw V16 rankings are invalid for {query_id}")
        raw_ranking = raw_rankings.get(BASELINE_METHOD, [])
        if not isinstance(raw_ranking, list) or not raw_ranking:
            raise ValueError(f"Raw V16 baseline ranking is empty for {query_id}")
        if str(raw_ranking[0].get("item_id", "")) != top_item_id:
            raise ValueError(f"Frozen and raw V16 rankings differ for {query_id}")

        acceptance = raw.get("acceptance", {})
        if not isinstance(acceptance, Mapping):
            raise ValueError(f"Raw V16 acceptance is invalid for {query_id}")
        gate = acceptance.get(BASELINE_METHOD, {})
        if not isinstance(gate, Mapping) or not isinstance(
            gate.get("accepted"), bool
        ):
            raise ValueError(f"Raw V16 baseline decision is invalid for {query_id}")
        accepted = bool(gate["accepted"])
        records.append(
            {
                "query_id": query_id,
                "group_id": str(packet.get("group_id", "")).strip(),
                "scope": BASELINE_SCOPE,
                "judgments_read": False,
                "method": BASELINE_METHOD,
                "search_policy_version": 16,
                "retrieval_config_revision": str(
                    raw.get("retrieval_config_revision", "")
                ),
                "library_revision": str(raw.get("library_revision", "")),
                "ranking_sha256": ranking_sha256,
                "v16_accepted": accepted,
                "selected_item_id": top_item_id if accepted else None,
                "selected_rank": 1 if accepted else None,
                "gate_signal_name": str(gate.get("signal_name", "")),
                "gate_signal": gate.get("signal"),
                "gate_threshold": gate.get("threshold"),
            }
        )
    if any(not row["group_id"] for row in records):
        raise ValueError("Every baseline record needs a group_id")
    return records


def write_jsonl_new(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime_root = args.runtime_root.resolve()
    lock_path = PROJECT_ROOT / "config/v17_method_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    receipt_path = (
        runtime_root
        / "outputs/evaluation/v17/holdout/retrieval/"
        "holdout_retrieval_receipt.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("judgments_read") is not False:
        raise ValueError("Retrieval receipt must certify judgments_read=false")
    allowed_lock_hashes = {
        file_sha256(lock_path),
        str(lock.get("prior_lock_sha256", "")),
    }
    if receipt.get("method_lock_sha256") not in allowed_lock_hashes:
        raise ValueError("Retrieval receipt is not bound to this lock lineage")

    run = receipt.get("run_files", {}).get("v16_quality_hybrid", {})
    ranking_path = (
        runtime_root
        / "outputs/evaluation/v17/holdout/retrieval/runs/"
        "v16_quality_hybrid.jsonl"
    )
    ranking_hash = file_sha256(ranking_path)
    if not isinstance(run, Mapping) or run.get("sha256") != ranking_hash:
        raise ValueError("V16 ranking hash does not match the retrieval receipt")
    packets_path = (
        runtime_root
        / "data/evaluation/v17/human_study/holdout/review_packets.jsonl"
    )
    packets = read_jsonl(packets_path)
    raw_dir = runtime_root / "outputs/evaluation/v17/holdout/retrieval/raw"
    raw_paths = {
        str(row["query_id"]): raw_dir / f"{row['query_id']}_v16.json"
        for row in packets
    }
    raw_payloads = {
        query_id: json.loads(path.read_text(encoding="utf-8"))
        for query_id, path in raw_paths.items()
    }
    records = build_baseline_records(
        packets,
        read_jsonl(ranking_path),
        raw_payloads,
        ranking_sha256=ranking_hash,
    )
    output = (
        args.output.resolve()
        if args.output is not None
        else runtime_root / lock["execution"]["baseline_records"]
    )
    write_jsonl_new(output, records)
    print(
        json.dumps(
            {
                "status": "v16_holdout_baseline_frozen",
                "judgments_read": False,
                "query_count": len(records),
                "ranking_sha256": ranking_hash,
                "output": str(output),
                "output_sha256": file_sha256(output),
                "raw_payload_sha256": {
                    query_id: file_sha256(path)
                    for query_id, path in sorted(raw_paths.items())
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
