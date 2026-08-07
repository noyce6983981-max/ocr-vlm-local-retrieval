"""Materialize a traceable model-assisted adjudication layer for V17.

The two human-review rows remain immutable.  This script validates a compact
decision file against the active blinded candidate pool, expands each decision
to a complete candidate-relevance map, and writes an audit report describing
every difference from the primary and secondary reviewers.
"""

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

from ocr_vlm_retrieval.evaluation.judgments import (
    POOLED_RELEVANCE_TASK,
    RELEVANT_CANDIDATE_IN_POOL,
    pool_relevance_decision,
)

DEFAULT_STUDY_DIR = Path("data/evaluation/v17/human_study/calibration")
VALID_CONFIDENCE = {"high", "medium"}


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json_atomic(path: Path, payload: Any) -> None:
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relevant_items(row: dict[str, Any]) -> set[str]:
    return {
        item_id
        for item_id, is_relevant in row.get("candidate_relevance", {}).items()
        if is_relevant
    }


def materialize_adjudication(
    packets: list[dict[str, Any]],
    judgments: list[dict[str, Any]],
    decision_payload: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    packet_by_id = {row["query_id"]: row for row in packets}
    if len(packet_by_id) != len(packets):
        raise ValueError("Duplicate query_id in review packets")

    decisions = decision_payload.get("decisions", [])
    decision_by_id = {row["query_id"]: row for row in decisions}
    if len(decision_by_id) != len(decisions):
        raise ValueError("Duplicate query_id in decision file")
    if set(decision_by_id) != set(packet_by_id):
        missing = sorted(set(packet_by_id) - set(decision_by_id))
        extra = sorted(set(decision_by_id) - set(packet_by_id))
        raise ValueError(f"Decision/query mismatch: missing={missing}, extra={extra}")

    human_by_query: dict[str, list[dict[str, Any]]] = {}
    for row in judgments:
        human_by_query.setdefault(row["query_id"], []).append(row)

    adjudicated: list[dict[str, Any]] = []
    differences: list[dict[str, Any]] = []
    pool_relevance_counts: Counter[str] = Counter()
    confidence_counts: Counter[str] = Counter()

    for query_id in sorted(packet_by_id):
        packet = packet_by_id[query_id]
        decision = decision_by_id[query_id]
        pool_relevance = pool_relevance_decision(decision)
        confidence = str(decision.get("confidence", ""))
        if pool_relevance == "uncertain":
            raise ValueError(f"Adjudication cannot remain uncertain: {query_id}")
        if confidence not in VALID_CONFIDENCE:
            raise ValueError(f"Invalid confidence for {query_id}: {confidence}")

        candidate_ids = [row["item_id"] for row in packet["candidates"]]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError(f"Duplicate candidate in packet {query_id}")
        relevant = set(decision.get("relevant_item_ids", []))
        unknown = sorted(relevant - set(candidate_ids))
        if unknown:
            raise ValueError(f"Unknown relevant candidates for {query_id}: {unknown}")
        if pool_relevance == RELEVANT_CANDIDATE_IN_POOL and not relevant:
            raise ValueError(
                f"Relevant-in-pool decision has no relevant item: {query_id}"
            )
        if pool_relevance != RELEVANT_CANDIDATE_IN_POOL and relevant:
            raise ValueError(
                f"No-relevant/excluded decision has relevant items: {query_id}"
            )

        row = {
            "query_id": query_id,
            "reviewer_id": decision_payload["adjudicator_id"],
            "reviewer_role": "model_assisted_adjudicator",
            "reviewer_type": "model_assisted",
            "task_id": POOLED_RELEVANCE_TASK,
            "pool_relevance": pool_relevance,
            "pool_sha256": packet.get("pool_sha256"),
            "candidate_relevance": {
                item_id: item_id in relevant for item_id in candidate_ids
            },
            "confidence": confidence,
            "rationale": decision["rationale"],
            "evidence": decision.get("evidence", []),
            "study_fingerprint": packet["study_fingerprint"],
            "source_decisions_sha256": decision_payload["decision_revision"],
        }
        adjudicated.append(row)
        pool_relevance_counts[pool_relevance] += 1
        confidence_counts[confidence] += 1

        query_differences: list[dict[str, Any]] = []
        for human in human_by_query.get(query_id, []):
            human_relevant = relevant_items(human)
            human_pool_relevance = pool_relevance_decision(human)
            if human_pool_relevance != pool_relevance or human_relevant != relevant:
                query_differences.append(
                    {
                        "reviewer_id": human.get("reviewer_id"),
                        "reviewer_role": human.get("reviewer_role"),
                        "human_pool_relevance": human_pool_relevance,
                        "adjudicated_pool_relevance": pool_relevance,
                        "human_relevant_item_ids": sorted(human_relevant),
                        "adjudicated_relevant_item_ids": sorted(relevant),
                        "added_by_adjudication": sorted(relevant - human_relevant),
                        "removed_by_adjudication": sorted(human_relevant - relevant),
                    }
                )
        if query_differences:
            differences.append(
                {
                    "query_id": query_id,
                    "confidence": confidence,
                    "comparisons": query_differences,
                }
            )

    primary = [row for row in judgments if row.get("reviewer_role") == "primary"]
    secondary = [row for row in judgments if row.get("reviewer_role") == "secondary"]
    primary_changed = {
        entry["query_id"]
        for entry in differences
        if any(
            row["reviewer_role"] == "primary"
            for row in entry["comparisons"]
        )
    }
    secondary_changed = {
        entry["query_id"]
        for entry in differences
        if any(
            row["reviewer_role"] == "secondary"
            for row in entry["comparisons"]
        )
    }
    report = {
        "schema_version": 2,
        "task_id": POOLED_RELEVANCE_TASK,
        "status": "model_assisted_calibration_adjudication_complete",
        "adjudicator_id": decision_payload["adjudicator_id"],
        "adjudicator_type": "model_assisted_visual_audit_not_human_gold",
        "decision_revision": decision_payload["decision_revision"],
        "policy": decision_payload["policy"],
        "query_count": len(adjudicated),
        "human_judgment_count": len(judgments),
        "primary_judgment_count": len(primary),
        "secondary_judgment_count": len(secondary),
        "pool_relevance_counts": dict(sorted(pool_relevance_counts.items())),
        "confidence_counts": dict(sorted(confidence_counts.items())),
        "primary_queries_changed": len(primary_changed),
        "secondary_queries_changed": len(secondary_changed),
        "difference_query_count": len(differences),
        "differences": differences,
        "limitations": decision_payload["limitations"],
    }
    return adjudicated, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=DEFAULT_STUDY_DIR)
    parser.add_argument(
        "--decisions",
        type=Path,
        default=DEFAULT_STUDY_DIR / "model_audit_decisions.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    study_dir = project_path(args.study_dir)
    decisions_path = project_path(args.decisions)
    packets_path = study_dir / "review_packets.jsonl"
    judgments_path = study_dir / "judgments.jsonl"
    decision_payload = json.loads(decisions_path.read_text(encoding="utf-8"))
    decision_payload["decision_revision"] = sha256(decisions_path)
    adjudicated, report = materialize_adjudication(
        read_jsonl(packets_path),
        read_jsonl(judgments_path),
        decision_payload,
    )
    output_path = study_dir / "adjudicated_judgments.jsonl"
    report_path = study_dir / "model_audit_report.json"
    write_jsonl_atomic(output_path, adjudicated)
    report["source_artifacts_sha256"] = {
        "decisions": sha256(decisions_path),
        "review_packets": sha256(packets_path),
        "human_judgments": sha256(judgments_path),
        "adjudicated_judgments": sha256(output_path),
    }
    write_json_atomic(report_path, report)
    print(output_path)
    print(report_path)


if __name__ == "__main__":
    main()
