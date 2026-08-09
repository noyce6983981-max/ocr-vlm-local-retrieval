"""Validate V18 calibration review, report agreement, and freeze final labels."""

from __future__ import annotations

import argparse
import math
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

RELEVANT = "relevant_candidate_in_pool"
NOT_RELEVANT = "no_relevant_candidate_in_pool"


def _index_unique(
    rows: Sequence[Mapping[str, Any]], *, label: str
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for source in rows:
        query_id = str(source.get("query_id", "")).strip()
        if not query_id or query_id in indexed:
            raise ValueError(f"Every {label} row needs a unique query_id")
        indexed[query_id] = dict(source)
    return indexed


def _reviewer_identity(
    rows: Sequence[Mapping[str, Any]], *, expected_role: str
) -> str:
    reviewer_ids = {str(row.get("reviewer_id", "")).strip() for row in rows}
    roles = {str(row.get("reviewer_role", "")).strip() for row in rows}
    if len(reviewer_ids) != 1 or not next(iter(reviewer_ids)):
        raise ValueError(f"{expected_role} review needs exactly one reviewer ID")
    if roles != {expected_role}:
        raise ValueError(f"Expected reviewer_role={expected_role}")
    return next(iter(reviewer_ids))


def _validate_judgment(
    judgment: Mapping[str, Any], packet: Mapping[str, Any]
) -> None:
    query_id = str(packet["query_id"])
    if judgment.get("study_fingerprint") != packet.get("study_fingerprint"):
        raise ValueError(f"Study fingerprint mismatch for {query_id}")
    if judgment.get("pool_sha256") != packet.get("pool_sha256"):
        raise ValueError(f"Pool hash mismatch for {query_id}")
    relevance = judgment.get("candidate_relevance")
    if not isinstance(relevance, dict):
        raise ValueError(f"Missing candidate relevance for {query_id}")
    expected_ids = {str(row["item_id"]) for row in packet["candidates"]}
    if set(map(str, relevance)) != expected_ids:
        raise ValueError(f"Candidate set mismatch for {query_id}")
    expected_pool = RELEVANT if any(map(bool, relevance.values())) else NOT_RELEVANT
    if judgment.get("pool_relevance") != expected_pool:
        raise ValueError(f"Pool decision mismatch for {query_id}")


def finalize_review(
    packets: Sequence[Mapping[str, Any]],
    primary_rows: Sequence[Mapping[str, Any]],
    secondary_rows: Sequence[Mapping[str, Any]],
    adjudication_rows: Sequence[Mapping[str, Any]] | None = None,
    *,
    expected_query_count: int = 80,
    minimum_double_review_fraction: float = 0.30,
    expected_split: str = "calibration",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    packet_index = _index_unique(packets, label="packet")
    primary = _index_unique(primary_rows, label="primary judgment")
    secondary = _index_unique(secondary_rows, label="secondary judgment")
    if len(packet_index) != expected_query_count:
        raise ValueError(f"Expected {expected_query_count} calibration packets")
    if {str(row.get("split", "")) for row in packets} != {expected_split}:
        raise ValueError(f"V18 finalization may load {expected_split} packets only")
    if set(primary) != set(packet_index):
        raise ValueError("Primary reviewer must cover every calibration query")
    required_secondary = math.ceil(
        expected_query_count * minimum_double_review_fraction
    )
    if len(secondary) < required_secondary or not set(secondary) <= set(packet_index):
        raise ValueError("Secondary review does not meet the locked coverage")

    primary_id = _reviewer_identity(primary_rows, expected_role="primary")
    secondary_id = _reviewer_identity(secondary_rows, expected_role="secondary")
    if primary_id == secondary_id:
        raise ValueError("Primary and secondary reviewer IDs must differ")
    for query_id, judgment in primary.items():
        _validate_judgment(judgment, packet_index[query_id])
    for query_id, judgment in secondary.items():
        _validate_judgment(judgment, packet_index[query_id])

    secondary_groups = Counter(
        str(packet_index[query_id].get("group_id", "")) for query_id in secondary
    )
    if any(not group_id or count != 2 for group_id, count in secondary_groups.items()):
        raise ValueError("Secondary review must cover complete query pairs")
    agreement = agreement_report([*primary_rows, *secondary_rows])
    conflict_query_ids = {str(value) for value in agreement["conflict_query_ids"]}
    adjudication = _index_unique(
        adjudication_rows or [], label="adjudication judgment"
    )
    adjudicator_id: str | None = None
    if conflict_query_ids:
        if not adjudication:
            raise ValueError("Secondary conflicts require adjudication before finalization")
        if not conflict_query_ids <= set(adjudication):
            raise ValueError("Adjudication must cover every conflicting query")
        if not set(adjudication) <= set(packet_index):
            raise ValueError("Adjudication contains a query outside the frozen pool")
        adjudicator_id = _reviewer_identity(
            list(adjudication.values()), expected_role="adjudicator"
        )
        if adjudicator_id in {primary_id, secondary_id}:
            raise ValueError("Adjudicator ID must differ from both reviewers")
        for query_id, judgment in adjudication.items():
            _validate_judgment(judgment, packet_index[query_id])
        adjudication_groups = Counter(
            str(packet_index[query_id].get("group_id", ""))
            for query_id in adjudication
        )
        if any(
            not group_id or count != 2
            for group_id, count in adjudication_groups.items()
        ):
            raise ValueError("Adjudication must cover complete query pairs")
    elif adjudication:
        raise ValueError("Adjudication was supplied without reviewer conflicts")

    finalized = []
    for query_id in sorted(primary):
        adjudicated = query_id in conflict_query_ids
        row = dict(adjudication[query_id] if adjudicated else primary[query_id])
        double_reviewed = query_id in secondary
        row.update(
            {
                "finalized": True,
                "double_reviewed": double_reviewed,
                "finalization_method": (
                    "third_reviewer_adjudication"
                    if adjudicated
                    else "primary_secondary_exact_consensus"
                    if double_reviewed
                    else "primary_single_review"
                ),
                "primary_reviewer_id": primary_id,
                "secondary_reviewer_id": secondary_id if double_reviewed else None,
                "adjudicator_id": adjudicator_id if adjudicated else None,
                "adjudication_required": False,
                "adjudicated": adjudicated,
            }
        )
        finalized.append(row)
    report = {
        "status": f"v18_{expected_split}_review_complete",
        "executed_split": expected_split,
        "query_count": len(finalized),
        "pair_count": len({str(row.get("group_id", "")) for row in packets}),
        "primary_reviewer_id": primary_id,
        "secondary_reviewer_id": secondary_id,
        "reviewer_ids_distinct": True,
        "double_reviewed_query_count": len(secondary),
        "double_review_fraction": len(secondary) / len(packet_index),
        "secondary_complete_pair_count": len(secondary_groups),
        "agreement": agreement,
        "conflict_adjudication_required": False,
        "adjudicator_id": adjudicator_id,
        "adjudicated_query_count": len(conflict_query_ids),
        "adjudicated_query_ids": sorted(conflict_query_ids),
        "holdout_results_opened": expected_split == "holdout",
        "holdout_retrieval_executed": expected_split == "holdout",
        "v17_artifacts_modified": False,
    }
    return finalized, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--secondary", type=Path, required=True)
    parser.add_argument("--adjudication", type=Path)
    parser.add_argument("--output-judgments", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--expected-query-count", type=int, default=80)
    parser.add_argument("--minimum-double-review-fraction", type=float, default=0.30)
    parser.add_argument("--split", choices=("calibration", "holdout"), default="calibration")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = {
        "packets": args.packets.resolve(),
        "primary": args.primary.resolve(),
        "secondary": args.secondary.resolve(),
        "adjudication": args.adjudication.resolve() if args.adjudication else None,
        "output_judgments": args.output_judgments.resolve(),
        "output_report": args.output_report.resolve(),
    }
    finalized, report = finalize_review(
        read_jsonl(paths["packets"]),
        read_jsonl(paths["primary"]),
        read_jsonl(paths["secondary"]),
        read_jsonl(paths["adjudication"]) if paths["adjudication"] else None,
        expected_query_count=args.expected_query_count,
        minimum_double_review_fraction=args.minimum_double_review_fraction,
        expected_split=args.split,
    )
    write_jsonl_atomic(paths["output_judgments"], finalized)
    report.update(
        {
            "packets_sha256": file_sha256(paths["packets"]),
            "primary_judgments_sha256": file_sha256(paths["primary"]),
            "secondary_judgments_sha256": file_sha256(paths["secondary"]),
            "adjudication_judgments_sha256": (
                file_sha256(paths["adjudication"])
                if paths["adjudication"]
                else None
            ),
            "final_judgments_sha256": file_sha256(paths["output_judgments"]),
        }
    )
    write_json_atomic(paths["output_report"], report)
    print(paths["output_report"])


if __name__ == "__main__":
    main()
