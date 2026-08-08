"""Recompute V17 public metrics and verify the de-identified audit bundle."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from ocr_vlm_retrieval.evaluation.protocol_lock import file_sha256
from ocr_vlm_retrieval.evaluation.public_holdout import (
    analyze_public_paired_records,
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def require_close(actual: float, expected: float, *, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{label} differs: {actual} != {expected}")


def verify_bundle(
    *, paired_path: Path, audit_path: Path, summary_path: Path
) -> dict[str, Any]:
    audit = read_json(audit_path)
    summary = read_json(summary_path)
    expected_hash = str(audit["public_artifact"]["sha256"])
    if file_sha256(paired_path) != expected_hash:
        raise ValueError("Public paired-record hash differs from the audit")
    bootstrap = summary["bootstrap"]
    analysis = analyze_public_paired_records(
        read_jsonl(paired_path),
        bootstrap_repetitions=int(bootstrap["repetitions"]),
        seed=int(bootstrap["seed"]),
    )
    if analysis != audit["analysis"]:
        raise ValueError("Recomputed analysis differs from the published audit")

    metrics = analysis["metrics"]
    summary_metrics = summary["metrics"]
    comparisons = {
        "V16 end-to-end accuracy": (
            metrics["v16_pool_conditioned_end_to_end_accuracy"],
            summary_metrics["v16_pool_conditioned_end_to_end_accuracy"],
        ),
        "V17 end-to-end accuracy": (
            metrics["v17_pool_conditioned_end_to_end_accuracy"],
            summary_metrics["v17_pool_conditioned_end_to_end_accuracy"],
        ),
        "V16 false-accept rate": (
            metrics["v16_pool_conditioned_false_accept_rate"],
            summary_metrics["v16_pool_conditioned_false_accept_rate"],
        ),
        "V17 false-accept rate": (
            metrics["v17_pool_conditioned_false_accept_rate"],
            summary_metrics["v17_pool_conditioned_false_accept_rate"],
        ),
        "V17 Recall@3": (
            metrics["v17_retrieval_recall_at_3"],
            summary_metrics["retrieval_recall_at_3"],
        ),
        "V17 selected relevant rate": (
            metrics["v17_positive_end_to_end_success_rate"],
            summary_metrics["selected_relevant_query_rate"],
        ),
        "V17 acceptance coverage": (
            metrics["v17_acceptance_coverage"],
            summary_metrics["acceptance_coverage"],
        ),
    }
    for label, (actual, expected) in comparisons.items():
        require_close(float(actual), float(expected), label=label)

    end_to_end = analysis["paired_group_bootstrap"][
        "end_to_end_correct_v17_minus_v16"
    ]
    require_close(
        float(end_to_end["paired_difference"]),
        float(summary_metrics["end_to_end_paired_difference"]),
        label="end-to-end paired difference",
    )
    for actual, expected in zip(
        end_to_end["confidence_interval"],
        summary_metrics["end_to_end_group_bootstrap_95_ci"],
        strict=True,
    ):
        require_close(float(actual), float(expected), label="end-to-end CI")
    return {
        "valid": True,
        "query_count": analysis["query_count"],
        "group_count": analysis["group_count"],
        "paired_records_sha256": expected_hash,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paired",
        type=Path,
        default=PROJECT_ROOT
        / "data/evaluation/v17/public_holdout_paired_records.jsonl",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=PROJECT_ROOT / "data/evaluation/v17/public_holdout_audit.json",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=PROJECT_ROOT / "data/evaluation/v17/final_holdout_summary.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            verify_bundle(
                paired_path=args.paired,
                audit_path=args.audit,
                summary_path=args.summary,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
