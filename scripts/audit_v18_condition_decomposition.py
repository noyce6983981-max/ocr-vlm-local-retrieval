"""Audit V18 condition decomposition on the isolated calibration split."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from ocr_vlm_retrieval.gating.attribute_coverage import (  # noqa: E402
    load_attribute_policy,
)
from ocr_vlm_retrieval.gating.condition_decomposition import (  # noqa: E402
    PARSER_VERSION,
    decompose_condition_query,
)
from ocr_vlm_retrieval.studies.query_split import (  # noqa: E402
    query_set_fingerprint,
)
from scripts.run_v17_calibration_retrieval import (  # noqa: E402
    file_sha256,
    read_jsonl,
    write_json_atomic,
)


def audit_plans(
    queries: Sequence[Mapping[str, Any]], policy: Mapping[str, Any]
) -> dict[str, Any]:
    plans = {
        str(row["query_id"]): decompose_condition_query(str(row["query"]), policy)
        for row in queries
    }
    requirement_counts = [len(plan.requirements) for plan in plans.values()]
    kind_counts = Counter(
        requirement.kind
        for plan in plans.values()
        for requirement in plan.requirements
    )
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in queries:
        groups[str(row["group_id"])].append(row)
    failed_changed_detection: list[str] = []
    for group_id, rows in groups.items():
        if len(rows) != 2:
            raise ValueError(f"V18 group {group_id} must contain exactly two queries")
        positive = next(row for row in rows if row["query_role"] == "positive")
        negative = next(
            row
            for row in rows
            if row["query_role"] == "single_condition_hard_negative"
        )
        positive_conditions = {
            (row.kind, row.value)
            for row in plans[str(positive["query_id"])].requirements
        }
        negative_conditions = {
            (row.kind, row.value)
            for row in plans[str(negative["query_id"])].requirements
        }
        if not positive_conditions - negative_conditions or not (
            negative_conditions - positive_conditions
        ):
            failed_changed_detection.append(group_id)
    return {
        "query_count": len(queries),
        "group_count": len(groups),
        "compositional_query_count": sum(
            plan.compositional for plan in plans.values()
        ),
        "minimum_requirement_count": min(requirement_counts),
        "maximum_requirement_count": max(requirement_counts),
        "mean_requirement_count": round(statistics.mean(requirement_counts), 8),
        "requirement_kind_counts": dict(sorted(kind_counts.items())),
        "changed_condition_detected_group_count": len(groups)
        - len(failed_changed_detection),
        "failed_changed_condition_group_ids": failed_changed_detection,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--scope-receipt", type=Path, required=True)
    parser.add_argument(
        "--policy",
        type=Path,
        default=PROJECT_ROOT / "config/v17_attribute_coverage.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    queries_path = args.queries.resolve()
    scope_path = args.scope_receipt.resolve()
    policy_path = args.policy.resolve()
    output_path = args.output.resolve()
    queries = read_jsonl(queries_path)
    scope = json.loads(scope_path.read_text(encoding="utf-8-sig"))
    if len(queries) != args.expected_count:
        raise ValueError(f"Expected {args.expected_count} calibration queries")
    if any(row.get("split") != "calibration" for row in queries):
        raise ValueError("Parser audit may load calibration queries only")
    fingerprint = query_set_fingerprint(queries)
    if scope.get("calibration_query_set_sha256") != fingerprint:
        raise ValueError("Scope query fingerprint mismatch")
    if scope.get("holdout_results_opened") is not False:
        raise ValueError("Holdout access invariant is not satisfied")
    policy = load_attribute_policy(policy_path)
    metrics = audit_plans(queries, policy)
    parser_source = PROJECT_ROOT / (
        "src/ocr_vlm_retrieval/gating/condition_decomposition.py"
    )
    receipt = {
        "status": "v18_calibration_parser_audited_not_method_locked",
        "parser_version": PARSER_VERSION,
        "metrics": metrics,
        "calibration_query_set_sha256": fingerprint,
        "queries_sha256": file_sha256(queries_path),
        "scope_receipt_sha256": file_sha256(scope_path),
        "policy_sha256": file_sha256(policy_path),
        "parser_source_sha256": file_sha256(parser_source),
        "relevance_judgments_read": False,
        "holdout_results_opened": False,
        "holdout_retrieval_executed": False,
        "v17_artifacts_modified": False,
    }
    write_json_atomic(output_path, receipt)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
