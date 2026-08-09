"""Select and lock one V18 L0-L3 configuration on calibration labels only."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from ocr_vlm_retrieval.gating.attribute_coverage import (  # noqa: E402
    AttributePlan,
    AttributeRequirement,
)
from ocr_vlm_retrieval.gating.listwise_selection import (  # noqa: E402
    CandidateEvidence,
    select_candidate,
)
from scripts.run_v17_calibration_retrieval import (  # noqa: E402
    file_sha256,
    read_jsonl,
    write_json_atomic,
)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def expand_method_parameters(
    method_id: str, definition: Mapping[str, Any]
) -> list[dict[str, Any]]:
    if method_id == "L0":
        return [dict(definition["parameters"])]
    grid = {key: list(values) for key, values in definition.get("grid", {}).items()}
    fixed = dict(definition.get("fixed", {}))
    paired_groups = [tuple(group) for group in definition.get("paired_grid_fields", [])]
    paired_options: list[list[dict[str, Any]]] = []
    paired_fields: set[str] = set()
    for group in paired_groups:
        if any(field in paired_fields or field not in grid for field in group):
            raise ValueError(f"Invalid paired grid fields for {method_id}: {group}")
        lengths = {len(grid[field]) for field in group}
        if len(lengths) != 1:
            raise ValueError(f"Paired grid lengths differ for {method_id}: {group}")
        paired_fields.update(group)
        paired_options.append(
            [dict(zip(group, values, strict=True)) for values in zip(*(grid[x] for x in group), strict=True)]
        )
    independent_fields = sorted(set(grid) - paired_fields)
    independent_options = [
        dict(zip(independent_fields, values, strict=True))
        for values in itertools.product(*(grid[field] for field in independent_fields))
    ] or [{}]
    paired_product = list(itertools.product(*paired_options)) if paired_options else [()]
    parameters: list[dict[str, Any]] = []
    for independent in independent_options:
        for paired_parts in paired_product:
            row = {**fixed, **independent}
            for paired in paired_parts:
                row.update(paired)
            parameters.append(row)
    return parameters


def attribute_plan_from_dict(source: Mapping[str, Any]) -> AttributePlan:
    requirements = tuple(
        AttributeRequirement(
            requirement_id=str(row["requirement_id"]),
            kind=str(row["kind"]),
            value=str(row["value"]),
            prompt=str(row["prompt"]),
            threshold=float(row["threshold"]),
            full_score=float(row["full_score"]),
            weight=float(row.get("weight", 1.0)),
            mandatory=bool(row.get("mandatory", True)),
        )
        for row in source.get("requirements", [])
    )
    return AttributePlan(
        query=str(source["query"]),
        requirements=requirements,
        compositional=bool(source["compositional"]),
        parser_version=int(source["parser_version"]),
        fingerprint=str(source["fingerprint"]),
    )


def verification_index(
    verification: Mapping[str, Any],
) -> dict[str, tuple[AttributePlan, list[CandidateEvidence]]]:
    if verification.get("status") != "complete":
        raise ValueError("V18 verification must be complete before calibration")
    if verification.get("scope") != "v18_calibration_base_top10_candidates_only":
        raise ValueError("Verification scope is not V18 calibration-only")
    if verification.get("judgments_read") is not False:
        raise ValueError("Verification must attest that judgments were not read")
    indexed: dict[str, tuple[AttributePlan, list[CandidateEvidence]]] = {}
    for result in verification.get("results", []):
        query_id = str(result.get("query_id", "")).strip()
        if not query_id or query_id in indexed:
            raise ValueError("Verification query IDs must be unique")
        plan = attribute_plan_from_dict(result["attribute_plan"])
        candidates = []
        for source in result.get("candidates", []):
            margins = {
                str(row["requirement_id"]): float(row["margin"])
                for row in source.get("contrastive_relation_evidence", [])
            }
            candidates.append(
                CandidateEvidence(
                    item_id=str(source["item_id"]),
                    retrieval_rank=int(source["retrieval_rank"]),
                    retrieval_score=float(source["ranking_score"]),
                    full_query_score=float(source["full_query_score"]),
                    resolved_requirement_scores={
                        str(key): float(value)
                        for key, value in source.get(
                            "resolved_requirement_scores", {}
                        ).items()
                    },
                    contrastive_relation_margins=margins,
                )
            )
        indexed[query_id] = (plan, candidates)
    return indexed


def judgment_index(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, bool]]:
    indexed: dict[str, dict[str, bool]] = {}
    for row in rows:
        query_id = str(row.get("query_id", "")).strip()
        if not query_id or query_id in indexed:
            raise ValueError("Calibration needs one final judgment per query")
        relevance = row.get("candidate_relevance")
        if not isinstance(relevance, dict):
            raise ValueError(f"Missing candidate relevance for {query_id}")
        indexed[query_id] = {
            str(item_id): bool(value) for item_id, value in relevance.items()
        }
    return indexed


def evaluate_configuration(
    method_id: str,
    parameters: Mapping[str, Any],
    queries: Sequence[Mapping[str, Any]],
    verification: Mapping[str, tuple[AttributePlan, Sequence[CandidateEvidence]]],
    judgments: Mapping[str, Mapping[str, bool]],
) -> dict[str, Any]:
    outcomes: list[dict[str, Any]] = []
    for query in queries:
        query_id = str(query["query_id"])
        plan, candidates = verification[query_id]
        decision = select_candidate(
            method_id,
            candidates,
            parameters,
            attribute_plan=plan if method_id == "L3" else None,
        )
        relevance = judgments[query_id]
        answerable_in_pool = any(relevance.values())
        selected_relevant = bool(
            decision.accepted
            and decision.selected_item_id is not None
            and relevance.get(decision.selected_item_id, False)
        )
        correct = selected_relevant if decision.accepted else not answerable_in_pool
        selected_rank = next(
            (
                row.retrieval_rank
                for row in candidates
                if row.item_id == decision.selected_item_id
            ),
            None,
        )
        outcomes.append(
            {
                "query_id": query_id,
                "group_id": str(query["group_id"]),
                "query_role": str(query["query_role"]),
                "answerable_in_pool": answerable_in_pool,
                "accepted": decision.accepted,
                "selected_relevant": selected_relevant,
                "selected_rank": selected_rank,
                "correct": correct,
                "false_accept": decision.accepted and not selected_relevant,
                "false_reject": not decision.accepted and answerable_in_pool,
                "decision": decision.to_dict(),
            }
        )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in outcomes:
        grouped[str(row["group_id"])].append(row)
    hard_no_answer = [
        row
        for row in outcomes
        if row["query_role"] == "single_condition_hard_negative"
        and not row["answerable_in_pool"]
    ]
    positives = [row for row in outcomes if row["query_role"] == "positive"]
    metrics = {
        "query_count": len(outcomes),
        "end_to_end_accuracy": sum(row["correct"] for row in outcomes)
        / len(outcomes),
        "grouped_pair_accuracy": sum(
            all(row["correct"] for row in rows) for rows in grouped.values()
        )
        / len(grouped),
        "false_accept_rate": sum(row["false_accept"] for row in outcomes)
        / len(outcomes),
        "false_reject_rate": sum(row["false_reject"] for row in outcomes)
        / len(outcomes),
        "hard_negative_no_answer_count": len(hard_no_answer),
        "hard_negative_no_answer_false_accept_rate": (
            sum(row["false_accept"] for row in hard_no_answer)
            / len(hard_no_answer)
            if hard_no_answer
            else 0.0
        ),
        "positive_top3_conversion": sum(
            bool(row["selected_relevant"] and (row["selected_rank"] or 99) <= 3)
            for row in positives
        )
        / len(positives),
        "hard_negative_pool_collision_count": sum(
            row["query_role"] == "single_condition_hard_negative"
            and row["answerable_in_pool"]
            for row in outcomes
        ),
    }
    return {
        "method_id": method_id,
        "parameters": dict(parameters),
        "metrics": {key: round(value, 8) if isinstance(value, float) else value for key, value in metrics.items()},
        "outcomes": outcomes,
    }


def configuration_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    metrics = row["metrics"]
    parameters = row["parameters"]
    return (
        -float(metrics["grouped_pair_accuracy"]),
        float(metrics["hard_negative_no_answer_false_accept_rate"]),
        -float(metrics["positive_top3_conversion"]),
        int(parameters.get("top_k", 999)),
        -float(parameters.get("selection_margin_threshold", 0.0)),
        str(row["method_id"]),
        json.dumps(parameters, sort_keys=True),
    )


def select_locked_configuration(
    evaluations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    baseline = next(row for row in evaluations if row["method_id"] == "L0")
    baseline_metrics = baseline["metrics"]
    eligible = []
    for source in evaluations:
        row = dict(source)
        metrics = row["metrics"]
        constraints = {
            "hard_negative_false_accept_not_above_L0": float(
                metrics["hard_negative_no_answer_false_accept_rate"]
            )
            <= float(
                baseline_metrics["hard_negative_no_answer_false_accept_rate"]
            ),
            "positive_top3_conversion_difference_at_least_minus_0_03": float(
                metrics["positive_top3_conversion"]
            )
            >= float(baseline_metrics["positive_top3_conversion"]) - 0.03,
        }
        row["constraints"] = constraints
        row["eligible"] = all(constraints.values())
        if row["eligible"]:
            eligible.append(row)
    if not eligible:
        raise ValueError("No V18 configuration satisfies the locked constraints")
    return min(eligible, key=configuration_sort_key)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument(
        "--methods",
        type=Path,
        default=PROJECT_ROOT / "config/studies/v18_methods.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = {
        "queries": args.queries.resolve(),
        "judgments": args.judgments.resolve(),
        "verification": args.verification.resolve(),
        "methods": args.methods.resolve(),
        "output": args.output.resolve(),
    }
    queries = read_jsonl(paths["queries"])
    if any(row.get("split") != "calibration" for row in queries):
        raise ValueError("V18 calibration may not load holdout queries")
    judgments = judgment_index(read_jsonl(paths["judgments"]))
    verification_payload = read_json(paths["verification"])
    verification = verification_index(verification_payload)
    query_ids = {str(row["query_id"]) for row in queries}
    if set(judgments) != query_ids or set(verification) != query_ids:
        raise ValueError("Queries, judgments, and verification IDs do not match")
    methods = read_json(paths["methods"])
    evaluations = [
        evaluate_configuration(
            method_id,
            parameters,
            queries,
            verification,
            judgments,
        )
        for method_id, definition in methods["methods"].items()
        for parameters in expand_method_parameters(method_id, definition)
    ]
    selected = select_locked_configuration(evaluations)
    summary = sorted(
        (
            {
                "method_id": row["method_id"],
                "parameters": row["parameters"],
                "metrics": row["metrics"],
            }
            for row in evaluations
        ),
        key=configuration_sort_key,
    )
    receipt = {
        "status": "v18_method_locked_holdout_not_opened",
        "selected_method_id": selected["method_id"],
        "selected_parameters": selected["parameters"],
        "selected_metrics": selected["metrics"],
        "selected_constraints": selected["constraints"],
        "configuration_count": len(evaluations),
        "leaderboard": summary,
        "queries_sha256": file_sha256(paths["queries"]),
        "judgments_sha256": file_sha256(paths["judgments"]),
        "verification_sha256": file_sha256(paths["verification"]),
        "methods_sha256": file_sha256(paths["methods"]),
        "holdout_results_opened": False,
        "holdout_retrieval_executed": False,
        "v17_artifacts_modified": False,
    }
    write_json_atomic(paths["output"], receipt)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
