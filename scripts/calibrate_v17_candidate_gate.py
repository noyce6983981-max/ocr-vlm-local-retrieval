"""Select a non-degenerate V17 candidate-verification gate on calibration."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from ocr_vlm_retrieval.evaluation.human_evaluation import grouped_paired_bootstrap
from ocr_vlm_retrieval.evaluation.judgments import (
    NO_RELEVANT_CANDIDATE_IN_POOL,
    POOLED_RELEVANCE_TASK,
    RELEVANT_CANDIDATE_IN_POOL,
    normalize_pool_judgment,
)

AGGREGATORS = ("full_query", "attribute_mean", "attribute_geometric", "weakest")
METHOD_PRIORITY = {
    "attribute_geometric": 0,
    "attribute_mean": 1,
    "full_query": 2,
    "weakest": 3,
}


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
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


def write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
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


def keyed_rows(
    rows: Iterable[Mapping[str, Any]], *, label: str
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for source in rows:
        query_id = str(source.get("query_id", "")).strip()
        if not query_id:
            raise ValueError(f"Every {label} row needs query_id")
        if query_id in result:
            raise ValueError(f"Duplicate {label} row for {query_id}")
        result[query_id] = dict(source)
    return result


def mean(values: Iterable[float]) -> float:
    collected = list(values)
    return sum(collected) / len(collected) if collected else 0.0


def geometric_mean(values: Iterable[float]) -> float:
    collected = [max(float(value), 1e-9) for value in values]
    if not collected:
        return 0.0
    return math.exp(sum(math.log(value) for value in collected) / len(collected))


def auc(rows: list[Mapping[str, Any]], field: str) -> float | None:
    positives = [float(row[field]) for row in rows if row["top1_relevant"]]
    negatives = [float(row[field]) for row in rows if not row["top1_relevant"]]
    if not positives or not negatives:
        return None
    pair_score = sum(
        float(positive > negative) + 0.5 * float(positive == negative)
        for positive in positives
        for negative in negatives
    )
    return pair_score / (len(positives) * len(negatives))


def threshold_metrics(
    rows: list[Mapping[str, Any]], *, field: str, threshold: float
) -> dict[str, Any]:
    accepted = [float(row[field]) >= threshold for row in rows]
    relevant_in_pool = [
        row
        for row in rows
        if row["pool_relevance"] == RELEVANT_CANDIDATE_IN_POOL
    ]
    no_relevant_in_pool = [
        row
        for row in rows
        if row["pool_relevance"] == NO_RELEVANT_CANDIDATE_IN_POOL
    ]
    relevant_count = sum(bool(row["top1_relevant"]) for row in rows)
    accepted_relevant = sum(
        decision and bool(row["top1_relevant"])
        for decision, row in zip(accepted, rows, strict=True)
    )
    false_accepts = sum(
        decision and row["pool_relevance"] == NO_RELEVANT_CANDIDATE_IN_POOL
        for decision, row in zip(accepted, rows, strict=True)
    )
    correct = sum(
        (
            row["pool_relevance"] == RELEVANT_CANDIDATE_IN_POOL
            and decision
            and bool(row["top1_relevant"])
        )
        or (
            row["pool_relevance"] == NO_RELEVANT_CANDIDATE_IN_POOL
            and not decision
        )
        for decision, row in zip(accepted, rows, strict=True)
    )
    return {
        "threshold": round(threshold, 2),
        "accepted_query_count": sum(accepted),
        "coverage": sum(accepted) / len(rows),
        "relevant_in_pool_accepted_count": sum(
            decision
            for decision, row in zip(accepted, rows, strict=True)
            if row["pool_relevance"] == RELEVANT_CANDIDATE_IN_POOL
        ),
        "relevant_in_pool_acceptance_rate": sum(
            decision
            for decision, row in zip(accepted, rows, strict=True)
            if row["pool_relevance"] == RELEVANT_CANDIDATE_IN_POOL
        )
        / len(relevant_in_pool),
        "top1_relevant_accepted_count": accepted_relevant,
        "top1_relevant_acceptance_rate": accepted_relevant / relevant_count,
        "pool_conditioned_false_accept_count": false_accepts,
        "pool_conditioned_false_accept_rate": (
            false_accepts / len(no_relevant_in_pool)
        ),
        "no_relevant_in_pool_rejection_rate": (
            1.0 - false_accepts / len(no_relevant_in_pool)
        ),
        "end_to_end_correct_count": correct,
        "pool_conditioned_end_to_end_accuracy": correct / len(rows),
        "degenerate_reject_all": not any(accepted),
    }


def calibrate(
    *,
    verification: Mapping[str, Any],
    judgments: Iterable[Mapping[str, Any]],
    baseline_rows: Iterable[Mapping[str, Any]],
    min_far_relative_reduction: float = 0.30,
    min_relevant_acceptance: float = 0.30,
    bootstrap_repetitions: int = 10_000,
    seed: int = 17,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    if verification.get("status") != "complete":
        raise ValueError("Candidate verification must be complete")
    if verification.get("judgments_read") is not False:
        raise ValueError("Verification artifact must certify judgments_read=false")
    judgment_by_id = {
        query_id: normalize_pool_judgment(row)
        for query_id, row in keyed_rows(judgments, label="judgment").items()
    }
    baseline_by_id = keyed_rows(baseline_rows, label="baseline")
    verification_by_id = keyed_rows(
        verification.get("results", []), label="verification"
    )
    if set(judgment_by_id) != set(verification_by_id):
        raise ValueError("Judgment and verification query IDs do not match")
    evaluated_judgment_ids = {
        query_id
        for query_id, row in judgment_by_id.items()
        if row.get("pool_relevance") != "excluded"
    }
    if not evaluated_judgment_ids.issubset(baseline_by_id):
        raise ValueError("Baseline decisions do not cover every query")

    rows: list[dict[str, Any]] = []
    excluded: list[str] = []
    for query_id in sorted(judgment_by_id):
        judgment = judgment_by_id[query_id]
        pool_relevance = str(judgment["pool_relevance"])
        if pool_relevance == "excluded":
            excluded.append(query_id)
            continue
        if pool_relevance not in {
            RELEVANT_CANDIDATE_IN_POOL,
            NO_RELEVANT_CANDIDATE_IN_POOL,
        }:
            raise ValueError(f"Unsupported pool relevance for {query_id}")
        candidates = verification_by_id[query_id].get("candidates", [])
        if len(candidates) != 1:
            raise ValueError("Gate calibration requires Top-1 verification scores")
        candidate = candidates[0]
        requirement_scores = [
            float(value) for value in candidate.get("requirement_scores", {}).values()
        ]
        if not requirement_scores:
            raise ValueError(f"Verification {query_id} has no requirement scores")
        item_id = str(candidate.get("item_id", ""))
        relevance = judgment.get("candidate_relevance", {})
        row = {
            "query_id": query_id,
            "group_id": verification_by_id[query_id].get("group_id"),
            "task_id": POOLED_RELEVANCE_TASK,
            "pool_relevance": pool_relevance,
            "top1_item_id": item_id,
            "top1_relevant": bool(relevance.get(item_id, False)),
            "v16_accepted": bool(baseline_by_id[query_id]["v16_accepted"]),
            "baseline_v16_pool_conditioned_correct": float(
                baseline_by_id[query_id].get(
                    "v16_pool_conditioned_correct",
                    baseline_by_id[query_id].get("v16_correct", 0.0),
                )
            ),
            "full_query": float(candidate["full_query_score"]),
            "attribute_mean": mean(requirement_scores),
            "attribute_geometric": geometric_mean(requirement_scores),
            "weakest": min(requirement_scores),
        }
        if not row["group_id"]:
            raise ValueError(f"Verification {query_id} has no group_id")
        rows.append(row)

    no_relevant_in_pool_rows = [
        row
        for row in rows
        if row["pool_relevance"] == NO_RELEVANT_CANDIDATE_IN_POOL
    ]
    baseline_far = mean(
        float(row["v16_accepted"]) for row in no_relevant_in_pool_rows
    )
    method_reports: dict[str, Any] = {}
    eligible_selections: list[tuple[tuple[float, ...], str, dict[str, Any]]] = []
    for method in AGGREGATORS:
        curve: list[dict[str, Any]] = []
        for index in range(101):
            metrics = threshold_metrics(rows, field=method, threshold=index / 100)
            far = float(metrics["pool_conditioned_false_accept_rate"])
            relative_reduction = (
                (baseline_far - far) / baseline_far if baseline_far else None
            )
            metrics[
                "pool_conditioned_false_accept_relative_reduction_vs_v16"
            ] = relative_reduction
            metrics["eligible"] = bool(
                relative_reduction is not None
                and relative_reduction >= min_far_relative_reduction
                and metrics["top1_relevant_acceptance_rate"] >= min_relevant_acceptance
                and not metrics["degenerate_reject_all"]
            )
            curve.append(metrics)
        eligible = [point for point in curve if point["eligible"]]
        if not eligible:
            selected = None
        else:
            selected = min(
                eligible,
                key=lambda point: (
                    -float(point["pool_conditioned_end_to_end_accuracy"]),
                    -float(point["top1_relevant_acceptance_rate"]),
                    float(point["pool_conditioned_false_accept_rate"]),
                    float(point["threshold"]),
                ),
            )
            cross_method_key = (
                -float(selected["pool_conditioned_end_to_end_accuracy"]),
                -float(selected["top1_relevant_acceptance_rate"]),
                float(selected["pool_conditioned_false_accept_rate"]),
                float(METHOD_PRIORITY[method]),
            )
            eligible_selections.append((cross_method_key, method, selected))
        method_reports[method] = {
            "top1_relevance_auc": auc(rows, method),
            "selected_operating_point": selected,
            "risk_coverage_curve": curve,
        }
    if not eligible_selections:
        raise ValueError("No non-degenerate gate satisfies calibration constraints")
    _, selected_method, selected_metrics = min(
        eligible_selections, key=lambda row: row[0]
    )
    selected_threshold = float(selected_metrics["threshold"])

    decisions: list[dict[str, Any]] = []
    for row in rows:
        accepted = float(row[selected_method]) >= selected_threshold
        decisions.append(
            {
                **row,
                "selected_method": selected_method,
                "selected_score": row[selected_method],
                "selected_threshold": selected_threshold,
                "v17_accepted": accepted,
                "v16_pool_conditioned_false_accept": float(
                    row["pool_relevance"] == NO_RELEVANT_CANDIDATE_IN_POOL
                    and row["v16_accepted"]
                ),
                "v17_pool_conditioned_false_accept": float(
                    row["pool_relevance"] == NO_RELEVANT_CANDIDATE_IN_POOL
                    and accepted
                ),
                "v16_pool_conditioned_correct": row[
                    "baseline_v16_pool_conditioned_correct"
                ],
                "v17_pool_conditioned_correct": float(
                    (
                        row["pool_relevance"] == RELEVANT_CANDIDATE_IN_POOL
                        and accepted
                        and row["top1_relevant"]
                    )
                    or (
                        row["pool_relevance"] == NO_RELEVANT_CANDIDATE_IN_POOL
                        and not accepted
                    )
                ),
            }
        )
    selected_no_relevant = [
        row
        for row in decisions
        if row["pool_relevance"] == NO_RELEVANT_CANDIDATE_IN_POOL
    ]
    bootstrap = {
        "pool_conditioned_false_accept_v17_minus_v16": grouped_paired_bootstrap(
            selected_no_relevant,
            baseline_field="v16_pool_conditioned_false_accept",
            contender_field="v17_pool_conditioned_false_accept",
            repetitions=bootstrap_repetitions,
            seed=seed,
        ),
        "end_to_end_correct_v17_minus_v16": grouped_paired_bootstrap(
            decisions,
            baseline_field="v16_pool_conditioned_correct",
            contender_field="v17_pool_conditioned_correct",
            repetitions=bootstrap_repetitions,
            seed=seed,
        ),
    }
    selected_gate = {
        "status": "selected_on_model_assisted_calibration_not_holdout_locked",
        "method": selected_method,
        "threshold": selected_threshold,
        "top_k_verified": 1,
        "model": verification.get("model"),
        "parser_policy_sha256": verification.get("policy_sha256"),
        "verification_ranking_sha256": verification.get("ranking_sha256"),
        "selection_constraints": {
            "minimum_pool_conditioned_false_accept_relative_reduction": (
                min_far_relative_reduction
            ),
            "minimum_top1_relevant_acceptance": min_relevant_acceptance,
            "reject_all_forbidden": True,
        },
        "selection_rule": (
            "Among 0.01-spaced thresholds satisfying the constraints, maximize "
            "end-to-end accuracy, then relevant-candidate acceptance, then minimize "
            "pool-conditioned false acceptance; prefer geometric evidence when "
            "methods tie."
        ),
    }
    report = {
        "status": "calibration_complete_holdout_not_run",
        "task_id": POOLED_RELEVANCE_TASK,
        "label_provenance": {
            "type": "model_assisted_visual_audit",
            "human_gold": False,
        },
        "scope": {
            "evaluated_query_count": len(rows),
            "excluded_query_ids": excluded,
            "holdout_read": False,
            "corpus_answerability_evaluated": False,
        },
        "baseline_v16_pool_conditioned_false_accept_rate": baseline_far,
        "aggregator_comparison": method_reports,
        "selected_gate": selected_gate,
        "selected_operating_point": selected_metrics,
        "paired_group_bootstrap": bootstrap,
        "limitations": [
            "The selected threshold was tuned on model-assisted calibration labels.",
            "The verifier still confuses some directional relation counterfactuals.",
            "Confidence intervals reflect only 39 evaluated calibration queries.",
        ],
    }
    return report, decisions, selected_gate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verification",
        type=Path,
        default=Path(
            "outputs/evaluation/v17/calibration/parser_v3/"
            "candidate_attribute_verification.json"
        ),
    )
    parser.add_argument(
        "--judgments",
        type=Path,
        default=Path(
            "data/evaluation/v17/human_study/calibration/adjudicated_judgments.jsonl"
        ),
    )
    parser.add_argument(
        "--baseline-records",
        type=Path,
        default=Path(
            "outputs/evaluation/v17/calibration/parser_v3/"
            "adjudicated_paired_records.jsonl"
        ),
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/evaluation/v17/calibration/parser_v3/"
            "candidate_gate_calibration_report.json"
        ),
    )
    parser.add_argument(
        "--decisions-output",
        type=Path,
        default=Path(
            "outputs/evaluation/v17/calibration/parser_v3/"
            "candidate_gate_paired_decisions.jsonl"
        ),
    )
    parser.add_argument(
        "--selected-gate-output",
        type=Path,
        default=Path("config/v17_candidate_verification_gate.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verification = json.loads(
        project_path(args.verification).read_text(encoding="utf-8")
    )
    report, decisions, selected_gate = calibrate(
        verification=verification,
        judgments=read_jsonl(project_path(args.judgments)),
        baseline_rows=read_jsonl(project_path(args.baseline_records)),
        bootstrap_repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )
    output = project_path(args.output)
    decisions_output = project_path(args.decisions_output)
    selected_gate_output = project_path(args.selected_gate_output)
    write_json_atomic(output, report)
    write_jsonl_atomic(decisions_output, decisions)
    write_json_atomic(selected_gate_output, selected_gate)
    print(output)
    print(decisions_output)
    print(selected_gate_output)


if __name__ == "__main__":
    main()
