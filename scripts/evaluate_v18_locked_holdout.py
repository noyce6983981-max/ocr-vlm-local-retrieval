"""Evaluate locked V18 L1/L2 against L0 on the one-shot human holdout."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from ocr_vlm_retrieval.gating.listwise_selection import (  # noqa: E402
    CandidateEvidence,
    select_candidate,
)
from scripts.calibrate_v18_listwise import judgment_index  # noqa: E402
from scripts.run_v17_calibration_retrieval import (  # noqa: E402
    file_sha256,
    read_jsonl,
    write_json_atomic,
)
from scripts.score_v18_candidate_verification import read_json  # noqa: E402


def verification_index(
    payload: Mapping[str, Any],
) -> dict[str, list[CandidateEvidence]]:
    if payload.get("status") != "complete":
        raise ValueError("V18 holdout verification is incomplete")
    if payload.get("scope") != "v18_holdout_locked_top_k_candidates_only":
        raise ValueError("V18 holdout verification scope is invalid")
    if payload.get("judgments_read") is not False:
        raise ValueError("V18 holdout verification read judgments")
    indexed: dict[str, list[CandidateEvidence]] = {}
    for result in payload.get("results", []):
        query_id = str(result.get("query_id", "")).strip()
        if not query_id or query_id in indexed:
            raise ValueError("Holdout verification query IDs must be unique")
        indexed[query_id] = [
            CandidateEvidence(
                item_id=str(row["item_id"]),
                retrieval_rank=int(row["retrieval_rank"]),
                retrieval_score=float(row["ranking_score"]),
                full_query_score=float(row["full_query_score"]),
                resolved_requirement_scores={},
                contrastive_relation_margins={},
            )
            for row in result.get("candidates", [])
        ]
    return indexed


def evaluate(
    method_id: str,
    parameters: Mapping[str, Any],
    queries: Sequence[Mapping[str, Any]],
    verification: Mapping[str, Sequence[CandidateEvidence]],
    judgments: Mapping[str, Mapping[str, bool]],
) -> dict[str, Any]:
    outcomes = []
    for query in queries:
        query_id = str(query["query_id"])
        candidates = verification[query_id]
        decision = select_candidate(method_id, candidates, parameters)
        relevance = judgments[query_id]
        answerable = any(relevance.values())
        selected_relevant = bool(
            decision.accepted
            and decision.selected_item_id is not None
            and relevance.get(decision.selected_item_id, False)
        )
        correct = selected_relevant if decision.accepted else not answerable
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
                "answerable_in_pool": answerable,
                "accepted": decision.accepted,
                "selected_relevant": selected_relevant,
                "selected_rank": selected_rank,
                "correct": correct,
                "false_accept": decision.accepted and not selected_relevant,
                "false_reject": not decision.accepted and answerable,
                "decision": decision.to_dict(),
            }
        )
    return {
        "method_id": method_id,
        "parameters": dict(parameters),
        "metrics": metrics_for_outcomes(outcomes),
        "outcomes": outcomes,
    }


def metrics_for_outcomes(outcomes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in outcomes:
        grouped[str(row["group_id"])].append(row)
    hard_no_answer = [
        row
        for row in outcomes
        if row["query_role"] == "single_condition_hard_negative"
        and not row["answerable_in_pool"]
    ]
    positives = [row for row in outcomes if row["query_role"] == "positive"]
    return {
        "query_count": len(outcomes),
        "group_count": len(grouped),
        "end_to_end_accuracy": sum(bool(row["correct"]) for row in outcomes)
        / len(outcomes),
        "grouped_pair_accuracy": sum(
            all(bool(row["correct"]) for row in rows) for rows in grouped.values()
        )
        / len(grouped),
        "false_accept_rate": sum(bool(row["false_accept"]) for row in outcomes)
        / len(outcomes),
        "false_reject_rate": sum(bool(row["false_reject"]) for row in outcomes)
        / len(outcomes),
        "hard_negative_no_answer_count": len(hard_no_answer),
        "hard_negative_no_answer_false_accept_rate": (
            sum(bool(row["false_accept"]) for row in hard_no_answer)
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


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def paired_group_bootstrap(
    baseline_outcomes: Sequence[Mapping[str, Any]],
    selected_outcomes: Sequence[Mapping[str, Any]],
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    if repetitions < 1:
        raise ValueError("Bootstrap repetitions must be positive")
    baseline_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    selected_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in baseline_outcomes:
        baseline_groups[str(row["group_id"])].append(row)
    for row in selected_outcomes:
        selected_groups[str(row["group_id"])].append(row)
    if set(baseline_groups) != set(selected_groups):
        raise ValueError("Bootstrap method groups do not match")
    group_ids = sorted(baseline_groups)
    rng = random.Random(seed)
    metric_names = (
        "end_to_end_accuracy",
        "grouped_pair_accuracy",
        "hard_negative_no_answer_false_accept_rate",
        "positive_top3_conversion",
    )
    deltas = {name: [] for name in metric_names}
    for _ in range(repetitions):
        sampled = [rng.choice(group_ids) for _ in group_ids]
        baseline_sample = [row for group in sampled for row in baseline_groups[group]]
        selected_sample = [row for group in sampled for row in selected_groups[group]]
        baseline_metrics = metrics_for_outcomes(baseline_sample)
        selected_metrics = metrics_for_outcomes(selected_sample)
        for name in metric_names:
            deltas[name].append(
                float(selected_metrics[name]) - float(baseline_metrics[name])
            )
    return {
        name: {
            "ci95": [
                round(_quantile(values, 0.025), 8),
                round(_quantile(values, 0.975), 8),
            ],
            "probability_delta_above_zero": round(
                sum(value > 0 for value in values) / repetitions, 8
            ),
        }
        for name, values in deltas.items()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--method-lock", type=Path, required=True)
    parser.add_argument("--methods", type=Path, required=True)
    parser.add_argument("--claim-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = {
        key: Path(value).resolve()
        for key, value in vars(args).items()
        if isinstance(value, Path)
    }
    queries = read_jsonl(paths["queries"])
    if len(queries) != 80 or any(row.get("split") != "holdout" for row in queries):
        raise ValueError("V18 locked evaluation requires exactly 80 holdout queries")
    judgments = judgment_index(read_jsonl(paths["judgments"]))
    verification_payload = read_json(paths["verification"])
    verification = verification_index(verification_payload)
    query_ids = {str(row["query_id"]) for row in queries}
    if set(judgments) != query_ids or set(verification) != query_ids:
        raise ValueError("V18 holdout query, judgment, and verification IDs differ")
    method_lock = read_json(paths["method_lock"])
    methods = read_json(paths["methods"])
    if method_lock.get("status") != "v18_method_locked_holdout_not_opened":
        raise ValueError("V18 holdout method lock is invalid")
    if method_lock.get("methods_sha256") != file_sha256(paths["methods"]):
        raise ValueError("V18 holdout method grid hash mismatch")
    if verification_payload.get("method_lock_sha256") != file_sha256(
        paths["method_lock"]
    ):
        raise ValueError("V18 holdout verification uses another method lock")
    selected_id = str(method_lock["selected_method_id"])
    if selected_id not in {"L1", "L2"}:
        raise ValueError("V18 holdout evaluator supports locked L1/L2 only")
    baseline = evaluate(
        "L0", methods["methods"]["L0"]["parameters"], queries, verification, judgments
    )
    selected = evaluate(
        selected_id,
        method_lock["selected_parameters"],
        queries,
        verification,
        judgments,
    )
    bootstrap = paired_group_bootstrap(
        baseline["outcomes"],
        selected["outcomes"],
        repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )
    baseline_metrics = baseline["metrics"]
    selected_metrics = selected["metrics"]
    release_constraints = {
        "grouped_pair_accuracy_not_below_L0": selected_metrics[
            "grouped_pair_accuracy"
        ]
        >= baseline_metrics["grouped_pair_accuracy"],
        "end_to_end_accuracy_above_L0": selected_metrics["end_to_end_accuracy"]
        > baseline_metrics["end_to_end_accuracy"],
        "hard_negative_false_accept_not_above_L0": selected_metrics[
            "hard_negative_no_answer_false_accept_rate"
        ]
        <= baseline_metrics["hard_negative_no_answer_false_accept_rate"],
        "positive_top3_conversion_difference_at_least_minus_0_03": selected_metrics[
            "positive_top3_conversion"
        ]
        >= baseline_metrics["positive_top3_conversion"] - 0.03,
    }
    report = {
        "status": "v18_one_shot_holdout_evaluation_complete",
        "baseline": baseline,
        "selected": selected,
        "metric_deltas": {
            key: round(float(selected_metrics[key]) - float(baseline_metrics[key]), 8)
            for key in (
                "end_to_end_accuracy",
                "grouped_pair_accuracy",
                "hard_negative_no_answer_false_accept_rate",
                "positive_top3_conversion",
            )
        },
        "paired_group_bootstrap": {
            "repetitions": args.bootstrap_repetitions,
            "seed": args.seed,
            "deltas": bootstrap,
        },
        "release_constraints": release_constraints,
        "release_recommended": all(release_constraints.values()),
        "queries_sha256": file_sha256(paths["queries"]),
        "judgments_sha256": file_sha256(paths["judgments"]),
        "verification_sha256": file_sha256(paths["verification"]),
        "method_lock_sha256": file_sha256(paths["method_lock"]),
        "methods_sha256": file_sha256(paths["methods"]),
        "claim_receipt_sha256": file_sha256(paths["claim_receipt"]),
        "holdout_results_opened": True,
        "holdout_evaluation_runs": 1,
        "v17_artifacts_modified": False,
    }
    write_json_atomic(paths["output"], report)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "metric_deltas",
                    "release_constraints",
                    "release_recommended",
                )
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
