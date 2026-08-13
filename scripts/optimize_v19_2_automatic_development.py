"""Optimize V19.2 condition aggregation on machine-only development evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ocr_vlm_retrieval.gating.candidate_verification import (  # noqa: E402
    load_ocr_lines,
)
from ocr_vlm_retrieval.gating.literal_evidence_v19_2 import (  # noqa: E402
    select_v19_2_literal_candidate,
)
from ocr_vlm_retrieval.runtime.cache import write_json_atomic  # noqa: E402
from scripts.evaluate_v19_ocr_literal_override import summarize_by_stratum  # noqa: E402
from scripts.run_v19_downstream_retrieval_pilot import read_json  # noqa: E402
from scripts.score_v19_v18_l1_development import summarize  # noqa: E402

EVALUATION_ROOT = ROOT / "outputs/evaluation/v19_2/automatic_optimization"
DEFAULT_ASSIGNMENTS = EVALUATION_ROOT / "development_assignments_machine.json"
DEFAULT_RETRIEVAL = EVALUATION_ROOT / "development_colqwen2_machine.json"
DEFAULT_CONFIG = ROOT / "config/studies/v19_2_automatic_optimization.json"
DEFAULT_OUTPUT = EVALUATION_ROOT / "development_v19_2_optimization.json"
DEFAULT_OCR_ROOT = ROOT / "outputs/user_library/ocr/json"
SPLIT = "v19_2_automatic_development_only"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--retrieval", type=Path, default=DEFAULT_RETRIEVAL)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--ocr-root", type=Path, default=DEFAULT_OCR_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ocr-min-confidence", type=float, default=0.35)
    return parser.parse_args()


def _with_recall(
    results: Sequence[Mapping[str, Any]],
    retrieval_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    summary: dict[str, Any] = dict(summarize(results))
    positives = [row for row in retrieval_results if bool(row["gold_answerable"])]
    hits = sum(
        bool(
            {str(value) for value in row.get("ranking_item_ids", [])[:3]}
            & {str(value) for value in row.get("gold_relevant_item_ids", [])}
        )
        for row in positives
    )
    summary["positive_recall_at_3"] = hits / len(positives) if positives else 0.0
    return summary


def _rate(rows: Sequence[Mapping[str, Any]], predicate: Any) -> float:
    return sum(bool(predicate(row)) for row in rows) / len(rows) if rows else 0.0


def _selected_relevant(row: Mapping[str, Any]) -> bool:
    return bool(row.get("accepted")) and str(row.get("selected_item_id")) in {
        str(value) for value in row.get("gold_relevant_item_ids", [])
    }


def extended_metrics(
    results: Sequence[Mapping[str, Any]],
    retrieval_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    metrics = _with_recall(results, retrieval_results)
    hard_negatives = [
        row
        for row in results
        if row.get("query_role") == "single_condition_hard_negative"
    ]
    topic_positives = [
        row
        for row in results
        if row.get("content_stratum") == "topic_discovery"
        and bool(row.get("gold_answerable"))
    ]
    metrics["hard_negative_far"] = _rate(
        hard_negatives, lambda row: bool(row.get("accepted"))
    )
    metrics["topic_positive_selected_relevant"] = _rate(
        topic_positives, _selected_relevant
    )
    return metrics


def stop_condition_audit(
    metrics: Mapping[str, Any], conditions: Mapping[str, Any]
) -> dict[str, Any]:
    checks = {
        "minimum_end_to_end_accuracy": float(metrics["end_to_end_accuracy"])
        >= float(conditions["minimum_end_to_end_accuracy"]),
        "minimum_topic_positive_selected_relevant": float(
            metrics["topic_positive_selected_relevant"]
        )
        >= float(conditions["minimum_topic_positive_selected_relevant"]),
        "maximum_hard_negative_far": float(metrics["hard_negative_far"])
        <= float(conditions["maximum_hard_negative_far"]),
        "minimum_positive_recall_at_3": float(metrics["positive_recall_at_3"])
        >= float(conditions["minimum_positive_recall_at_3"]),
    }
    return {"all_met": all(checks.values()), "checks": checks}


def evaluate_configuration(
    retrieval_results: Sequence[Mapping[str, Any]],
    lines_by_item: Mapping[str, Sequence[str]],
    *,
    top_k: int,
    fuzzy_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for retrieval in retrieval_results:
        item_ids = [
            str(item_id)
            for item_id in retrieval.get("ranking_item_ids", [])[:top_k]
        ]
        decision = select_v19_2_literal_candidate(
            str(retrieval["query"]),
            item_ids,
            lines_by_item,
            fuzzy_threshold=fuzzy_threshold,
        )
        selected = decision["selected_item_id"]
        results.append(
            {
                "query_id": retrieval["query_id"],
                "query_role": retrieval["query_role"],
                "content_stratum": retrieval["content_stratum"],
                "gold_answerable": bool(retrieval["gold_answerable"]),
                "gold_relevant_item_ids": list(
                    retrieval.get("gold_relevant_item_ids", [])
                ),
                "candidate_item_ids": item_ids,
                "selected_item_id": selected,
                "selected_retrieval_rank": (
                    item_ids.index(str(selected)) + 1 if selected is not None else None
                ),
                "accepted": bool(decision["accepted"]),
                "decision_source": "v19_2_complete_condition_aggregation",
            }
        )
        decisions.append(
            {
                "query_id": retrieval["query_id"],
                "query": retrieval["query"],
                "query_role": retrieval["query_role"],
                "gold_answerable": bool(retrieval["gold_answerable"]),
                **decision,
            }
        )
    return results, decisions


def truncate_decision(decision: Mapping[str, Any], *, top_k: int) -> dict[str, Any]:
    """Reuse evidence computed at maximum Top-K for a smaller candidate prefix."""

    evidence = [
        dict(row)
        for row in decision.get("candidate_evidence", [])
        if int(row["retrieval_rank"]) <= top_k
    ]
    selected = next(
        (
            str(row["item_id"])
            for row in evidence
            if bool(row.get("all_constraints_matched"))
        ),
        None,
    )
    return {
        **decision,
        "candidate_evidence": evidence,
        "accepted": selected is not None,
        "selected_item_id": selected,
        "selection_reason": (
            "all_v19_2_literal_constraints_matched"
            if selected is not None
            else "no_candidate_matches_complete_v19_2_contract"
        ),
    }


def results_from_decisions(
    retrieval_results: Sequence[Mapping[str, Any]],
    decisions: Sequence[Mapping[str, Any]],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    if len(retrieval_results) != len(decisions):
        raise ValueError("retrieval and decision counts differ")
    results: list[dict[str, Any]] = []
    for retrieval, decision in zip(retrieval_results, decisions, strict=True):
        item_ids = [
            str(item_id)
            for item_id in retrieval.get("ranking_item_ids", [])[:top_k]
        ]
        selected = decision.get("selected_item_id")
        results.append(
            {
                "query_id": retrieval["query_id"],
                "query_role": retrieval["query_role"],
                "content_stratum": retrieval["content_stratum"],
                "gold_answerable": bool(retrieval["gold_answerable"]),
                "gold_relevant_item_ids": list(
                    retrieval.get("gold_relevant_item_ids", [])
                ),
                "candidate_item_ids": item_ids,
                "selected_item_id": selected,
                "selected_retrieval_rank": (
                    item_ids.index(str(selected)) + 1 if selected is not None else None
                ),
                "accepted": bool(decision["accepted"]),
                "decision_source": "v19_2_complete_condition_aggregation",
            }
        )
    return results


def _selection_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    metrics = row["metrics"]
    return (
        not bool(row["stop_condition_audit"]["all_met"]),
        -float(metrics["end_to_end_accuracy"]),
        float(metrics["hard_negative_far"]),
        -float(metrics["positive_selected_relevant"]),
        int(row["parameters"]["top_k"]),
        -float(row["parameters"]["ocr_fuzzy_threshold"]),
    )


def main() -> int:
    args = parse_args()
    assignments = read_json(args.assignments)
    retrieval = read_json(args.retrieval)
    config = read_json(args.config)
    if assignments.get("split") != SPLIT or retrieval.get("split") != SPLIT:
        raise ValueError("optimization may read V19.2 automatic development only")
    if assignments.get("human_review_used") is not False:
        raise ValueError("automatic optimization must not use human review")
    assignment_sha = hashlib.sha256(args.assignments.read_bytes()).hexdigest()
    if retrieval.get("source_assignment_sha256") != assignment_sha:
        raise ValueError("retrieval and assignment hashes differ")
    retrieval_results = [dict(row) for row in retrieval.get("results", [])]
    if len(retrieval_results) != 48:
        raise ValueError("expected 48 automatic development retrieval rows")
    maximum_top_k = max(int(value) for value in config["optimization_grid"]["top_k"])
    candidate_ids = {
        str(item_id)
        for row in retrieval_results
        for item_id in row.get("ranking_item_ids", [])[:maximum_top_k]
    }
    lines_by_item = {
        item_id: load_ocr_lines(
            args.ocr_root / f"{item_id}.json",
            minimum_confidence=args.ocr_min_confidence,
        )
        for item_id in candidate_ids
    }
    conditions = config["automatic_stop_conditions"]
    trials: list[dict[str, Any]] = []
    trial_outputs: dict[tuple[int, float], tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
    decisions_by_threshold: dict[float, list[dict[str, Any]]] = {}
    for threshold in config["optimization_grid"]["ocr_fuzzy_threshold"]:
        _, decisions_by_threshold[float(threshold)] = evaluate_configuration(
            retrieval_results,
            lines_by_item,
            top_k=maximum_top_k,
            fuzzy_threshold=float(threshold),
        )
    for top_k in config["optimization_grid"]["top_k"]:
        for threshold in config["optimization_grid"]["ocr_fuzzy_threshold"]:
            top_k_value = int(top_k)
            threshold_value = float(threshold)
            parameters: dict[str, Any] = {
                "top_k": top_k_value,
                "ocr_fuzzy_threshold": threshold_value,
            }
            decisions = [
                truncate_decision(decision, top_k=top_k_value)
                for decision in decisions_by_threshold[threshold_value]
            ]
            results = results_from_decisions(
                retrieval_results, decisions, top_k=top_k_value
            )
            metrics = extended_metrics(results, retrieval_results)
            audit = stop_condition_audit(metrics, conditions)
            trials.append(
                {
                    "parameters": parameters,
                    "metrics": metrics,
                    "stop_condition_audit": audit,
                }
            )
            trial_outputs[(top_k_value, threshold_value)] = (
                results,
                decisions,
            )
    trials.sort(key=_selection_key)
    selected_trial = trials[0]
    selected_parameters = selected_trial["parameters"]
    selected_results, selected_decisions = trial_outputs[
        (
            int(selected_parameters["top_k"]),
            float(selected_parameters["ocr_fuzzy_threshold"]),
        )
    ]
    raw_results = []
    for row in retrieval_results:
        ranking = [str(value) for value in row.get("ranking_item_ids", [])]
        selected = ranking[0] if ranking else None
        raw_results.append(
            {
                **row,
                "candidate_item_ids": ranking[:1],
                "selected_item_id": selected,
                "accepted": selected is not None,
            }
        )
    result = {
        "schema_version": 1,
        "status": (
            "automatic_stop_conditions_met"
            if selected_trial["stop_condition_audit"]["all_met"]
            else "automatic_stop_conditions_not_met"
        ),
        "study_id": config["study_id"],
        "split": SPLIT,
        "eligible_for_final_claim": False,
        "human_review_used": False,
        "future_holdout_opened": False,
        "source_assignment_sha256": assignment_sha,
        "source_retrieval_sha256": hashlib.sha256(
            args.retrieval.read_bytes()
        ).hexdigest(),
        "runtime_inputs": ["query_text", "candidate_item_ids", "candidate_ocr"],
        "forbidden_runtime_inputs": [
            "content_stratum",
            "query_role",
            "gold_answerable",
            "gold_relevant_item_ids",
        ],
        "selection_rule": config["optimization_grid"]["selection_order"],
        "trial_count": len(trials),
        "selected_parameters": selected_parameters,
        "selected_metrics": selected_trial["metrics"],
        "selected_stop_condition_audit": selected_trial["stop_condition_audit"],
        "raw_colqwen2_top1": extended_metrics(raw_results, retrieval_results),
        "delta_vs_raw_colqwen2_top1": {
            key: round(
                float(selected_trial["metrics"][key])
                - float(extended_metrics(raw_results, retrieval_results)[key]),
                8,
            )
            for key in (
                "positive_selected_relevant",
                "negative_correct_reject_rate",
                "end_to_end_accuracy",
                "hard_negative_far",
            )
        },
        "selected_by_stratum": summarize_by_stratum(selected_results),
        "decision_reason_counts": dict(
            sorted(Counter(str(row["reason"]) for row in selected_decisions).items())
        ),
        "trials": trials,
        "selected_decisions": selected_decisions,
        "selected_results": selected_results,
        "claim_boundary": config["claim_boundary"],
    }
    write_json_atomic(args.output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "selected_parameters": selected_parameters,
                "selected_metrics": result["selected_metrics"],
                "raw_colqwen2_top1": result["raw_colqwen2_top1"],
                "delta_vs_raw_colqwen2_top1": result[
                    "delta_vs_raw_colqwen2_top1"
                ],
                "decision_reason_counts": result["decision_reason_counts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
