"""Evaluate a conservative V19 candidate on development artifacts only."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ocr_vlm_retrieval.gating.selective_intervention import (  # noqa: E402
    SelectiveInterventionPolicy,
    admit_guarded_intervention,
)
from ocr_vlm_retrieval.runtime.cache import write_json_atomic  # noqa: E402

DEFAULT_ASSIGNMENTS = (
    ROOT
    / "outputs/evaluation/v19/selective_intervention"
    / "development_route_assignments.json"
)
DEFAULT_BASELINE = (
    ROOT
    / "outputs/evaluation/v19/selective_intervention"
    / "development_true_v18_l1_top3.json"
)
DEFAULT_LATE_INTERACTION = (
    ROOT
    / "outputs/evaluation/v19/selective_intervention"
    / "development_late_interaction_attributes.json"
)
DEFAULT_GUARDED_DIR = (
    ROOT / "outputs/evaluation/v19/selective_intervention/retrieval/guarded"
)
DEFAULT_POLICY = ROOT / "config/studies/v19_guarded_intervention_development.json"
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/evaluation/v19/selective_intervention"
    / "development_guarded_intervention_v2.json"
)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def keyed_rows(
    rows: Sequence[Mapping[str, Any]], label: str
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for source in rows:
        query_id = str(source.get("query_id", "")).strip()
        if not query_id or query_id in result:
            raise ValueError(f"{label} query IDs must be non-empty and unique")
        result[query_id] = dict(source)
    return result


def policy_from_config(config: Mapping[str, Any]) -> SelectiveInterventionPolicy:
    if config.get("status") != "development_candidate_not_frozen":
        raise ValueError("V19 intervention config must remain development-only")
    return SelectiveInterventionPolicy(**dict(config["intervention_policy"]))


def ordered_guarded_candidates(
    guarded_payload: Mapping[str, Any], late_result: Mapping[str, Any]
) -> list[dict[str, Any]]:
    scored = {
        str(row["item_id"]): dict(row) for row in late_result.get("candidates", [])
    }
    rankings = guarded_payload.get("rankings", {})
    ranking_rows = (
        rankings.get("quality_hybrid", []) if isinstance(rankings, Mapping) else []
    )
    candidates: list[dict[str, Any]] = []
    for rank, ranking in enumerate(ranking_rows[:3], start=1):
        item_id = str(ranking.get("item_id", ""))
        if item_id not in scored:
            raise ValueError(f"missing late-interaction score for {item_id}")
        candidates.append({**scored[item_id], "retrieval_rank": rank})
    return candidates


def outcome_correct(row: Mapping[str, Any]) -> bool:
    if not bool(row.get("gold_answerable")):
        return not bool(row.get("accepted"))
    return bool(row.get("accepted")) and str(row.get("selected_item_id")) in {
        str(item_id) for item_id in row.get("gold_relevant_item_ids", [])
    }


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    positives = [row for row in rows if bool(row.get("gold_answerable"))]
    negatives = [row for row in rows if not bool(row.get("gold_answerable"))]
    hard = [
        row
        for row in negatives
        if row.get("query_role") == "single_condition_hard_negative"
    ]
    neighbor = [
        row for row in negatives if row.get("query_role") == "unanswerable_neighbor"
    ]

    def rate(numerator: int, denominator: int) -> float:
        return round(numerator / denominator, 8) if denominator else 0.0

    selected_relevant = sum(outcome_correct(row) for row in positives)
    false_accepts = sum(bool(row.get("accepted")) for row in negatives)
    return {
        "query_count": len(rows),
        "answerable_count": len(positives),
        "negative_count": len(negatives),
        "positive_selected_relevant": rate(selected_relevant, len(positives)),
        "negative_correct_reject_rate": rate(
            len(negatives) - false_accepts, len(negatives)
        ),
        "end_to_end_accuracy": rate(
            sum(outcome_correct(row) for row in rows), len(rows)
        ),
        "negative_far": rate(false_accepts, len(negatives)),
        "hard_negative_far": rate(
            sum(bool(row.get("accepted")) for row in hard), len(hard)
        ),
        "neighbor_negative_far": rate(
            sum(bool(row.get("accepted")) for row in neighbor), len(neighbor)
        ),
        "false_reject_rate": rate(
            sum(not bool(row.get("accepted")) for row in positives),
            len(positives),
        ),
    }


def evaluate(
    assignments_payload: Mapping[str, Any],
    baseline_payload: Mapping[str, Any],
    late_payload: Mapping[str, Any],
    guarded_payloads: Mapping[str, Mapping[str, Any]],
    policy: SelectiveInterventionPolicy,
) -> dict[str, Any]:
    if assignments_payload.get("split") != "v19_reviewed_development_only":
        raise ValueError("V19 evaluation may read development assignments only")
    if late_payload.get("split") != "development_only":
        raise ValueError("late-interaction artifact is not development-only")
    if baseline_payload.get("method") != "frozen_v18_L1_max_verifier_score":
        raise ValueError("baseline artifact is not the frozen V18 L1 method")
    assignments = keyed_rows(assignments_payload.get("assignments", []), "assignment")
    baselines = keyed_rows(baseline_payload.get("results", []), "baseline")
    late = keyed_rows(late_payload.get("results", []), "late interaction")
    if set(assignments) != set(baselines):
        raise ValueError("assignment and baseline query IDs do not match")
    if set(late) != set(guarded_payloads):
        raise ValueError("late-interaction and guarded query IDs do not match")
    if not set(late).issubset(assignments):
        raise ValueError("late-interaction contains an unknown query")

    baseline_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    decision_reasons: Counter[str] = Counter()
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for query_id, assignment in assignments.items():
        baseline = baselines[query_id]
        common = {
            "query_id": query_id,
            "query_role": assignment.get("query_role"),
            "content_stratum": assignment.get("content_stratum"),
            "gold_answerable": bool(baseline.get("gold_answerable")),
            "gold_relevant_item_ids": list(baseline.get("gold_relevant_item_ids", [])),
        }
        baseline_row = {
            **common,
            "accepted": bool(baseline.get("accepted")),
            "selected_item_id": baseline.get("selected_item_id"),
        }
        baseline_rows.append(baseline_row)
        if query_id in late:
            decision = admit_guarded_intervention(
                baseline,
                ordered_guarded_candidates(guarded_payloads[query_id], late[query_id]),
                late[query_id]["attribute_plan"],
                policy,
            )
        else:
            decision = admit_guarded_intervention(baseline, [], {}, policy)
        decision_reasons[decision.reason] += 1
        candidate_row = {
            **common,
            "accepted": decision.accepted,
            "selected_item_id": decision.selected_item_id,
            "intervention": decision.to_dict(),
        }
        candidate_rows.append(candidate_row)
        by_stratum[str(common["content_stratum"])].append(candidate_row)

    baseline_metrics = summarize(baseline_rows)
    candidate_metrics = summarize(candidate_rows)
    delta = {
        key: round(float(candidate_metrics[key]) - float(baseline_metrics[key]), 8)
        for key in baseline_metrics
        if key not in {"query_count", "answerable_count", "negative_count"}
    }
    promotion = {
        "end_to_end_gain_at_least_0_05": delta["end_to_end_accuracy"] >= 0.05,
        "negative_far_not_increased": delta["negative_far"] <= 0.0,
        "hard_negative_far_not_increased": delta["hard_negative_far"] <= 0.0,
    }
    return {
        "study_id": "v19-selective-intervention-development-v2",
        "status": "development_evaluated_holdout_untouched",
        "eligible_for_final_claim": False,
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "delta": delta,
        "intervention_count": sum(
            bool(row["intervention"]["intervention_applied"]) for row in candidate_rows
        ),
        "decision_reason_counts": dict(sorted(decision_reasons.items())),
        "promotion_checks_available_without_latency": promotion,
        "automated_decision": "continue_development"
        if not all(promotion.values())
        else "latency_check_required",
        "per_stratum_candidate": {
            key: summarize(rows) for key, rows in sorted(by_stratum.items())
        },
        "records": candidate_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument(
        "--late-interaction", type=Path, default=DEFAULT_LATE_INTERACTION
    )
    parser.add_argument("--guarded-dir", type=Path, default=DEFAULT_GUARDED_DIR)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    assignments = read_json(args.assignments)
    baseline = read_json(args.baseline)
    late = read_json(args.late_interaction)
    guarded = {
        str(row["query_id"]): read_json(
            args.guarded_dir / f"{row['query_id']}_b21.json"
        )
        for row in late.get("results", [])
    }
    report = evaluate(
        assignments,
        baseline,
        late,
        guarded,
        policy_from_config(read_json(args.policy)),
    )
    write_json_atomic(args.output, report)
    print(
        "V19 guarded development: "
        f"{report['baseline']['end_to_end_accuracy']:.1%} -> "
        f"{report['candidate']['end_to_end_accuracy']:.1%}; "
        f"interventions={report['intervention_count']}"
    )
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
