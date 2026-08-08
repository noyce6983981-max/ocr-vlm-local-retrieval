"""Build and verify privacy-preserving reports for a sealed holdout study."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from ocr_vlm_retrieval.evaluation.human_evaluation import (
    grouped_paired_bootstrap,
)
from ocr_vlm_retrieval.evaluation.judgments import (
    NO_RELEVANT_CANDIDATE_IN_POOL,
    RELEVANT_CANDIDATE_IN_POOL,
    normalize_pool_judgment,
)

PUBLIC_RECORD_FIELDS = frozenset(
    {
        "query_id",
        "group_id",
        "pool_relevance",
        "v16_relevant_in_top3",
        "v17_relevant_in_top3",
        "v16_accepted",
        "v16_selected_relevant",
        "v16_correct",
        "v17_accepted",
        "v17_selected_rank",
        "v17_selected_relevant",
        "v17_correct",
    }
)


def _keyed_rows(
    rows: Iterable[Mapping[str, Any]], *, label: str
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for source in rows:
        query_id = str(source.get("query_id", "")).strip()
        if not query_id or query_id in result:
            raise ValueError(f"Invalid or duplicate {label} query_id {query_id!r}")
        result[query_id] = dict(source)
    return result


def _bool_field(row: Mapping[str, Any], field: str) -> bool:
    value = row.get(field)
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be boolean")
    return value


def build_public_paired_records(
    *,
    final_report: Mapping[str, Any],
    v16_rankings: Iterable[Mapping[str, Any]],
    judgments: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Remove item/source identity while retaining paired evaluation outcomes."""

    if final_report.get("status") != "v17_holdout_evaluated_once":
        raise ValueError("Final report is not a sealed V17 holdout result")
    if final_report.get("method_tuned_on_holdout") is not False:
        raise ValueError("Final report must certify method_tuned_on_holdout=false")
    decisions = _keyed_rows(final_report.get("decisions", []), label="decision")
    rankings = _keyed_rows(v16_rankings, label="V16 ranking")
    normalized_judgments = {
        query_id: normalize_pool_judgment(row)
        for query_id, row in _keyed_rows(judgments, label="judgment").items()
    }
    if not set(decisions) == set(rankings) == set(normalized_judgments):
        raise ValueError("Report, V16 rankings, and judgments need identical queries")
    if len(decisions) != int(final_report.get("query_count", 0)):
        raise ValueError("Final report query_count is inconsistent")

    raw_group_ids = sorted(
        {str(row.get("group_id", "")).strip() for row in decisions.values()}
    )
    if not raw_group_ids or raw_group_ids[0] == "":
        raise ValueError("Every decision needs a non-empty group_id")
    group_aliases = {
        group_id: f"group_{index:03d}"
        for index, group_id in enumerate(raw_group_ids, start=1)
    }

    public_rows: list[dict[str, Any]] = []
    for index, query_id in enumerate(sorted(decisions), start=1):
        decision = decisions[query_id]
        judgment = normalized_judgments[query_id]
        ranking = rankings[query_id].get("ranking")
        if not isinstance(ranking, list) or len(ranking) < 3:
            raise ValueError(f"V16 ranking for {query_id} has fewer than 3 rows")
        relevance = judgment.get("candidate_relevance")
        if not isinstance(relevance, Mapping):
            raise ValueError(f"Candidate relevance for {query_id} is invalid")
        v16_top3_ids = [
            str(candidate.get("item_id", "")).strip() for candidate in ranking[:3]
        ]
        if any(not item_id for item_id in v16_top3_ids):
            raise ValueError(f"V16 Top-3 for {query_id} contains an invalid item")
        v16_relevant_in_top3 = any(
            bool(relevance.get(item_id, False)) for item_id in v16_top3_ids
        )
        pool_relevance = str(judgment["pool_relevance"])
        if pool_relevance != str(decision.get("pool_relevance")):
            raise ValueError(f"Pool relevance differs for {query_id}")
        selected_rank = decision.get("selected_rank")
        if selected_rank is not None and not isinstance(selected_rank, int):
            raise ValueError("v17 selected_rank must be an integer or null")
        group_id = str(decision["group_id"])
        public_rows.append(
            {
                "query_id": f"holdout_{index:03d}",
                "group_id": group_aliases[group_id],
                "pool_relevance": pool_relevance,
                "v16_relevant_in_top3": v16_relevant_in_top3,
                "v17_relevant_in_top3": _bool_field(
                    decision, "relevant_in_top_k"
                ),
                "v16_accepted": _bool_field(decision, "v16_accepted"),
                "v16_selected_relevant": _bool_field(
                    decision, "v16_selected_relevant"
                ),
                "v16_correct": bool(decision["v16_pool_conditioned_correct"]),
                "v17_accepted": _bool_field(decision, "accepted"),
                "v17_selected_rank": selected_rank,
                "v17_selected_relevant": _bool_field(
                    decision, "selected_relevant"
                ),
                "v17_correct": bool(decision["v17_pool_conditioned_correct"]),
            }
        )
    validate_public_paired_records(public_rows)
    return public_rows


def validate_public_paired_records(rows: Sequence[Mapping[str, Any]]) -> None:
    """Reject leaked identities, malformed rows, and inconsistent decisions."""

    if not rows:
        raise ValueError("At least one public paired record is required")
    query_ids: set[str] = set()
    for row in rows:
        fields = set(row)
        if fields != PUBLIC_RECORD_FIELDS:
            raise ValueError(
                "Public record fields differ from the privacy allowlist: "
                f"missing={sorted(PUBLIC_RECORD_FIELDS - fields)}, "
                f"extra={sorted(fields - PUBLIC_RECORD_FIELDS)}"
            )
        query_id = str(row["query_id"])
        group_id = str(row["group_id"])
        if not query_id.startswith("holdout_") or not group_id.startswith("group_"):
            raise ValueError("Public IDs must use remapped holdout/group aliases")
        if query_id in query_ids:
            raise ValueError(f"Duplicate public query_id {query_id}")
        query_ids.add(query_id)
        for field in PUBLIC_RECORD_FIELDS - {
            "query_id",
            "group_id",
            "pool_relevance",
            "v17_selected_rank",
        }:
            _bool_field(row, field)
        pool_relevance = str(row["pool_relevance"])
        if pool_relevance not in {
            RELEVANT_CANDIDATE_IN_POOL,
            NO_RELEVANT_CANDIDATE_IN_POOL,
        }:
            raise ValueError(f"Invalid public pool_relevance {pool_relevance!r}")
        selected_rank = row["v17_selected_rank"]
        if selected_rank is not None and (
            not isinstance(selected_rank, int) or not 1 <= selected_rank <= 3
        ):
            raise ValueError("v17_selected_rank must be null or an integer in [1, 3]")
        if bool(row["v17_accepted"]) != (selected_rank is not None):
            raise ValueError("V17 acceptance and selected rank are inconsistent")


def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    if not rows:
        raise ValueError(f"Cannot compute {field} over an empty slice")
    return sum(float(row[field]) for row in rows) / len(rows)


def analyze_public_paired_records(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_repetitions: int = 10_000,
    seed: int = 17,
) -> dict[str, Any]:
    """Recompute the public V17 metrics and the decision-stage error funnel."""

    validate_public_paired_records(rows)
    relevant_rows = [
        row
        for row in rows
        if row["pool_relevance"] == RELEVANT_CANDIDATE_IN_POOL
    ]
    no_relevant_rows = [
        row
        for row in rows
        if row["pool_relevance"] == NO_RELEVANT_CANDIDATE_IN_POOL
    ]
    if not relevant_rows or not no_relevant_rows:
        raise ValueError("Public records need relevant and no-relevant pool strata")

    bootstrap_rows = [dict(row) for row in rows]
    for row in bootstrap_rows:
        no_relevant = row["pool_relevance"] == NO_RELEVANT_CANDIDATE_IN_POOL
        row["v16_false_accept"] = bool(no_relevant and row["v16_accepted"])
        row["v17_false_accept"] = bool(no_relevant and row["v17_accepted"])
    relevant_bootstrap_rows = [
        row
        for row in bootstrap_rows
        if row["pool_relevance"] == RELEVANT_CANDIDATE_IN_POOL
    ]
    no_relevant_bootstrap_rows = [
        row
        for row in bootstrap_rows
        if row["pool_relevance"] == NO_RELEVANT_CANDIDATE_IN_POOL
    ]

    v17_top3_rows = [row for row in relevant_rows if row["v17_relevant_in_top3"]]
    selected_relevant_rows = [
        row for row in relevant_rows if row["v17_selected_relevant"]
    ]
    reachable_failures = [
        row
        for row in v17_top3_rows
        if not bool(row["v17_selected_relevant"])
    ]
    accepted_rows = [row for row in rows if row["v17_accepted"]]
    correct_rejections = [row for row in no_relevant_rows if not row["v17_accepted"]]

    return {
        "query_count": len(rows),
        "group_count": len({str(row["group_id"]) for row in rows}),
        "funnel": {
            "relevant_candidate_in_pool": len(relevant_rows),
            "v17_relevant_in_top3": len(v17_top3_rows),
            "v17_selected_relevant": len(selected_relevant_rows),
            "v17_top3_recall_miss": len(relevant_rows) - len(v17_top3_rows),
            "v17_reachable_but_not_selected": len(reachable_failures),
            "reachable_all_candidates_below_threshold": sum(
                not bool(row["v17_accepted"]) for row in reachable_failures
            ),
            "reachable_wrong_candidate_accepted": sum(
                bool(row["v17_accepted"]) for row in reachable_failures
            ),
            "no_relevant_candidate_in_pool": len(no_relevant_rows),
            "v17_correct_rejection": len(correct_rejections),
            "v17_false_accept": len(no_relevant_rows) - len(correct_rejections),
        },
        "metrics": {
            "v16_retrieval_recall_at_3": _mean(
                relevant_rows, "v16_relevant_in_top3"
            ),
            "v17_retrieval_recall_at_3": _mean(
                relevant_rows, "v17_relevant_in_top3"
            ),
            "v17_top3_reachable_conversion_rate": (
                len(selected_relevant_rows) / len(v17_top3_rows)
            ),
            "v17_accepted_selected_relevant_precision": (
                len(selected_relevant_rows) / len(accepted_rows)
            ),
            "v17_positive_end_to_end_success_rate": (
                len(selected_relevant_rows) / len(relevant_rows)
            ),
            "v17_no_relevant_in_pool_rejection_rate": (
                len(correct_rejections) / len(no_relevant_rows)
            ),
            "v16_pool_conditioned_end_to_end_accuracy": _mean(
                rows, "v16_correct"
            ),
            "v17_pool_conditioned_end_to_end_accuracy": _mean(
                rows, "v17_correct"
            ),
            "v16_pool_conditioned_false_accept_rate": _mean(
                no_relevant_bootstrap_rows, "v16_false_accept"
            ),
            "v17_pool_conditioned_false_accept_rate": _mean(
                no_relevant_bootstrap_rows, "v17_false_accept"
            ),
            "v17_acceptance_coverage": len(accepted_rows) / len(rows),
        },
        "paired_group_bootstrap": {
            "retrieval_recall_at_3_v17_minus_v16": grouped_paired_bootstrap(
                relevant_bootstrap_rows,
                baseline_field="v16_relevant_in_top3",
                contender_field="v17_relevant_in_top3",
                repetitions=bootstrap_repetitions,
                seed=seed,
            ),
            "end_to_end_correct_v17_minus_v16": grouped_paired_bootstrap(
                bootstrap_rows,
                baseline_field="v16_correct",
                contender_field="v17_correct",
                repetitions=bootstrap_repetitions,
                seed=seed,
            ),
            "false_accept_v17_minus_v16": grouped_paired_bootstrap(
                no_relevant_bootstrap_rows,
                baseline_field="v16_false_accept",
                contender_field="v17_false_accept",
                repetitions=bootstrap_repetitions,
                seed=seed,
            ),
        },
        "claim_boundary": (
            "Pooled-relevance holdout only. Accepted-result precision and rejection "
            "rates are conditional on the frozen 20-candidate pools, not the corpus."
        ),
    }


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_verification_runtime(
    verification: Mapping[str, Any],
) -> dict[str, Any]:
    """Extract descriptive runtime data without rerunning the verifier."""

    if verification.get("status") != "complete":
        raise ValueError("Verification artifact must be complete")
    if verification.get("judgments_read") is not False:
        raise ValueError("Runtime source must certify judgments_read=false")
    result_rows = verification.get("results")
    if not isinstance(result_rows, list) or not result_rows:
        raise ValueError("Verification artifact has no result rows")
    elapsed = [float(row["elapsed_seconds"]) for row in result_rows]
    if any(value <= 0 for value in elapsed):
        raise ValueError("Per-query elapsed times must be positive")
    completed = int(verification.get("completed_query_count", 0))
    if completed != len(elapsed):
        raise ValueError("completed_query_count differs from runtime rows")
    return {
        "completed_query_count": completed,
        "model_load_seconds": float(verification["load_seconds"]),
        "verification_elapsed_seconds": float(verification["elapsed_seconds"]),
        "per_query_elapsed_seconds": {
            "mean": sum(elapsed) / len(elapsed),
            "p50": _percentile(elapsed, 0.50),
            "p95": _percentile(elapsed, 0.95),
            "minimum": min(elapsed),
            "maximum": max(elapsed),
        },
        "peak_reserved_gib": float(verification["peak_reserved_gib"]),
        "scope": (
            "Descriptive label-blind Top-3 verifier timing from the sealed artifact; "
            "not production end-to-end search latency."
        ),
    }
