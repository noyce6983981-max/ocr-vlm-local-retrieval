from __future__ import annotations

import hashlib
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.finalize_v19_formal_primary_review import (  # noqa: E402
    write_json_atomic,
)

ROUTES = (
    "text_evidence",
    "visual_discovery",
    "visual_metadata",
    "entity_exact",
    "topic_discovery",
    "mixed",
)
FORMAL = ROOT / "data/evaluation/v19/formal"
OUTPUTS = ROOT / "outputs/evaluation/v19/formal/holdout_once"
CLAIM = FORMAL / "holdout_claim.json"
AUTHORIZATION = FORMAL / "holdout_authorization.json"
METHOD_LOCK = FORMAL / "method_lock.json"
HOLDOUT = FORMAL / "holdout_queries_final_sealed.csv"
B0 = OUTPUTS / "b0_rule_only.json"
B1 = OUTPUTS / "b1_llm_only_prompt_v4.json"
B21 = OUTPUTS / "b21_hybrid_prompt_v4_guard3.json"
B3 = OUTPUTS / "b3_human_oracle.json"
COMPARISON = OUTPUTS / "b21_vs_b0.json"
RECEIPT = FORMAL / "holdout_results_receipt.json"


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain an object")
    return payload


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def accuracy(rows: list[dict[str, Any]]) -> float:
    correct = sum(
        1
        for row in rows
        if str(row["predicted_route"]) == str(row["gold_route"])
    )
    return correct / len(rows)


def macro_f1(rows: list[dict[str, Any]]) -> float:
    scores: list[float] = []
    for route in ROUTES:
        true_positive = sum(
            row["gold_route"] == route and row["predicted_route"] == route
            for row in rows
        )
        false_positive = sum(
            row["gold_route"] != route and row["predicted_route"] == route
            for row in rows
        )
        false_negative = sum(
            row["gold_route"] == route and row["predicted_route"] != route
            for row in rows
        )
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        scores.append(
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    return statistics.fmean(scores)


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def paired_family_bootstrap(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    *,
    repetitions: int = 10_000,
    seed: int = 1901,
) -> dict[str, Any]:
    baseline_by_id = {str(row["query_id"]): row for row in baseline}
    candidate_by_id = {str(row["query_id"]): row for row in candidate}
    if set(baseline_by_id) != set(candidate_by_id):
        raise ValueError("bootstrap methods have different query IDs")
    family_queries: dict[str, list[str]] = defaultdict(list)
    for row in baseline:
        family_queries[str(row["family_id"])].append(str(row["query_id"]))
    family_ids = sorted(family_queries)
    if len(family_ids) != 30 or set(map(len, family_queries.values())) != {4}:
        raise ValueError("formal holdout must contain 30 four-query families")
    rng = random.Random(seed)
    candidate_accuracy_values: list[float] = []
    candidate_macro_values: list[float] = []
    accuracy_delta_values: list[float] = []
    macro_delta_values: list[float] = []
    for _ in range(repetitions):
        sampled_families = [rng.choice(family_ids) for _ in family_ids]
        sampled_query_ids = [
            query_id
            for family_id in sampled_families
            for query_id in family_queries[family_id]
        ]
        baseline_rows = [baseline_by_id[query_id] for query_id in sampled_query_ids]
        candidate_rows = [
            candidate_by_id[query_id] for query_id in sampled_query_ids
        ]
        baseline_accuracy = accuracy(baseline_rows)
        candidate_accuracy = accuracy(candidate_rows)
        baseline_macro = macro_f1(baseline_rows)
        candidate_macro = macro_f1(candidate_rows)
        candidate_accuracy_values.append(candidate_accuracy)
        candidate_macro_values.append(candidate_macro)
        accuracy_delta_values.append(candidate_accuracy - baseline_accuracy)
        macro_delta_values.append(candidate_macro - baseline_macro)

    def interval(values: list[float]) -> dict[str, float]:
        return {
            "lower_95": percentile(values, 0.025),
            "upper_95": percentile(values, 0.975),
        }

    return {
        "unit": "paraphrase_family",
        "family_count": len(family_ids),
        "repetitions": repetitions,
        "seed": seed,
        "candidate_accuracy_95ci": interval(candidate_accuracy_values),
        "candidate_macro_f1_95ci": interval(candidate_macro_values),
        "accuracy_delta_vs_b0_95ci": interval(accuracy_delta_values),
        "macro_f1_delta_vs_b0_95ci": interval(macro_delta_values),
    }


def validate_method_payloads(payloads: list[dict[str, Any]]) -> None:
    expected_split = "adjudicated_formal_holdout_30pct_double_review"
    query_sets: list[set[str]] = []
    for payload in payloads:
        if payload.get("split") != expected_split:
            raise ValueError("holdout result has an unexpected split")
        if int(payload.get("query_count", 0)) != 120:
            raise ValueError("holdout result does not contain 120 queries")
        predictions = list(payload.get("predictions", []))
        query_ids = {str(row["query_id"]) for row in predictions}
        if len(predictions) != 120 or len(query_ids) != 120:
            raise ValueError("holdout predictions are incomplete")
        query_sets.append(query_ids)
    if any(query_ids != query_sets[0] for query_ids in query_sets[1:]):
        raise ValueError("holdout methods evaluated different queries")


def main() -> int:
    claim = read_json(CLAIM)
    authorization = read_json(AUTHORIZATION)
    method_lock = read_json(METHOD_LOCK)
    b0 = read_json(B0)
    b1 = read_json(B1)
    b21 = read_json(B21)
    b3 = read_json(B3)
    comparison = read_json(COMPARISON)
    if claim.get("status") != "v19_one_shot_holdout_claimed":
        raise ValueError("V19 holdout was not claimed")
    if claim.get("method_lock_sha256") != sha256_file(METHOD_LOCK):
        raise ValueError("holdout claim references a different method lock")
    if claim.get("holdout_query_sha256") != sha256_file(HOLDOUT):
        raise ValueError("holdout claim references different queries")
    if authorization.get("authorization_scope") is None:
        raise ValueError("holdout authorization scope is missing")
    if method_lock.get("holdout_executed") is not False:
        raise ValueError("locked pre-run method metadata was modified")
    validate_method_payloads([b0, b1, b21, b3])

    b0_metrics = b0["metrics"]
    b1_metrics = b1["metrics"]
    b21_metrics = b21["metrics"]
    b21_operational = b21["operational"]
    gates = {
        "macro_f1_delta_vs_rule": (
            b21_metrics["macro_f1"] - b0_metrics["macro_f1"] >= 0.05
        ),
        "ambiguous_subset_macro_f1_delta_vs_rule": (
            comparison["invocation_subset_macro_f1_delta"] >= 0.10
        ),
        "llm_call_rate": b21_metrics["llm_call_rate"] <= 0.40,
        "invalid_json_rate": (
            b21_metrics["invalid_json_rate_per_llm_call"] <= 0.01
        ),
        "fallback_rate": b21_metrics["fallback_rate"] == 0.0,
        "warm_p95_added_latency": (
            b21_operational["p95_route_latency_ms"] <= 1500
        ),
    }
    bootstrap = paired_family_bootstrap(
        list(b0["predictions"]), list(b21["predictions"])
    )
    receipt = {
        "schema_version": 1,
        "study_id": "v19-local-llm-structured-intent-routing",
        "status": "v19_one_shot_holdout_complete",
        "eligible_for_final_claim": all(gates.values()),
        "holdout_executed": True,
        "holdout_query_count": 120,
        "holdout_family_count": 30,
        "b0_rule_only": b0_metrics,
        "b1_llm_only": {
            "metrics": b1_metrics,
            "operational": b1["operational"],
        },
        "selected_b21_hybrid": {
            "metrics": b21_metrics,
            "operational": b21_operational,
            "accuracy_delta_vs_b0": (
                b21_metrics["accuracy"] - b0_metrics["accuracy"]
            ),
            "macro_f1_delta_vs_b0": (
                b21_metrics["macro_f1"] - b0_metrics["macro_f1"]
            ),
            "invocation_subset_macro_f1_delta_vs_b0": comparison[
                "invocation_subset_macro_f1_delta"
            ],
            "fixed_query_count": len(comparison["fixed_query_ids"]),
            "regressed_query_count": len(comparison["regressed_query_ids"]),
        },
        "b3_human_oracle": b3["metrics"],
        "bootstrap": bootstrap,
        "release_gates": gates,
        "all_release_gates_passed": all(gates.values()),
        "method_lock_sha256": sha256_file(METHOD_LOCK),
        "authorization_sha256": sha256_file(AUTHORIZATION),
        "claim_sha256": sha256_file(CLAIM),
        "holdout_query_sha256": sha256_file(HOLDOUT),
        "result_sha256": {
            "b0": sha256_file(B0),
            "b1": sha256_file(B1),
            "b21": sha256_file(B21),
            "b3": sha256_file(B3),
            "comparison": sha256_file(COMPARISON),
        },
    }
    write_json_atomic(RECEIPT, receipt)
    print(
        "V19 one-shot holdout finalized: "
        f"accuracy={b21_metrics['accuracy']:.4f}, "
        f"macro_f1={b21_metrics['macro_f1']:.4f}, "
        f"all_gates={all(gates.values())}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
