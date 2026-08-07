"""Apply a frozen V17 gate once without tuning on holdout judgments."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from ocr_vlm_retrieval.evaluation.human_evaluation import grouped_paired_bootstrap
from ocr_vlm_retrieval.evaluation.judgments import (
    NO_RELEVANT_CANDIDATE_IN_POOL,
    POOLED_RELEVANCE_TASK,
    RELEVANT_CANDIDATE_IN_POOL,
    normalize_pool_judgment,
)
from ocr_vlm_retrieval.evaluation.protocol_lock import file_sha256
from ocr_vlm_retrieval.gating.gate_calibration import (
    evaluate_topk_operating_point,
    mean,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def keyed_rows(
    rows: Iterable[Mapping[str, Any]], *, label: str
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        query_id = str(row.get("query_id", "")).strip()
        if not query_id or query_id in result:
            raise ValueError(f"Invalid or duplicate {label} query_id {query_id!r}")
        result[query_id] = dict(row)
    return result


def write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def evaluate_locked_holdout(
    *,
    method_lock: Mapping[str, Any],
    verification: Mapping[str, Any],
    judgments: Iterable[Mapping[str, Any]],
    baseline_rows: Iterable[Mapping[str, Any]],
    bootstrap_repetitions: int,
    seed: int,
) -> dict[str, Any]:
    if method_lock.get("status") != "method_locked_holdout_sealed":
        raise ValueError("V17 method lock is not in the final locked state")
    if verification.get("status") != "complete":
        raise ValueError("Holdout verification must be complete")
    if verification.get("judgments_read") is not False:
        raise ValueError("Holdout verifier must certify judgments_read=false")
    if verification.get("scope") != "v17_holdout_ranked_candidates_only":
        raise ValueError("Verification artifact is not marked as holdout-only")
    aggregation = method_lock.get("aggregation", {})
    if not isinstance(aggregation, Mapping):
        raise ValueError("Method lock aggregation is invalid")
    inference = method_lock.get("inference", {})
    if not isinstance(inference, Mapping):
        raise ValueError("Method lock inference settings are invalid")
    top_k = int(inference["top_k"])
    if aggregation.get("selection_policy") != (
        "highest_retrieval_rank_among_passed"
    ):
        raise ValueError("Holdout evaluator only supports the locked rank-first policy")
    if int(verification.get("top_k", 0)) < top_k:
        raise ValueError("Holdout verification depth is below the locked K")
    if verification.get("policy_sha256") != method_lock.get(
        "parser_policy_sha256"
    ):
        raise ValueError("Holdout parser policy hash does not match the lock")
    if verification.get("ranking_sha256") != method_lock.get(
        "candidate_ranking_sha256"
    ):
        raise ValueError("Holdout ranking policy hash does not match the lock")

    judgment_by_id = {
        query_id: normalize_pool_judgment(row)
        for query_id, row in keyed_rows(judgments, label="judgment").items()
    }
    verification_by_id = keyed_rows(
        verification.get("results", []), label="verification"
    )
    baseline_by_id = keyed_rows(baseline_rows, label="baseline")
    if not (
        set(judgment_by_id) == set(verification_by_id) == set(baseline_by_id)
    ):
        raise ValueError("Holdout inputs must contain identical query IDs")
    if len(judgment_by_id) != int(method_lock["holdout"]["query_count"]):
        raise ValueError("Holdout query count does not match the lock")

    rows: list[dict[str, Any]] = []
    for query_id in sorted(judgment_by_id):
        judgment = judgment_by_id[query_id]
        if judgment.get("adjudicated") is not True:
            raise ValueError("Every final holdout judgment must be adjudicated")
        if int(judgment.get("independent_reviewer_count", 0)) < 2:
            raise ValueError("Every holdout judgment needs two independent reviewers")
        pool_relevance = str(judgment["pool_relevance"])
        if pool_relevance not in {
            RELEVANT_CANDIDATE_IN_POOL,
            NO_RELEVANT_CANDIDATE_IN_POOL,
        }:
            raise ValueError("Final holdout cannot contain uncertain/excluded rows")
        relevance = judgment.get("candidate_relevance", {})
        candidates = verification_by_id[query_id].get("candidates", [])
        if not isinstance(relevance, Mapping) or not isinstance(candidates, list):
            raise ValueError("Invalid holdout candidate labels")
        candidate_rows = []
        seen_candidate_ids: set[str] = set()
        for source in candidates:
            candidate = dict(source)
            item_id = str(candidate.get("item_id", "")).strip()
            if not item_id or item_id in seen_candidate_ids:
                raise ValueError(f"Invalid candidate ID for {query_id}: {item_id!r}")
            seen_candidate_ids.add(item_id)
            if item_id not in relevance:
                raise ValueError(f"Missing holdout label for {query_id}/{item_id}")
            candidate["relevant"] = bool(relevance[item_id])
            candidate_rows.append(candidate)
        group_id = str(verification_by_id[query_id].get("group_id", ""))
        if not group_id:
            raise ValueError(f"Missing holdout group_id for {query_id}")
        rows.append(
            {
                "query_id": query_id,
                "group_id": group_id,
                "pool_relevance": pool_relevance,
                "candidates": candidate_rows,
            }
        )

    metrics, decisions = evaluate_topk_operating_point(
        rows,
        top_k=top_k,
        method=str(aggregation["method"]),
        threshold=float(aggregation["threshold"]),
        contrastive_relations=bool(aggregation["contrastive_relations"]),
        relation_margin_threshold=float(aggregation["relation_margin_threshold"]),
    )
    for decision in decisions:
        baseline = baseline_by_id[str(decision["query_id"])]
        if baseline.get("group_id") != decision["group_id"]:
            raise ValueError("V16 baseline group_id does not match holdout verification")
        if not isinstance(baseline.get("v16_accepted"), bool):
            raise ValueError("V16 baseline accepted decision must be boolean")
        if not isinstance(baseline.get("v16_pool_conditioned_correct"), bool):
            raise ValueError("V16 baseline correctness must be boolean")
        no_relevant = (
            decision["pool_relevance"] == NO_RELEVANT_CANDIDATE_IN_POOL
        )
        decision["v16_pool_conditioned_false_accept"] = float(
            no_relevant and baseline["v16_accepted"]
        )
        decision["v17_pool_conditioned_false_accept"] = float(
            no_relevant and decision["accepted"]
        )
        decision["v16_pool_conditioned_correct"] = float(
            baseline["v16_pool_conditioned_correct"]
        )
        decision["v17_pool_conditioned_correct"] = float(
            decision["pool_conditioned_correct"]
        )
    no_relevant_decisions = [
        row
        for row in decisions
        if row["pool_relevance"] == NO_RELEVANT_CANDIDATE_IN_POOL
    ]
    return {
        "status": "v17_holdout_evaluated_once",
        "task_id": POOLED_RELEVANCE_TASK,
        "method_tuned_on_holdout": False,
        "selection_policy": "highest_retrieval_rank_among_passed",
        "query_count": len(rows),
        "metrics": metrics,
        "baseline_v16_pool_conditioned_false_accept_rate": mean(
            float(row["v16_pool_conditioned_false_accept"])
            for row in no_relevant_decisions
        ),
        "paired_group_bootstrap": {
            "pool_conditioned_false_accept_v17_minus_v16": (
                grouped_paired_bootstrap(
                    no_relevant_decisions,
                    baseline_field="v16_pool_conditioned_false_accept",
                    contender_field="v17_pool_conditioned_false_accept",
                    repetitions=bootstrap_repetitions,
                    seed=seed,
                )
            ),
            "end_to_end_correct_v17_minus_v16": grouped_paired_bootstrap(
                decisions,
                baseline_field="v16_pool_conditioned_correct",
                contender_field="v17_pool_conditioned_correct",
                repetitions=bootstrap_repetitions,
                seed=seed,
            ),
        },
        "decisions": decisions,
        "claim_boundary": (
            "Final V17-Compositional-80 pooled-relevance holdout only; "
            "not a corpus-level no-answer estimate."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method-lock", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--baseline-records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    method_lock = json.loads(args.method_lock.read_text(encoding="utf-8"))
    report = evaluate_locked_holdout(
        method_lock=method_lock,
        verification=json.loads(args.verification.read_text(encoding="utf-8")),
        judgments=read_jsonl(args.judgments),
        baseline_rows=read_jsonl(args.baseline_records),
        bootstrap_repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )
    report["method_lock_sha256"] = file_sha256(args.method_lock)
    report["input_hashes"] = {
        "verification": file_sha256(args.verification),
        "judgments": file_sha256(args.judgments),
        "baseline_records": file_sha256(args.baseline_records),
    }
    write_new_json(args.output, report)
    print(args.output)


if __name__ == "__main__":
    main()
