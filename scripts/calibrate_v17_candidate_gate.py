"""Select a Top-K V17 candidate-verification gate on calibration only."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from collections.abc import Iterable, Mapping, Sequence
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
from ocr_vlm_retrieval.gating.gate_calibration import (
    AGGREGATORS,
    evaluate_topk_operating_point,
    mark_eligibility,
    mean,
    select_near_optimal_operating_point,
)

DEFAULT_TOP_K_VALUES = (1, 3, 5)
DEFAULT_RELATION_MARGINS = (0.0, 0.03, 0.05, 0.08, 0.10)


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


def merge_judgment_extensions(
    judgments: Iterable[Mapping[str, Any]],
    extensions: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Attach audited Top-5 labels while preserving the frozen base judgments."""

    base_rows = [copy.deepcopy(dict(row)) for row in judgments]
    by_id = keyed_rows(base_rows, label="judgment")
    for extension in extensions:
        query_id = str(extension.get("query_id", "")).strip()
        if query_id not in by_id:
            raise ValueError(f"Judgment extension has unknown query_id {query_id!r}")
        if extension.get("split") != "calibration":
            raise ValueError("Judgment extensions must be calibration-only")
        additions = extension.get("candidate_relevance", {})
        current = by_id[query_id].get("candidate_relevance", {})
        if not isinstance(additions, Mapping) or not isinstance(current, dict):
            raise ValueError("Candidate relevance must be a mapping")
        for item_id, relevant in additions.items():
            key = str(item_id).strip()
            if not key or key in current:
                raise ValueError(
                    f"Judgment extension candidate {key!r} is invalid or duplicated"
                )
            current[key] = bool(relevant)
    return base_rows


def _best_point(points: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    eligible = [dict(point) for point in points if point.get("eligible")]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda point: (
            -float(point["pool_conditioned_end_to_end_accuracy"]),
            -float(point["selected_relevant_query_rate"]),
            float(point["pool_conditioned_false_accept_rate"]),
            float(point["threshold"]),
            float(point["relation_margin_threshold"]),
        ),
    )


def _evidence_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    relation_count = 0
    ocr_count = 0
    ocr_exact = 0
    ocr_fuzzy = 0
    for row in rows:
        for candidate in row["candidates"]:
            relation_count += len(candidate.get("contrastive_relation_evidence", []))
            for evidence in candidate.get("ocr_evidence", {}).values():
                ocr_count += 1
                level = str(evidence.get("match_level", "none"))
                ocr_exact += int(level == "exact")
                ocr_fuzzy += int(level == "fuzzy")
    return {
        "contrastive_relation_evidence_count": relation_count,
        "explicit_ocr_requirement_evidence_count": ocr_count,
        "ocr_exact_match_count": ocr_exact,
        "ocr_fuzzy_match_count": ocr_fuzzy,
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
    top_k_values: Sequence[int] = DEFAULT_TOP_K_VALUES,
    relation_margins: Sequence[float] = DEFAULT_RELATION_MARGINS,
    near_optimal_accuracy_tolerance: float = 0.03,
    require_requested_top_k: bool = False,
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
    evaluated_ids = {
        query_id
        for query_id, row in judgment_by_id.items()
        if row.get("pool_relevance") != "excluded"
    }
    if not evaluated_ids.issubset(baseline_by_id):
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
        raw_candidates = verification_by_id[query_id].get("candidates", [])
        if not isinstance(raw_candidates, list) or not raw_candidates:
            raise ValueError(f"Verification {query_id} has no candidates")
        relevance = judgment.get("candidate_relevance", {})
        if not isinstance(relevance, Mapping):
            raise ValueError(f"Judgment {query_id} has invalid candidate relevance")
        candidates: list[dict[str, Any]] = []
        seen_items: set[str] = set()
        for source_candidate in raw_candidates:
            candidate = dict(source_candidate)
            item_id = str(candidate.get("item_id", "")).strip()
            if not item_id or item_id in seen_items:
                raise ValueError(f"Verification {query_id} has invalid candidate IDs")
            if item_id not in relevance:
                raise ValueError(
                    f"Judgment {query_id} does not cover candidate {item_id}"
                )
            seen_items.add(item_id)
            candidate["relevant"] = bool(relevance[item_id])
            candidates.append(candidate)
        group_id = str(verification_by_id[query_id].get("group_id", "")).strip()
        if not group_id:
            raise ValueError(f"Verification {query_id} has no group_id")
        rows.append(
            {
                "query_id": query_id,
                "group_id": group_id,
                "task_id": POOLED_RELEVANCE_TASK,
                "pool_relevance": pool_relevance,
                "v16_accepted": bool(baseline_by_id[query_id]["v16_accepted"]),
                "baseline_v16_pool_conditioned_correct": float(
                    baseline_by_id[query_id].get(
                        "v16_pool_conditioned_correct",
                        baseline_by_id[query_id].get("v16_correct", 0.0),
                    )
                ),
                "candidates": candidates,
            }
        )

    maximum_available_k = min(len(row["candidates"]) for row in rows)
    requested_k = sorted({int(value) for value in top_k_values})
    if any(value < 1 for value in requested_k):
        raise ValueError("Every requested Top-K value must be positive")
    if require_requested_top_k and maximum_available_k < max(requested_k):
        raise ValueError(
            f"Verification artifact only supports K={maximum_available_k}; "
            f"requested K={max(requested_k)}"
        )
    evaluated_k = [value for value in requested_k if value <= maximum_available_k]
    if not evaluated_k:
        raise ValueError("No requested Top-K value is available")

    no_relevant_rows = [
        row
        for row in rows
        if row["pool_relevance"] == NO_RELEVANT_CANDIDATE_IN_POOL
    ]
    baseline_far = mean(float(row["v16_accepted"]) for row in no_relevant_rows)
    contrastive_available = any(
        candidate.get("contrastive_relation_evidence")
        for row in rows
        for candidate in row["candidates"]
    )
    contrastive_modes = (False, True) if contrastive_available else (False,)
    operating_points: list[dict[str, Any]] = []
    for top_k in evaluated_k:
        for method in AGGREGATORS:
            for contrastive in contrastive_modes:
                margins = relation_margins if contrastive else (0.0,)
                for margin in margins:
                    for threshold_index in range(101):
                        metrics, _ = evaluate_topk_operating_point(
                            rows,
                            top_k=top_k,
                            method=method,
                            threshold=threshold_index / 100,
                            contrastive_relations=contrastive,
                            relation_margin_threshold=float(margin),
                        )
                        operating_points.append(
                            mark_eligibility(
                                metrics,
                                baseline_false_accept_rate=baseline_far,
                                minimum_false_accept_relative_reduction=(
                                    min_far_relative_reduction
                                ),
                                minimum_selected_relevant_rate=(
                                    min_relevant_acceptance
                                ),
                            )
                        )

    selected_metrics = select_near_optimal_operating_point(
        operating_points,
        accuracy_tolerance=near_optimal_accuracy_tolerance,
    )
    _, selected_decisions = evaluate_topk_operating_point(
        rows,
        top_k=int(selected_metrics["top_k"]),
        method=str(selected_metrics["method"]),
        threshold=float(selected_metrics["threshold"]),
        contrastive_relations=bool(selected_metrics["contrastive_relations"]),
        relation_margin_threshold=float(
            selected_metrics["relation_margin_threshold"]
        ),
    )
    source_by_id = {str(row["query_id"]): row for row in rows}
    decisions: list[dict[str, Any]] = []
    for decision in selected_decisions:
        source = source_by_id[str(decision["query_id"])]
        no_relevant = (
            decision["pool_relevance"] == NO_RELEVANT_CANDIDATE_IN_POOL
        )
        decisions.append(
            {
                **decision,
                "task_id": POOLED_RELEVANCE_TASK,
                "selected_method": selected_metrics["method"],
                "selected_threshold": selected_metrics["threshold"],
                "top_k_verified": selected_metrics["top_k"],
                "contrastive_relations": selected_metrics[
                    "contrastive_relations"
                ],
                "relation_margin_threshold": selected_metrics[
                    "relation_margin_threshold"
                ],
                "v16_pool_conditioned_false_accept": float(
                    no_relevant and source["v16_accepted"]
                ),
                "v17_pool_conditioned_false_accept": float(
                    no_relevant and decision["accepted"]
                ),
                "v16_pool_conditioned_correct": source[
                    "baseline_v16_pool_conditioned_correct"
                ],
                "v17_pool_conditioned_correct": float(
                    decision["pool_conditioned_correct"]
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
    ablation_summary = []
    for top_k in evaluated_k:
        for contrastive in contrastive_modes:
            subset = [
                point
                for point in operating_points
                if int(point["top_k"]) == top_k
                and bool(point["contrastive_relations"]) == contrastive
            ]
            ablation_summary.append(
                {
                    "top_k": top_k,
                    "contrastive_relations": contrastive,
                    "best_eligible_operating_point": _best_point(subset),
                }
            )

    selected_gate = {
        "status": "selected_on_model_assisted_calibration_not_holdout_locked",
        "method": selected_metrics["method"],
        "threshold": selected_metrics["threshold"],
        "top_k_verified": selected_metrics["top_k"],
        "selection_policy": selected_metrics["selection_policy"],
        "contrastive_relations": selected_metrics["contrastive_relations"],
        "relation_margin_threshold": selected_metrics[
            "relation_margin_threshold"
        ],
        "ocr_evidence_policy": verification.get("ocr_policy"),
        "verification_prompts": verification.get("verification_prompts"),
        "model": verification.get("model"),
        "parser_policy_sha256": verification.get("policy_sha256"),
        "verification_ranking_sha256": verification.get("ranking_sha256"),
        "verification_packet_extension_sha256": verification.get(
            "packet_extension_sha256"
        ),
        "verification_schema_version": verification.get("schema_version"),
        "selection_constraints": {
            "minimum_pool_conditioned_false_accept_relative_reduction": (
                min_far_relative_reduction
            ),
            "minimum_selected_relevant_query_rate": min_relevant_acceptance,
            "near_optimal_accuracy_tolerance": near_optimal_accuracy_tolerance,
            "reject_all_forbidden": True,
        },
        "selection_rule": (
            "Among eligible operating points, retain those within the fixed "
            "accuracy tolerance of the best calibration result, choose the "
            "smallest K, then maximize end-to-end accuracy and relevant-query "
            "success while minimizing pool-conditioned false acceptance. "
            "Return the highest retrieval-ranked candidate that passes."
        ),
    }
    query_count = int(verification.get("query_count", len(rows) + len(excluded)))
    elapsed_seconds = float(verification.get("elapsed_seconds", 0.0))
    report = {
        "status": "topk_calibration_complete_holdout_not_run",
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
            "requested_top_k_values": requested_k,
            "evaluated_top_k_values": evaluated_k,
        },
        "baseline_v16_pool_conditioned_false_accept_rate": baseline_far,
        "evidence_coverage": _evidence_counts(rows),
        "verification_runtime": {
            "artifact_top_k": verification.get("top_k", maximum_available_k),
            "total_elapsed_seconds": elapsed_seconds,
            "mean_elapsed_seconds_per_query": (
                elapsed_seconds / query_count if query_count else None
            ),
            "peak_reserved_gib": verification.get("peak_reserved_gib"),
            "selected_mean_verified_candidate_count": selected_metrics[
                "mean_verified_candidate_count"
            ],
        },
        "ablation_summary": ablation_summary,
        "selected_gate": selected_gate,
        "selected_operating_point": selected_metrics,
        "paired_group_bootstrap": bootstrap,
        "operating_points": operating_points,
        "limitations": [
            "Method selection uses model-assisted calibration labels, not human gold.",
            "Pool-conditioned false acceptance is not corpus-level answerability.",
            "The Top-K scorer batches candidates; sequential production latency "
            "was not measured, so verified-candidate count is only a cost proxy.",
            "Bootstrap intervals do not correct for calibration grid selection.",
            "Only the sealed holdout can provide the final independent estimate.",
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
            "candidate_attribute_verification_top5_contrastive.json"
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
    parser.add_argument(
        "--judgment-extension",
        type=Path,
        default=Path(
            "data/evaluation/v17/human_study/calibration/"
            "top5_extension_judgments.jsonl"
        ),
    )
    parser.add_argument(
        "--top-k-values",
        type=int,
        nargs="+",
        default=list(DEFAULT_TOP_K_VALUES),
    )
    parser.add_argument(
        "--relation-margins",
        type=float,
        nargs="+",
        default=list(DEFAULT_RELATION_MARGINS),
    )
    parser.add_argument("--near-optimal-accuracy-tolerance", type=float, default=0.03)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/evaluation/v17/calibration/parser_v3/"
            "candidate_gate_topk_calibration_report.json"
        ),
    )
    parser.add_argument(
        "--decisions-output",
        type=Path,
        default=Path(
            "outputs/evaluation/v17/calibration/parser_v3/"
            "candidate_gate_topk_paired_decisions.jsonl"
        ),
    )
    parser.add_argument(
        "--selected-gate-output",
        type=Path,
        default=Path("config/v17_candidate_verification_gate_topk_candidate.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verification = json.loads(
        project_path(args.verification).read_text(encoding="utf-8")
    )
    report, decisions, selected_gate = calibrate(
        verification=verification,
        judgments=merge_judgment_extensions(
            read_jsonl(project_path(args.judgments)),
            read_jsonl(project_path(args.judgment_extension)),
        ),
        baseline_rows=read_jsonl(project_path(args.baseline_records)),
        bootstrap_repetitions=args.bootstrap_repetitions,
        seed=args.seed,
        top_k_values=args.top_k_values,
        relation_margins=args.relation_margins,
        near_optimal_accuracy_tolerance=args.near_optimal_accuracy_tolerance,
        require_requested_top_k=True,
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
