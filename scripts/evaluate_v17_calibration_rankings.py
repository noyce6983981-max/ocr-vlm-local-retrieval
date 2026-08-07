"""Evaluate V17 calibration rankings on audited pooled-relevance labels.

The calibration candidate pool contains the union of the top three results from
each retrieval route.  Consequently, this script intentionally reports only
Top-1, Top-3, and MRR@3 ranking metrics.  It never reads the sealed holdout set.
"""

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

RUN_NAMES = (
    "bm25",
    "global_visual",
    "text",
    "v16_quality_hybrid",
    "v17_attribute_coverage",
    "v17_quality_hybrid",
)
SYSTEM_VERSIONS = ("v16", "v17")


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            rows.append(payload)
    return rows


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
    keyed: dict[str, dict[str, Any]] = {}
    for source in rows:
        query_id = str(source.get("query_id", "")).strip()
        if not query_id:
            raise ValueError(f"Every {label} row needs query_id")
        if query_id in keyed:
            raise ValueError(f"Duplicate {label} row for {query_id}")
        keyed[query_id] = dict(source)
    return keyed


def ranking_item_ids(row: Mapping[str, Any]) -> list[str]:
    ranking = row.get("ranking")
    if not isinstance(ranking, list):
        raise ValueError("Every run row needs a ranking list")
    item_ids: list[str] = []
    for result in ranking:
        if not isinstance(result, Mapping):
            raise ValueError("Every ranking result must be an object")
        item_id = str(result.get("item_id", "")).strip()
        if not item_id:
            raise ValueError("Every ranking result needs item_id")
        item_ids.append(item_id)
    return item_ids


def reciprocal_rank_at_3(item_ids: list[str], relevant: set[str]) -> float:
    for rank, item_id in enumerate(item_ids[:3], start=1):
        if item_id in relevant:
            return 1.0 / rank
    return 0.0


def mean(values: Iterable[float]) -> float:
    collected = list(values)
    return sum(collected) / len(collected) if collected else 0.0


def load_acceptance(raw_dir: Path, query_id: str, version: str) -> tuple[bool, str]:
    path = raw_dir / f"{query_id}_{version}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    decision = payload.get("acceptance", {}).get("quality_hybrid", {})
    if not isinstance(decision.get("accepted"), bool):
        raise ValueError(f"Missing quality_hybrid acceptance in {path}")
    return bool(decision["accepted"]), str(decision.get("reason", ""))


def evaluate(
    *,
    judgments: Iterable[Mapping[str, Any]],
    packets: Iterable[Mapping[str, Any]],
    run_rows: Mapping[str, Iterable[Mapping[str, Any]]],
    raw_dir: Path,
    bootstrap_repetitions: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    judgment_by_id = keyed_rows(judgments, label="judgment")
    packet_by_id = keyed_rows(packets, label="packet")
    run_by_name = {
        name: keyed_rows(rows, label=f"{name} run") for name, rows in run_rows.items()
    }
    unknown_run_names = set(run_by_name) - set(RUN_NAMES)
    if unknown_run_names:
        raise ValueError("Unknown run inputs: " + ", ".join(sorted(unknown_run_names)))
    required_runs = {"v16_quality_hybrid", "v17_quality_hybrid"}
    if not required_runs.issubset(run_by_name):
        raise ValueError("V16 and V17 quality-hybrid runs are required")
    evaluated_run_names = tuple(name for name in RUN_NAMES if name in run_by_name)
    if set(judgment_by_id) != set(packet_by_id):
        raise ValueError("Judgment and packet query IDs do not match")
    for name, rows in run_by_name.items():
        if set(rows) != set(judgment_by_id):
            raise ValueError(f"{name} query IDs do not match judgments")

    paired_rows: list[dict[str, Any]] = []
    excluded_query_ids: list[str] = []
    relevant_in_pool_ids: list[str] = []
    no_relevant_in_pool_ids: list[str] = []
    for query_id in sorted(judgment_by_id):
        judgment = normalize_pool_judgment(judgment_by_id[query_id])
        pool_relevance = str(judgment["pool_relevance"])
        if pool_relevance == "excluded":
            excluded_query_ids.append(query_id)
            continue
        if pool_relevance not in {
            RELEVANT_CANDIDATE_IN_POOL,
            NO_RELEVANT_CANDIDATE_IN_POOL,
        }:
            raise ValueError(
                f"Unsupported pool relevance for {query_id}: {pool_relevance!r}"
            )
        relevant = {
            str(item_id)
            for item_id, value in judgment.get("candidate_relevance", {}).items()
            if bool(value)
        }
        target_ids = (
            relevant_in_pool_ids
            if pool_relevance == RELEVANT_CANDIDATE_IN_POOL
            else no_relevant_in_pool_ids
        )
        target_ids.append(query_id)

        row: dict[str, Any] = {
            "query_id": query_id,
            "group_id": str(packet_by_id[query_id].get("group_id", "")).strip(),
            "query_family": packet_by_id[query_id].get("query_family"),
            "task_id": POOLED_RELEVANCE_TASK,
            "pool_relevance": pool_relevance,
            "relevant_item_ids": sorted(relevant),
        }
        if not row["group_id"]:
            raise ValueError(f"Packet {query_id} has no group_id")
        for name in evaluated_run_names:
            item_ids = ranking_item_ids(run_by_name[name][query_id])
            pool_ids = set(judgment.get("candidate_relevance", {}))
            unknown = set(item_ids[:3]) - pool_ids
            if unknown:
                prefix = (
                    f"{name} top-three results for {query_id} are outside "
                    "the audited pool: "
                )
                raise ValueError(prefix + ", ".join(sorted(unknown)))
            row[f"{name}_top1_item_id"] = item_ids[0] if item_ids else None
            row[f"{name}_r_at_1"] = float(bool(item_ids and item_ids[0] in relevant))
            row[f"{name}_r_at_3"] = float(bool(set(item_ids[:3]) & relevant))
            row[f"{name}_mrr_at_3"] = reciprocal_rank_at_3(item_ids, relevant)
        for version in SYSTEM_VERSIONS:
            accepted, reason = load_acceptance(raw_dir, query_id, version)
            top1_relevant = bool(row[f"{version}_quality_hybrid_r_at_1"])
            row[f"{version}_accepted"] = accepted
            row[f"{version}_acceptance_reason"] = reason
            row[f"{version}_pool_conditioned_false_accept"] = float(
                pool_relevance == NO_RELEVANT_CANDIDATE_IN_POOL and accepted
            )
            row[f"{version}_pool_conditioned_correct"] = float(
                (
                    pool_relevance == RELEVANT_CANDIDATE_IN_POOL
                    and accepted
                    and top1_relevant
                )
                or (
                    pool_relevance == NO_RELEVANT_CANDIDATE_IN_POOL
                    and not accepted
                )
            )
        paired_rows.append(row)

    relevant_in_pool_rows = [
        row
        for row in paired_rows
        if row["pool_relevance"] == RELEVANT_CANDIDATE_IN_POOL
    ]
    no_relevant_in_pool_rows = [
        row
        for row in paired_rows
        if row["pool_relevance"] == NO_RELEVANT_CANDIDATE_IN_POOL
    ]
    ranking_metrics: dict[str, dict[str, Any]] = {}
    for name in evaluated_run_names:
        ranking_metrics[name] = {
            "relevant_in_pool_query_count": len(relevant_in_pool_rows),
            "recall_at_1": mean(
                float(row[f"{name}_r_at_1"]) for row in relevant_in_pool_rows
            ),
            "recall_at_3": mean(
                float(row[f"{name}_r_at_3"]) for row in relevant_in_pool_rows
            ),
            "mrr_at_3": mean(
                float(row[f"{name}_mrr_at_3"])
                for row in relevant_in_pool_rows
            ),
        }

    pool_conditioned: dict[str, dict[str, Any]] = {}
    for version in SYSTEM_VERSIONS:
        accepted_relevant = sum(
            bool(row[f"{version}_accepted"]) for row in relevant_in_pool_rows
        )
        false_accepts = sum(
            bool(row[f"{version}_accepted"])
            for row in no_relevant_in_pool_rows
        )
        rejected_empty_pool = len(no_relevant_in_pool_rows) - false_accepts
        correct = sum(
            float(row[f"{version}_pool_conditioned_correct"])
            for row in paired_rows
        )
        accepted_total = sum(bool(row[f"{version}_accepted"]) for row in paired_rows)
        pool_conditioned[version] = {
            "relevant_in_pool_query_count": len(relevant_in_pool_rows),
            "relevant_in_pool_accepted_count": accepted_relevant,
            "relevant_in_pool_acceptance_rate": (
                accepted_relevant / len(relevant_in_pool_rows)
                if relevant_in_pool_rows
                else 0.0
            ),
            "no_relevant_in_pool_query_count": len(no_relevant_in_pool_rows),
            "no_relevant_in_pool_rejected_count": rejected_empty_pool,
            "no_relevant_in_pool_rejection_rate": (
                rejected_empty_pool / len(no_relevant_in_pool_rows)
                if no_relevant_in_pool_rows
                else 0.0
            ),
            "pool_conditioned_false_accept_count": false_accepts,
            "pool_conditioned_false_accept_rate": (
                false_accepts / len(no_relevant_in_pool_rows)
                if no_relevant_in_pool_rows
                else 0.0
            ),
            "accepted_query_count": accepted_total,
            "end_to_end_correct_count": int(correct),
            "pool_conditioned_end_to_end_accuracy": (
                correct / len(paired_rows) if paired_rows else 0.0
            ),
            "degenerate_reject_all": accepted_total == 0,
        }

    bootstrap = {
        "relevant_in_pool_recall_at_3_v17_minus_v16": grouped_paired_bootstrap(
            relevant_in_pool_rows,
            baseline_field="v16_quality_hybrid_r_at_3",
            contender_field="v17_quality_hybrid_r_at_3",
            repetitions=bootstrap_repetitions,
            seed=seed,
        ),
        "pool_conditioned_false_accept_v17_minus_v16": grouped_paired_bootstrap(
            no_relevant_in_pool_rows,
            baseline_field="v16_pool_conditioned_false_accept",
            contender_field="v17_pool_conditioned_false_accept",
            repetitions=bootstrap_repetitions,
            seed=seed,
        ),
        "end_to_end_correct_v17_minus_v16": grouped_paired_bootstrap(
            paired_rows,
            baseline_field="v16_pool_conditioned_correct",
            contender_field="v17_pool_conditioned_correct",
            repetitions=bootstrap_repetitions,
            seed=seed,
        ),
    }
    baseline_far = pool_conditioned["v16"]["pool_conditioned_false_accept_rate"]
    contender_far = pool_conditioned["v17"]["pool_conditioned_false_accept_rate"]
    relative_far_reduction = (
        (baseline_far - contender_far) / baseline_far if baseline_far else None
    )
    r3_delta = (
        ranking_metrics["v17_quality_hybrid"]["recall_at_3"]
        - ranking_metrics["v16_quality_hybrid"]["recall_at_3"]
    )
    report: dict[str, Any] = {
        "status": "calibration_only_model_assisted_labels",
        "task_id": POOLED_RELEVANCE_TASK,
        "label_provenance": {
            "artifact": (
                "data/evaluation/v17/human_study/calibration/"
                "adjudicated_judgments.jsonl"
            ),
            "type": "model_assisted_visual_audit",
            "human_gold": False,
            "warning": (
                "These labels preserve the human review separately but are "
                "model-assisted calibration labels, not an independent human gold "
                "standard."
            ),
        },
        "scope": {
            "candidate_pool_limit": 20,
            "supported_ranking_cutoffs": [1, 3],
            "holdout_read": False,
            "evaluated_query_count": len(paired_rows),
            "relevant_in_pool_query_count": len(relevant_in_pool_ids),
            "no_relevant_in_pool_query_count": len(no_relevant_in_pool_ids),
            "corpus_answerability_evaluated": False,
            "excluded_query_ids": excluded_query_ids,
            "evaluated_run_names": list(evaluated_run_names),
        },
        "ranking_metrics": ranking_metrics,
        "pool_conditioned_selective_retrieval": pool_conditioned,
        "paired_group_bootstrap": bootstrap,
        "research_hypotheses": {
            "h1_pool_conditioned_false_accept_relative_reduction_at_least_30_percent": {
                "relative_reduction": relative_far_reduction,
                "point_estimate_passed": bool(
                    relative_far_reduction is not None
                    and relative_far_reduction >= 0.30
                ),
                "valid_success": bool(
                    relative_far_reduction is not None
                    and relative_far_reduction >= 0.30
                    and not pool_conditioned["v17"]["degenerate_reject_all"]
                ),
            },
            "h2_relevant_in_pool_recall_at_3_loss_no_more_than_3pp": {
                "difference_v17_minus_v16": r3_delta,
                "point_estimate_passed": r3_delta >= -0.03,
            },
        },
        "diagnosis": {
            "ranking_improved": r3_delta > 0.0,
            "acceptance_guard_degenerate": pool_conditioned["v17"][
                "degenerate_reject_all"
            ],
            "conclusion": (
                "V17 improves candidate ranking on calibration, but its current "
                "attribute acceptance guard rejects every evaluated query. H1 is "
                "therefore not a valid success and the guard must be revised before "
                "freezing."
                if pool_conditioned["v17"]["degenerate_reject_all"]
                else "V17 does not exhibit reject-all degeneration on calibration."
            ),
        },
        "bootstrap_configuration": {
            "repetitions": bootstrap_repetitions,
            "seed": seed,
            "sampling_unit": "group_id",
        },
    }
    return report, paired_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--judgments",
        type=Path,
        default=Path(
            "data/evaluation/v17/human_study/calibration/adjudicated_judgments.jsonl"
        ),
    )
    parser.add_argument(
        "--packets",
        type=Path,
        default=Path(
            "data/evaluation/v17/human_study/calibration/review_packets.jsonl"
        ),
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("outputs/evaluation/v17/calibration/runs"),
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("outputs/evaluation/v17/calibration/raw"),
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        choices=RUN_NAMES,
        default=RUN_NAMES,
        help="Run files to evaluate; V16/V17 quality-hybrid runs are required.",
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/evaluation/v17/calibration/adjudicated_calibration_report.json"
        ),
    )
    parser.add_argument(
        "--paired-output",
        type=Path,
        default=Path(
            "outputs/evaluation/v17/calibration/adjudicated_paired_records.jsonl"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    judgments = read_jsonl(project_path(args.judgments))
    packets = read_jsonl(project_path(args.packets))
    run_dir = project_path(args.run_dir)
    raw_dir = project_path(args.raw_dir)
    report, paired_rows = evaluate(
        judgments=judgments,
        packets=packets,
        run_rows={name: read_jsonl(run_dir / f"{name}.jsonl") for name in args.runs},
        raw_dir=raw_dir,
        bootstrap_repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )
    output = project_path(args.output)
    paired_output = project_path(args.paired_output)
    write_json_atomic(output, report)
    write_jsonl_atomic(paired_output, paired_rows)
    print(output)
    print(paired_output)


if __name__ == "__main__":
    main()
