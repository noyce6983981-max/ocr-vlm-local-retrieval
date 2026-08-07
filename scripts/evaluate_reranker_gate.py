"""Calibrate and validate the reranker-based open-set verification gate."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path(
            "data/evaluation/"
            "public_dataset_1500_retrieval_queries_formal.csv"
        ),
    )
    parser.add_argument(
        "--reranker-scores",
        type=Path,
        default=Path(
            "outputs/evaluation/library_retrieval/"
            "dev_v2_reranker_scores.json"
        ),
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path(
            "outputs/evaluation/library_retrieval/"
            "selected_retrieval_config_v2_candidate.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/evaluation/library_retrieval/"
            "reranker_gate_evaluation.json"
        ),
    )
    parser.add_argument(
        "--final-config",
        type=Path,
        default=Path(
            "outputs/evaluation/library_retrieval/"
            "selected_retrieval_config_v2_final.json"
        ),
    )
    return parser.parse_args()


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_queries(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            row["query_id"]: row for row in csv.DictReader(handle)
        }


def is_no_answer(row: dict[str, str]) -> bool:
    return row.get("is_no_answer", "").strip().lower() == "true"


def relevant_ids(row: dict[str, str]) -> set[str]:
    values = {
        value.strip()
        for value in row.get("relevant_item_ids", "").split(";")
        if value.strip()
    }
    if not values and row.get("expected_item_id", "").strip():
        values.add(row["expected_item_id"].strip())
    return values


def reranked_rank(
    result: dict[str, Any], query: dict[str, str]
) -> int | None:
    relevant = relevant_ids(query)
    return next(
        (
            rank
            for rank, item_id in enumerate(
                result["reranked_item_ids"], start=1
            )
            if item_id in relevant
        ),
        None,
    )


def choose_threshold(
    train_rows: list[dict[str, Any]],
    queries: dict[str, dict[str, str]],
) -> tuple[float, dict[str, float]]:
    scores = sorted({float(row["top_score"]) for row in train_rows})
    candidates = [0.0, 1.0, *scores]
    best: tuple[float, float, float] | None = None
    selected = 0.5
    selected_metrics: dict[str, float] = {}
    for threshold in candidates:
        positives = [
            float(row["top_score"]) >= threshold
            for row in train_rows
            if not is_no_answer(queries[row["query_id"]])
        ]
        negatives = [
            float(row["top_score"]) < threshold
            for row in train_rows
            if is_no_answer(queries[row["query_id"]])
        ]
        positive_rate = sum(positives) / len(positives)
        negative_rate = sum(negatives) / len(negatives)
        balanced = 0.5 * (positive_rate + negative_rate)
        accuracy = (
            sum(positives) + sum(negatives)
        ) / (len(positives) + len(negatives))
        key = (balanced, accuracy, threshold)
        if best is None or key > best:
            best = key
            selected = threshold
            selected_metrics = {
                "answerable_acceptance_rate": round(
                    positive_rate, 6
                ),
                "no_answer_rejection_accuracy": round(
                    negative_rate, 6
                ),
                "balanced_accuracy": round(balanced, 6),
                "accuracy": round(accuracy, 6),
            }
    return selected, selected_metrics


def evaluate_split(
    rows: list[dict[str, Any]],
    queries: dict[str, dict[str, str]],
    threshold: float,
) -> dict[str, Any]:
    answerable = [
        row
        for row in rows
        if not is_no_answer(queries[row["query_id"]])
    ]
    negatives = [
        row
        for row in rows
        if is_no_answer(queries[row["query_id"]])
    ]
    ranks = [
        reranked_rank(row, queries[row["query_id"]])
        for row in answerable
    ]
    answerable_accepted = [
        float(row["top_score"]) >= threshold for row in answerable
    ]
    negatives_rejected = [
        float(row["top_score"]) < threshold for row in negatives
    ]
    end_to_end = [
        accepted and rank == 1
        for accepted, rank in zip(answerable_accepted, ranks)
    ] + negatives_rejected
    return {
        "query_count": len(rows),
        "answerable_count": len(answerable),
        "no_answer_count": len(negatives),
        "recall_at_1": round(
            sum(rank == 1 for rank in ranks) / len(ranks), 6
        ),
        "recall_at_3": round(
            sum(rank is not None and rank <= 3 for rank in ranks)
            / len(ranks),
            6,
        ),
        "answerable_acceptance_rate": round(
            sum(answerable_accepted) / len(answerable_accepted), 6
        ),
        "no_answer_rejection_accuracy": round(
            sum(negatives_rejected) / len(negatives_rejected), 6
        ),
        "false_accept_rate": round(
            1.0
            - sum(negatives_rejected) / len(negatives_rejected),
            6,
        ),
        "end_to_end_accuracy": round(
            sum(end_to_end) / len(end_to_end), 6
        ),
        "details": [
            {
                "query_id": row["query_id"],
                "is_no_answer": is_no_answer(
                    queries[row["query_id"]]
                ),
                "top_score": row["top_score"],
                "accepted": float(row["top_score"]) >= threshold,
                "expected_rank": (
                    None
                    if is_no_answer(queries[row["query_id"]])
                    else reranked_rank(
                        row, queries[row["query_id"]]
                    )
                ),
                "top_item_id": row["reranked_item_ids"][0],
            }
            for row in rows
        ],
    }


def main() -> None:
    args = parse_args()
    queries = read_queries(project_path(args.queries))
    reranker_payload = json.loads(
        project_path(args.reranker_scores).read_text(encoding="utf-8")
    )
    rows = reranker_payload["results"]
    train_rows = [row for row in rows if row["split"] == "train"]
    validation_rows = [
        row for row in rows if row["split"] == "validation"
    ]
    threshold, threshold_train_metrics = choose_threshold(
        train_rows, queries
    )
    report = {
        "protocol": (
            "threshold calibrated on train only; validation used once "
            "for candidate verification; sealed test not reused"
        ),
        "model": reranker_payload["model"],
        "top_k": reranker_payload["top_k"],
        "threshold": round(threshold, 8),
        "threshold_train_metrics": threshold_train_metrics,
        "train_metrics": evaluate_split(
            train_rows, queries, threshold
        ),
        "validation_metrics": evaluate_split(
            validation_rows, queries, threshold
        ),
        "runtime": {
            "model_load_seconds": reranker_payload["load_seconds"],
            "batch_inference_seconds": reranker_payload[
                "inference_seconds"
            ],
            "peak_reserved_gib": reranker_payload[
                "peak_reserved_gib"
            ],
        },
    }
    output_path = project_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    final_config = json.loads(
        project_path(args.base_config).read_text(encoding="utf-8")
    )
    final_config["protocol"] = (
        "query-aware ranker selected on validation; reranker threshold "
        "calibrated on train and verified on validation; old sealed test "
        "was not reused after v2 development"
    )
    final_config["reranker_gate"] = {
        "enabled_for_precision_mode": True,
        "model": reranker_payload["model"],
        "top_k": reranker_payload["top_k"],
        "threshold": round(threshold, 8),
        "instruction": (
            "full multimodal match required; partial resemblance "
            "is insufficient"
        ),
    }
    final_config["precision_validation_metrics"] = {
        key: value
        for key, value in report["validation_metrics"].items()
        if key != "details"
    }
    final_config_path = project_path(args.final_config)
    final_config_path.write_text(
        json.dumps(final_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    metrics = report["validation_metrics"]
    print(
        f"threshold={threshold:.4f}, "
        f"validation R@1={metrics['recall_at_1']:.4f}, "
        f"reject={metrics['no_answer_rejection_accuracy']:.4f}, "
        f"E2E={metrics['end_to_end_accuracy']:.4f}"
    )
    print(output_path)
    print(final_config_path)


if __name__ == "__main__":
    main()
