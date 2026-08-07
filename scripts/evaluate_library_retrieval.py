"""Select and evaluate the reviewed 1,500-page multimodal retriever.

The selection workflow keeps the test split sealed:
1. tune fusion parameters on train;
2. compare family winners on validation;
3. save one frozen ranker plus one train-fitted open-set gate;
4. evaluate test only in a separate ``--mode test`` invocation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.bm25_retrieval import bm25_query_weight  # noqa: E402
from scripts.query_routing import infer_retrieval_route  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("select", "test"), default="select"
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path(
            "data/evaluation/"
            "public_dataset_1500_retrieval_queries_formal.csv"
        ),
    )
    parser.add_argument(
        "--text-scores",
        type=Path,
        default=Path(
            "outputs/evaluation/library_retrieval/"
            "dev_text_bm25_scores.npz"
        ),
    )
    parser.add_argument(
        "--visual-scores",
        type=Path,
        default=Path(
            "outputs/evaluation/library_retrieval/"
            "dev_visual_scores.npz"
        ),
    )
    parser.add_argument(
        "--ocr-summary",
        type=Path,
        default=Path("outputs/user_library/ocr/summary.csv"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "outputs/evaluation/library_retrieval/"
            "selected_retrieval_config.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/evaluation/library_retrieval/"
            "development_selection.json"
        ),
    )
    return parser.parse_args()


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_queries(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["query_id"]: row for row in rows}


def read_ocr_confidences(
    path: Path, item_ids: list[str]
) -> np.ndarray:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    values = {
        row["item_id"]: float(row.get("mean_confidence") or 0.0)
        for row in rows
    }
    return np.asarray(
        [values.get(item_id, 0.0) for item_id in item_ids],
        dtype=np.float32,
    )


def align_archive(
    archive: Any,
    *,
    score_key: str,
    query_ids: list[str],
    item_ids: list[str],
) -> np.ndarray:
    source_queries = archive["query_ids"].tolist()
    source_items = archive["item_ids"].tolist()
    query_lookup = {
        query_id: index
        for index, query_id in enumerate(source_queries)
    }
    item_lookup = {
        item_id: index for index, item_id in enumerate(source_items)
    }
    missing_queries = set(query_ids) - set(query_lookup)
    missing_items = set(item_ids) - set(item_lookup)
    if missing_queries or missing_items:
        raise ValueError(
            "Score archive alignment failed: "
            f"{len(missing_queries)} queries and "
            f"{len(missing_items)} items are missing."
        )
    query_order = [query_lookup[value] for value in query_ids]
    item_order = [item_lookup[value] for value in item_ids]
    return archive[score_key][np.ix_(query_order, item_order)].astype(
        np.float32
    )


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    minimum = matrix.min(axis=1, keepdims=True)
    maximum = matrix.max(axis=1, keepdims=True)
    return (matrix - minimum) / np.maximum(maximum - minimum, 1e-8)


def reciprocal_rank_matrix(
    matrix: np.ndarray,
    *,
    k: int = 60,
    top_n: int = 60,
    positive_only: bool = False,
) -> np.ndarray:
    output = np.zeros_like(matrix, dtype=np.float32)
    count = min(top_n, matrix.shape[1])
    for row_index, row in enumerate(matrix):
        order = np.argsort(-row, kind="stable")[:count]
        for rank, column in enumerate(order, start=1):
            if positive_only and float(row[column]) <= 0.0:
                continue
            output[row_index, column] = 1.0 / (k + rank)
    return output


def score_for_config(
    config: dict[str, Any],
    *,
    dense_raw: np.ndarray,
    bm25_raw: np.ndarray,
    visual_raw: np.ndarray,
    metadata_raw: np.ndarray,
    confidences: np.ndarray,
    queries: list[dict[str, str]],
) -> np.ndarray:
    dense = normalize_rows(dense_raw)
    bm25 = normalize_rows(bm25_raw)
    visual = normalize_rows(visual_raw)
    metadata = normalize_rows(metadata_raw)
    family = config["family"]
    params = config.get("params", {})

    if family == "dense":
        return dense
    if family == "bm25":
        return bm25
    if family == "visual":
        return visual
    if family == "metadata":
        return metadata
    if family == "fixed":
        alpha = float(params["text_weight"])
        return alpha * dense + (1.0 - alpha) * visual
    if family == "adaptive":
        cap = float(params["text_weight_cap"])
        sparse_scale = float(params["sparse_scale"])
        text_weights = np.clip(cap * confidences, 0.0, 1.0)
        sparse_weights = np.asarray(
            [bm25_query_weight(row["query"]) for row in queries],
            dtype=np.float32,
        )[:, None]
        return (
            dense * text_weights[None, :]
            + visual * (1.0 - text_weights[None, :])
            + sparse_scale * sparse_weights * bm25
        )
    if family in {"rrf", "quality_rrf"}:
        dense_rrf = reciprocal_rank_matrix(dense_raw)
        bm25_rrf = reciprocal_rank_matrix(
            bm25_raw, positive_only=True
        )
        visual_rrf = reciprocal_rank_matrix(visual_raw)
        if family == "rrf":
            return dense_rrf + bm25_rrf + visual_rrf
        quality = np.clip(confidences, 0.0, 1.0)
        return (
            0.5 * quality[None, :] * dense_rrf
            + 0.5 * quality[None, :] * bm25_rrf
            + (1.0 - 0.5 * quality[None, :]) * visual_rrf
        )
    if family == "query_aware":
        text_bm25_weight = float(params["text_bm25_weight"])
        visual_weight = float(params["visual_weight"])
        cap = float(params["text_weight_cap"])
        sparse_scale = float(params["sparse_scale"])
        text_weights = np.clip(cap * confidences, 0.0, 1.0)
        output = np.empty_like(dense, dtype=np.float32)
        for index, row in enumerate(queries):
            route = infer_retrieval_route(row["query"])
            if route == "text_evidence":
                output[index] = (
                    dense[index]
                    + text_bm25_weight * bm25[index]
                )
            elif route == "visual_metadata":
                output[index] = (
                    visual_weight * visual[index]
                    + (1.0 - visual_weight) * metadata[index]
                )
            else:
                sparse_weight = (
                    sparse_scale * bm25_query_weight(row["query"])
                )
                output[index] = (
                    dense[index] * text_weights
                    + visual[index] * (1.0 - text_weights)
                    + sparse_weight * bm25[index]
                )
        return output
    raise ValueError(f"Unknown retrieval family: {family}")


def is_no_answer(row: dict[str, str]) -> bool:
    return (
        row.get("is_no_answer", "").strip().lower() == "true"
        or row.get("review_decision") == "no_answer"
    )


def relevant_ids(row: dict[str, str]) -> set[str]:
    values = {
        value.strip()
        for value in row.get("relevant_item_ids", "").split(";")
        if value.strip()
    }
    if not values and row.get("expected_item_id", "").strip():
        values.add(row["expected_item_id"].strip())
    return values


def rank_of_relevant(
    score_row: np.ndarray,
    item_ids: list[str],
    relevant: set[str],
) -> tuple[int | None, list[str]]:
    order = np.argsort(-score_row, kind="stable")
    top_ids = [item_ids[int(index)] for index in order[:5]]
    if not relevant:
        return None, top_ids
    rank = next(
        (
            position
            for position, index in enumerate(order, start=1)
            if item_ids[int(index)] in relevant
        ),
        None,
    )
    return rank, top_ids


def ndcg_at_10(
    score_row: np.ndarray,
    item_ids: list[str],
    relevant: set[str],
) -> float:
    if not relevant:
        return 0.0
    order = np.argsort(-score_row, kind="stable")[:10]
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, index in enumerate(order, start=1)
        if item_ids[int(index)] in relevant
    )
    ideal_hits = min(len(relevant), 10)
    ideal = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )
    return dcg / ideal


def gate_features(
    dense_raw: np.ndarray,
    bm25_raw: np.ndarray,
    visual_raw: np.ndarray,
) -> np.ndarray:
    features: list[list[float]] = []
    for dense, bm25, visual in zip(dense_raw, bm25_raw, visual_raw):
        dense_top = np.sort(dense)[-2:]
        bm25_top = np.sort(bm25)[-2:]
        visual_top = np.sort(visual)[-2:]
        features.append(
            [
                float(dense_top[-1]),
                float(visual_top[-1]),
                math.log1p(max(float(bm25_top[-1]), 0.0)),
                float(dense_top[-1] - dense_top[-2]),
                float(visual_top[-1] - visual_top[-2]),
                math.log1p(
                    max(float(bm25_top[-1] - bm25_top[-2]), 0.0)
                ),
            ]
        )
    return np.asarray(features, dtype=np.float64)


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-values))


def fit_open_set_gate(
    features: np.ndarray, labels: np.ndarray
) -> dict[str, Any]:
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-8] = 1.0
    x = (features - mean) / scale
    x = np.column_stack([np.ones(len(x)), x])
    labels = labels.astype(np.float64)
    positives = max(float(labels.sum()), 1.0)
    negatives = max(float((1.0 - labels).sum()), 1.0)
    sample_weights = np.where(
        labels > 0.5,
        0.5 / positives,
        0.5 / negatives,
    )
    coefficients = np.zeros(x.shape[1], dtype=np.float64)
    learning_rate = 0.08
    regularization = 0.03
    for _ in range(5000):
        probabilities = sigmoid(x @ coefficients)
        gradient = x.T @ (
            sample_weights * (probabilities - labels)
        )
        gradient[1:] += regularization * coefficients[1:]
        coefficients -= learning_rate * gradient

    probabilities = sigmoid(x @ coefficients)
    ordered_probabilities = np.sort(probabilities)
    thresholds = sorted(
        {
            0.0,
            1.0,
            *ordered_probabilities.tolist(),
            *(
                (
                    ordered_probabilities[:-1]
                    + ordered_probabilities[1:]
                )
                / 2.0
            ).tolist(),
        }
    )
    best: tuple[float, float, float] | None = None
    selected = 0.5
    for threshold in thresholds:
        accepted = probabilities >= threshold
        true_positive_rate = float(accepted[labels == 1].mean())
        true_negative_rate = float((~accepted[labels == 0]).mean())
        balanced = 0.5 * (true_positive_rate + true_negative_rate)
        accuracy = float((accepted == (labels == 1)).mean())
        key = (balanced, accuracy, threshold)
        if best is None or key > best:
            best = key
            selected = float(threshold)
    return {
        "feature_names": [
            "dense_top1",
            "visual_top1",
            "log1p_bm25_top1",
            "dense_margin",
            "visual_margin",
            "log1p_bm25_margin",
        ],
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "coefficients": coefficients.tolist(),
        "threshold": selected,
        "train_balanced_accuracy": round(best[0], 6),
        "train_accuracy": round(best[1], 6),
    }


def apply_open_set_gate(
    gate: dict[str, Any], features: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(gate["mean"], dtype=np.float64)
    scale = np.asarray(gate["scale"], dtype=np.float64)
    coefficients = np.asarray(
        gate["coefficients"], dtype=np.float64
    )
    standardized = (features - mean) / scale
    x = np.column_stack([np.ones(len(standardized)), standardized])
    probabilities = sigmoid(x @ coefficients)
    return probabilities >= float(gate["threshold"]), probabilities


def evaluate(
    score_matrix: np.ndarray,
    *,
    queries: list[dict[str, str]],
    item_ids: list[str],
    accepted: np.ndarray,
    acceptance_probabilities: np.ndarray,
) -> dict[str, Any]:
    answerable_ranks: list[int | None] = []
    ndcgs: list[float] = []
    details: list[dict[str, Any]] = []
    answerable_acceptance: list[bool] = []
    no_answer_rejections: list[bool] = []
    end_to_end: list[bool] = []

    for index, (row, scores) in enumerate(zip(queries, score_matrix)):
        no_answer = is_no_answer(row)
        relevant = relevant_ids(row)
        rank, top_ids = rank_of_relevant(scores, item_ids, relevant)
        query_accepted = bool(accepted[index])
        if no_answer:
            no_answer_rejections.append(not query_accepted)
            end_to_end.append(not query_accepted)
        else:
            answerable_ranks.append(rank)
            ndcgs.append(ndcg_at_10(scores, item_ids, relevant))
            answerable_acceptance.append(query_accepted)
            end_to_end.append(query_accepted and rank == 1)
        details.append(
            {
                "query_id": row["query_id"],
                "split": row["split"],
                "is_no_answer": no_answer,
                "accepted": query_accepted,
                "acceptance_probability": round(
                    float(acceptance_probabilities[index]), 6
                ),
                "expected_rank": rank,
                "top_5_item_ids": top_ids,
                "relevant_item_ids": sorted(relevant),
            }
        )

    answerable_count = len(answerable_ranks)
    no_answer_count = len(no_answer_rejections)
    recall_1 = (
        sum(rank == 1 for rank in answerable_ranks) / answerable_count
        if answerable_count
        else 0.0
    )
    recall_3 = (
        sum(
            rank is not None and rank <= 3
            for rank in answerable_ranks
        )
        / answerable_count
        if answerable_count
        else 0.0
    )
    recall_5 = (
        sum(
            rank is not None and rank <= 5
            for rank in answerable_ranks
        )
        / answerable_count
        if answerable_count
        else 0.0
    )
    mrr = (
        sum(1.0 / rank if rank else 0.0 for rank in answerable_ranks)
        / answerable_count
        if answerable_count
        else 0.0
    )
    coverage = (
        sum(answerable_acceptance) / answerable_count
        if answerable_count
        else 0.0
    )
    rejection_accuracy = (
        sum(no_answer_rejections) / no_answer_count
        if no_answer_count
        else 0.0
    )
    return {
        "query_count": len(queries),
        "answerable_count": answerable_count,
        "no_answer_count": no_answer_count,
        "recall_at_1": round(recall_1, 6),
        "recall_at_3": round(recall_3, 6),
        "recall_at_5": round(recall_5, 6),
        "mrr": round(mrr, 6),
        "ndcg_at_10": round(
            float(np.mean(ndcgs)) if ndcgs else 0.0, 6
        ),
        "answerable_acceptance_rate": round(coverage, 6),
        "no_answer_rejection_accuracy": round(
            rejection_accuracy, 6
        ),
        "false_accept_rate": round(1.0 - rejection_accuracy, 6),
        "end_to_end_accuracy": round(
            sum(end_to_end) / len(end_to_end), 6
        ),
        "details": details,
    }


def candidate_configs() -> dict[str, list[dict[str, Any]]]:
    return {
        "dense": [{"family": "dense", "params": {}}],
        "bm25": [{"family": "bm25", "params": {}}],
        "visual": [{"family": "visual", "params": {}}],
        "metadata": [{"family": "metadata", "params": {}}],
        "fixed": [
            {
                "family": "fixed",
                "params": {"text_weight": value},
            }
            for value in (0.2, 0.35, 0.5, 0.65, 0.8)
        ],
        "adaptive": [
            {
                "family": "adaptive",
                "params": {
                    "text_weight_cap": cap,
                    "sparse_scale": sparse_scale,
                },
            }
            for cap in (0.4, 0.6, 0.8, 1.0)
            for sparse_scale in (0.0, 0.5, 1.0, 1.5, 2.0)
        ],
        "rrf": [{"family": "rrf", "params": {}}],
        "quality_rrf": [
            {"family": "quality_rrf", "params": {}}
        ],
        "query_aware": [
            {
                "family": "query_aware",
                "params": {
                    "text_bm25_weight": text_bm25_weight,
                    "visual_weight": visual_weight,
                    "text_weight_cap": 0.6,
                    "sparse_scale": 0.5,
                },
            }
            for text_bm25_weight in (0.25, 0.5, 1.0)
            for visual_weight in (0.25, 0.5, 0.75)
        ],
    }


def ranker_objective(metrics: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(metrics["recall_at_1"]),
        float(metrics["recall_at_3"]),
        float(metrics["mrr"]),
        float(metrics["ndcg_at_10"]),
    )


def selection_objective(metrics: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(metrics["end_to_end_accuracy"]),
        float(metrics["recall_at_1"]),
        float(metrics["mrr"]),
        float(metrics["no_answer_rejection_accuracy"]),
    )


def load_inputs(
    args: argparse.Namespace,
) -> tuple[
    list[dict[str, str]],
    list[str],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    text_archive = np.load(project_path(args.text_scores))
    visual_archive = np.load(project_path(args.visual_scores))
    query_ids = text_archive["query_ids"].tolist()
    visual_query_ids = visual_archive["query_ids"].tolist()
    if set(query_ids) != set(visual_query_ids):
        raise ValueError("Text and visual archives contain different queries.")
    query_lookup = read_queries(project_path(args.queries))
    missing = set(query_ids) - set(query_lookup)
    if missing:
        raise ValueError(f"Unknown query IDs: {sorted(missing)}")
    queries = [query_lookup[query_id] for query_id in query_ids]
    item_ids = text_archive["item_ids"].tolist()
    dense = text_archive["dense_scores"].astype(np.float32)
    bm25 = text_archive["bm25_scores"].astype(np.float32)
    if "metadata_scores" not in text_archive:
        raise ValueError(
            "Text score archive does not contain metadata_scores."
        )
    metadata = text_archive["metadata_scores"].astype(np.float32)
    visual = align_archive(
        visual_archive,
        score_key="visual_scores",
        query_ids=query_ids,
        item_ids=item_ids,
    )
    confidences = read_ocr_confidences(
        project_path(args.ocr_summary), item_ids
    )
    return (
        queries,
        item_ids,
        dense,
        bm25,
        visual,
        metadata,
        confidences,
    )


def subset(
    split: str,
    queries: list[dict[str, str]],
    *matrices: np.ndarray,
) -> tuple[list[dict[str, str]], list[np.ndarray]]:
    indices = [
        index
        for index, row in enumerate(queries)
        if row["split"] == split
    ]
    return (
        [queries[index] for index in indices],
        [matrix[indices] for matrix in matrices],
    )


def select_mode(args: argparse.Namespace) -> dict[str, Any]:
    (
        queries,
        item_ids,
        dense,
        bm25,
        visual,
        metadata,
        confidences,
    ) = load_inputs(args)
    present_splits = {row["split"] for row in queries}
    if present_splits != {"train", "validation"}:
        raise ValueError(
            "Selection requires exactly train and validation archives; "
            f"received {sorted(present_splits)}."
        )

    train_queries, train_matrices = subset(
        "train", queries, dense, bm25, visual, metadata
    )
    (
        train_dense,
        train_bm25,
        train_visual,
        train_metadata,
    ) = train_matrices
    validation_queries, validation_matrices = subset(
        "validation", queries, dense, bm25, visual, metadata
    )
    (
        validation_dense,
        validation_bm25,
        validation_visual,
        validation_metadata,
    ) = validation_matrices

    train_features = gate_features(
        train_dense, train_bm25, train_visual
    )
    train_labels = np.asarray(
        [not is_no_answer(row) for row in train_queries],
        dtype=np.int8,
    )
    gate = fit_open_set_gate(train_features, train_labels)
    train_accepted, train_probabilities = apply_open_set_gate(
        gate, train_features
    )
    validation_accepted, validation_probabilities = (
        apply_open_set_gate(
            gate,
            gate_features(
                validation_dense,
                validation_bm25,
                validation_visual,
            ),
        )
    )

    family_winners: list[dict[str, Any]] = []
    tuning_rows: list[dict[str, Any]] = []
    for family, configs in candidate_configs().items():
        family_best: tuple[tuple[float, ...], dict[str, Any]] | None = None
        for config in configs:
            scores = score_for_config(
                config,
                dense_raw=train_dense,
                bm25_raw=train_bm25,
                visual_raw=train_visual,
                metadata_raw=train_metadata,
                confidences=confidences,
                queries=train_queries,
            )
            metrics = evaluate(
                scores,
                queries=train_queries,
                item_ids=item_ids,
                accepted=train_accepted,
                acceptance_probabilities=train_probabilities,
            )
            compact_metrics = {
                key: value
                for key, value in metrics.items()
                if key != "details"
            }
            tuning_rows.append(
                {"config": config, "train_metrics": compact_metrics}
            )
            key = ranker_objective(metrics)
            if family_best is None or key > family_best[0]:
                family_best = (key, config)
        assert family_best is not None
        family_winners.append(family_best[1])

    finalist_rows: list[dict[str, Any]] = []
    selected: tuple[
        tuple[float, ...], dict[str, Any], dict[str, Any]
    ] | None = None
    for config in family_winners:
        train_scores = score_for_config(
            config,
            dense_raw=train_dense,
            bm25_raw=train_bm25,
            visual_raw=train_visual,
            metadata_raw=train_metadata,
            confidences=confidences,
            queries=train_queries,
        )
        validation_scores = score_for_config(
            config,
            dense_raw=validation_dense,
            bm25_raw=validation_bm25,
            visual_raw=validation_visual,
            metadata_raw=validation_metadata,
            confidences=confidences,
            queries=validation_queries,
        )
        train_metrics = evaluate(
            train_scores,
            queries=train_queries,
            item_ids=item_ids,
            accepted=train_accepted,
            acceptance_probabilities=train_probabilities,
        )
        validation_metrics = evaluate(
            validation_scores,
            queries=validation_queries,
            item_ids=item_ids,
            accepted=validation_accepted,
            acceptance_probabilities=validation_probabilities,
        )
        row = {
            "config": config,
            "train_metrics": {
                key: value
                for key, value in train_metrics.items()
                if key != "details"
            },
            "validation_metrics": {
                key: value
                for key, value in validation_metrics.items()
                if key != "details"
            },
            "validation_details": validation_metrics["details"],
        }
        finalist_rows.append(row)
        key = selection_objective(validation_metrics)
        if selected is None or key > selected[0]:
            selected = (key, config, row)
    assert selected is not None

    frozen_config = {
        "protocol": (
            "fusion parameters tuned on train; family selected on "
            "validation; test sealed until separate invocation"
        ),
        "ranker": selected[1],
        "open_set_gate": gate,
        "train_query_count": len(train_queries),
        "validation_query_count": len(validation_queries),
        "item_count": len(item_ids),
        "selection_metrics": selected[2]["validation_metrics"],
    }
    config_path = project_path(args.config)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(frozen_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "protocol": frozen_config["protocol"],
        "open_set_gate": gate,
        "tuning_candidates": tuning_rows,
        "family_finalists": finalist_rows,
        "selected_config": frozen_config,
    }


def test_mode(args: argparse.Namespace) -> dict[str, Any]:
    (
        queries,
        item_ids,
        dense,
        bm25,
        visual,
        metadata,
        confidences,
    ) = load_inputs(args)
    present_splits = {row["split"] for row in queries}
    if present_splits != {"test"}:
        raise ValueError(
            "Test mode accepts only a sealed test archive; "
            f"received {sorted(present_splits)}."
        )
    frozen = json.loads(
        project_path(args.config).read_text(encoding="utf-8")
    )
    accepted, probabilities = apply_open_set_gate(
        frozen["open_set_gate"],
        gate_features(dense, bm25, visual),
    )
    scores = score_for_config(
        frozen["ranker"],
        dense_raw=dense,
        bm25_raw=bm25,
        visual_raw=visual,
        metadata_raw=metadata,
        confidences=confidences,
        queries=queries,
    )
    metrics = evaluate(
        scores,
        queries=queries,
        item_ids=item_ids,
        accepted=accepted,
        acceptance_probabilities=probabilities,
    )
    return {
        "protocol": "single sealed test evaluation of frozen configuration",
        "frozen_config": frozen,
        "test_metrics": metrics,
    }


def main() -> None:
    args = parse_args()
    report = (
        select_mode(args) if args.mode == "select" else test_mode(args)
    )
    output_path = project_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if args.mode == "select":
        selected = report["selected_config"]
        metrics = selected["selection_metrics"]
        print(f"Selected ranker: {selected['ranker']}")
        print(
            "Validation: "
            f"R@1={metrics['recall_at_1']:.4f}, "
            f"R@3={metrics['recall_at_3']:.4f}, "
            f"MRR={metrics['mrr']:.4f}, "
            f"reject={metrics['no_answer_rejection_accuracy']:.4f}, "
            f"E2E={metrics['end_to_end_accuracy']:.4f}"
        )
    else:
        metrics = report["test_metrics"]
        print(
            "Sealed test: "
            f"R@1={metrics['recall_at_1']:.4f}, "
            f"R@3={metrics['recall_at_3']:.4f}, "
            f"MRR={metrics['mrr']:.4f}, "
            f"reject={metrics['no_answer_rejection_accuracy']:.4f}, "
            f"E2E={metrics['end_to_end_accuracy']:.4f}"
        )
    print(f"Report: {output_path}")


if __name__ == "__main__":
    main()
