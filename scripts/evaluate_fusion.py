"""Compare fixed and OCR-quality-aware fusion without loading any model."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate multimodal score fusion.")
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("data/evaluation/retrieval_queries.csv"),
    )
    parser.add_argument(
        "--ocr-summary",
        type=Path,
        default=Path("outputs/ocr_batch/summary.csv"),
    )
    parser.add_argument(
        "--text-scores",
        type=Path,
        default=Path("outputs/evaluation/text_score_matrix.npz"),
    )
    parser.add_argument(
        "--visual-scores",
        type=Path,
        default=Path("outputs/evaluation/visual_score_matrix.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/evaluation/fusion_retrieval.json"),
    )
    parser.add_argument(
        "--fixed-text-weight",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--adaptive-text-weight-cap",
        type=float,
        default=0.6,
        help="Candidate text weight = cap * mean OCR confidence.",
    )
    return parser.parse_args()


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_queries(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    minimum = matrix.min(axis=1, keepdims=True)
    maximum = matrix.max(axis=1, keepdims=True)
    scale = np.maximum(maximum - minimum, 1e-8)
    return (matrix - minimum) / scale


def align_matrix(
    archive: Any,
    query_ids: list[str],
    item_ids: list[str],
) -> np.ndarray:
    source_queries = archive["query_ids"].tolist()
    source_items = archive["item_ids"].tolist()
    query_order = [source_queries.index(query_id) for query_id in query_ids]
    item_order = [source_items.index(item_id) for item_id in item_ids]
    return archive["scores"][np.ix_(query_order, item_order)].astype(np.float32)


def relevant_item_ids(query: dict[str, str]) -> set[str]:
    values = [
        value.strip()
        for value in query.get("relevant_item_ids", "").split(";")
        if value.strip()
    ]
    return set(values or [query["expected_item_id"]])


def evaluate(
    name: str,
    score_matrix: np.ndarray,
    queries: list[dict[str, str]],
    item_ids: list[str],
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for query, scores in zip(queries, score_matrix):
        order = np.argsort(-scores)
        relevant = relevant_item_ids(query)
        expected_rank = next(
            rank
            for rank, index in enumerate(order, start=1)
            if item_ids[int(index)] in relevant
        )
        results.append(
            {
                "query_id": query["query_id"],
                "expected_item_id": query["expected_item_id"],
                "relevant_item_ids": sorted(relevant),
                "expected_rank": expected_rank,
                "top_3_item_ids": [item_ids[index] for index in order[:3]],
            }
        )

    ranks = [row["expected_rank"] for row in results]
    metrics = {
        "query_count": len(results),
        "recall_at_1": round(sum(rank == 1 for rank in ranks) / len(ranks), 6),
        "recall_at_3": round(sum(rank <= 3 for rank in ranks) / len(ranks), 6),
        "mrr": round(statistics.fmean(1.0 / rank for rank in ranks), 6),
    }
    return {"name": name, "metrics": metrics, "results": results}


def main() -> None:
    args = parse_args()
    queries = read_queries(project_path(args.queries))
    query_ids = [row["query_id"] for row in queries]

    with project_path(args.ocr_summary).open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        ocr_rows = list(csv.DictReader(handle))
    item_ids = [row["item_id"] for row in ocr_rows]
    confidences = np.array(
        [float(row["mean_confidence"]) for row in ocr_rows], dtype=np.float32
    )

    text_archive = np.load(project_path(args.text_scores))
    visual_archive = np.load(project_path(args.visual_scores))
    text = normalize_rows(
        align_matrix(text_archive, query_ids, item_ids)
    )
    visual = normalize_rows(
        align_matrix(visual_archive, query_ids, item_ids)
    )

    fixed_weight = args.fixed_text_weight
    fixed_scores = fixed_weight * text + (1.0 - fixed_weight) * visual

    # The rule is fixed before looking at fusion results. High-confidence OCR
    # can contribute at most 0.6; no-text images rely entirely on vision.
    text_weights = np.clip(
        args.adaptive_text_weight_cap * confidences, 0.0, 1.0
    )
    adaptive_scores = (
        text * text_weights[None, :]
        + visual * (1.0 - text_weights[None, :])
    )

    report = {
        "settings": {
            "normalization": "per-query min-max",
            "fixed_text_weight": fixed_weight,
            "adaptive_rule": (
                "candidate_text_weight = "
                f"{args.adaptive_text_weight_cap} * mean_ocr_confidence"
            ),
            "adaptive_text_weights": {
                item_id: round(float(weight), 6)
                for item_id, weight in zip(item_ids, text_weights)
            },
        },
        "fixed_fusion": evaluate(
            "fixed_fusion", fixed_scores, queries, item_ids
        ),
        "quality_adaptive_fusion": evaluate(
            "quality_adaptive_fusion", adaptive_scores, queries, item_ids
        ),
    }
    output_path = project_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for key in ("fixed_fusion", "quality_adaptive_fusion"):
        metrics = report[key]["metrics"]
        print(
            f"{key}: Recall@1={metrics['recall_at_1']:.4f}, "
            f"Recall@3={metrics['recall_at_3']:.4f}, "
            f"MRR={metrics['mrr']:.4f}"
        )
        failures = [
            row
            for row in report[key]["results"]
            if row["expected_rank"] != 1
        ]
        for row in failures:
            print(
                f"  {row['query_id']}: rank={row['expected_rank']}, "
                f"top1={row['top_3_item_ids'][0]}"
            )
    print(f"Report: {output_path}")


if __name__ == "__main__":
    main()
