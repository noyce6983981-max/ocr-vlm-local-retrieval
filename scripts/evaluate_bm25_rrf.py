"""Evaluate BM25 and three-route RRF on the reviewed dataset_v1 queries."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.bm25_retrieval import (  # noqa: E402
    bm25_query_weight,
    build_bm25_payload,
    score_bm25,
)
from scripts.demo_backend import (  # noqa: E402
    align_score_matrix,
    build_method_scores,
    build_rrf_scores,
    retrieval_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifest/dataset_v1_manifest.jsonl"),
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("data/evaluation/dataset_v1_queries.csv"),
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/processed/dataset_v1_corpus.jsonl"),
    )
    parser.add_argument(
        "--ocr-summary",
        type=Path,
        default=Path("outputs/ocr_dataset_v1/summary.csv"),
    )
    parser.add_argument(
        "--text-scores",
        type=Path,
        default=Path(
            "outputs/evaluation_dataset_v1/text_score_matrix.npz"
        ),
    )
    parser.add_argument(
        "--visual-scores",
        type=Path,
        default=Path(
            "outputs/evaluation_dataset_v1/visual_score_matrix.npz"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/evaluation_dataset_v1/bm25_rrf_retrieval.json"
        ),
    )
    return parser.parse_args()


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def expected_rank(
    scores: np.ndarray, expected_index: int
) -> int:
    order = np.argsort(-scores, kind="stable")
    return int(np.where(order == expected_index)[0][0]) + 1


def main() -> None:
    args = parse_args()
    manifest = read_jsonl(project_path(args.manifest))
    queries = read_csv(project_path(args.queries))
    corpus = read_jsonl(project_path(args.corpus))
    ocr_rows = read_csv(project_path(args.ocr_summary))
    item_ids = [row["item_id"] for row in manifest]
    query_ids = [row["query_id"] for row in queries]
    expected_ids = [row["expected_item_id"] for row in queries]
    item_columns = {item_id: index for index, item_id in enumerate(item_ids)}

    bm25_index = build_bm25_payload(
        [str(row["text"]) for row in corpus]
    )
    bm25_matrix = np.zeros(
        (len(queries), len(item_ids)), dtype=np.float32
    )
    for query_index, query in enumerate(queries):
        chunk_scores = score_bm25(query["query"], bm25_index)
        for chunk, score in zip(corpus, chunk_scores):
            column = item_columns[chunk["item_id"]]
            bm25_matrix[query_index, column] = max(
                bm25_matrix[query_index, column], float(score)
            )

    with np.load(project_path(args.text_scores)) as archive:
        dense_matrix = align_score_matrix(
            archive, query_ids, item_ids
        )
    with np.load(project_path(args.visual_scores)) as archive:
        visual_matrix = align_score_matrix(
            archive, query_ids, item_ids
        )
    confidence_by_id = {
        row["item_id"]: float(row["mean_confidence"])
        for row in ocr_rows
    }
    confidences = np.array(
        [confidence_by_id.get(item_id, 0.0) for item_id in item_ids],
        dtype=np.float32,
    )
    baseline_scores, _ = build_method_scores(
        dense_matrix, visual_matrix, confidences
    )
    rrf_matrix = np.zeros_like(dense_matrix)
    quality_rrf_matrix = np.zeros_like(dense_matrix)
    quality_hybrid_matrix = np.zeros_like(dense_matrix)
    for query_index in range(len(queries)):
        methods, _ = build_rrf_scores(
            dense_matrix[query_index],
            bm25_matrix[query_index],
            visual_matrix[query_index],
            confidences,
            top_n=len(item_ids),
        )
        rrf_matrix[query_index] = methods["rrf"]
        quality_rrf_matrix[query_index] = methods["quality_rrf"]
        normalized_bm25 = bm25_matrix[query_index]
        scale = max(
            float(normalized_bm25.max() - normalized_bm25.min()),
            1e-8,
        )
        normalized_bm25 = (
            normalized_bm25 - normalized_bm25.min()
        ) / scale
        quality_hybrid_matrix[query_index] = (
            baseline_scores["adaptive"][query_index]
            + bm25_query_weight(queries[query_index]["query"])
            * normalized_bm25
        )

    matrices = {
        "dense": baseline_scores["text"],
        "bm25": bm25_matrix,
        "image": baseline_scores["visual"],
        "adaptive_dense_image": baseline_scores["adaptive"],
        "dense_bm25_image_rrf": rrf_matrix,
        "quality_aware_rrf": quality_rrf_matrix,
        "query_aware_hybrid": quality_hybrid_matrix,
    }
    metrics = {
        name: {
            "query_count": len(queries),
            **{
                key: round(value, 6)
                for key, value in retrieval_metrics(
                    matrix,
                    expected_ids,
                    item_ids,
                ).items()
            },
        }
        for name, matrix in matrices.items()
    }
    per_query: list[dict[str, Any]] = []
    for query_index, query in enumerate(queries):
        expected_index = item_columns[query["expected_item_id"]]
        row: dict[str, Any] = {
            "query_id": query["query_id"],
            "query": query["query"],
            "query_type": query["query_type"],
            "expected_item_id": query["expected_item_id"],
        }
        for name, matrix in matrices.items():
            rank = expected_rank(matrix[query_index], expected_index)
            order = np.argsort(
                -matrix[query_index], kind="stable"
            )[:3]
            row[f"{name}_rank"] = rank
            row[f"{name}_top_3"] = [
                item_ids[index] for index in order
            ]
        per_query.append(row)

    report = {
        "status": "success",
        "settings": {
            "bm25": {"k1": 1.5, "b": 0.75},
            "rrf": {
                "k": 60,
                "top_n": len(item_ids),
                "branches": ["dense", "bm25", "image"],
            },
            "quality_aware_rrf": (
                "0.5*c*dense_rrf + 0.5*c*bm25_rrf "
                "+ (1-0.5*c)*image_rrf"
            ),
            "query_aware_hybrid": (
                "OCR-quality adaptive Dense/Image score + "
                "query-length-calibrated normalized BM25"
            ),
        },
        "metrics": metrics,
        "per_query": per_query,
    }
    output_path = project_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(output_path)


if __name__ == "__main__":
    main()
