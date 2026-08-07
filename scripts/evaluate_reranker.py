"""Evaluate Qwen3-VL-Reranker on adaptive-fusion Top-K candidates."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_REPO = PROJECT_ROOT / "third_party/Qwen3-VL-Embedding"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(OFFICIAL_REPO) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_REPO))

from scripts.bm25_retrieval import (  # noqa: E402
    bm25_query_weight,
    build_bm25_payload,
    score_bm25,
)
from src.models.qwen3_vl_reranker import Qwen3VLReranker  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("data/evaluation/dataset_v1_queries.csv"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifest/dataset_v1_manifest.jsonl"),
    )
    parser.add_argument(
        "--ocr-summary",
        type=Path,
        default=Path("outputs/ocr_dataset_v1/summary.csv"),
    )
    parser.add_argument(
        "--ocr-json-dir",
        type=Path,
        default=Path("outputs/ocr_dataset_v1/json"),
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
        "--corpus",
        type=Path,
        default=Path("data/processed/dataset_v1_corpus.jsonl"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/qwen3-vl-reranker-2b"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/evaluation_dataset_v1/reranker_retrieval.json"
        ),
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--candidate-source",
        choices=("adaptive", "query_hybrid"),
        default="adaptive",
    )
    return parser.parse_args()


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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
    return archive["scores"][np.ix_(query_order, item_order)].astype(
        np.float32
    )


def ocr_text(directory: Path, item_id: str) -> str:
    path = directory / f"{item_id}.json"
    if not path.is_file():
        return ""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(
        str(value) for value in payload.get("rec_texts", [])
    )[:1500]


def metrics(ranks: list[int]) -> dict[str, float | int]:
    return {
        "query_count": len(ranks),
        "recall_at_1": round(
            sum(rank == 1 for rank in ranks) / len(ranks), 6
        ),
        "recall_at_3": round(
            sum(rank <= 3 for rank in ranks) / len(ranks), 6
        ),
        "mrr": round(
            statistics.fmean(1.0 / rank for rank in ranks), 6
        ),
    }


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive.")
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3-VL reranking requires CUDA.")

    queries = read_csv(project_path(args.queries))
    manifest = read_jsonl(project_path(args.manifest))
    ocr_rows = read_csv(project_path(args.ocr_summary))
    query_ids = [row["query_id"] for row in queries]
    item_ids = [row["item_id"] for row in manifest]
    if item_ids != [row["item_id"] for row in ocr_rows]:
        raise ValueError("Manifest and OCR summary item orders differ.")

    with np.load(project_path(args.text_scores)) as archive:
        text = normalize_rows(align_matrix(archive, query_ids, item_ids))
    with np.load(project_path(args.visual_scores)) as archive:
        visual = normalize_rows(align_matrix(archive, query_ids, item_ids))
    confidences = np.array(
        [float(row["mean_confidence"]) for row in ocr_rows],
        dtype=np.float32,
    )
    text_weights = np.clip(0.6 * confidences, 0.0, 1.0)
    adaptive = (
        text * text_weights[None, :]
        + visual * (1.0 - text_weights[None, :])
    )
    candidate_scores = adaptive
    if args.candidate_source == "query_hybrid":
        corpus = read_jsonl(project_path(args.corpus))
        bm25_index = build_bm25_payload(
            [str(row["text"]) for row in corpus]
        )
        bm25 = np.zeros_like(adaptive)
        item_columns = {
            item_id: index for index, item_id in enumerate(item_ids)
        }
        for query_index, query in enumerate(queries):
            chunk_scores = score_bm25(query["query"], bm25_index)
            for chunk, score in zip(corpus, chunk_scores):
                column = item_columns[chunk["item_id"]]
                bm25[query_index, column] = max(
                    bm25[query_index, column], float(score)
                )
        bm25 = normalize_rows(bm25)
        candidate_scores = adaptive + np.array(
            [
                bm25_query_weight(query["query"])
                for query in queries
            ],
            dtype=np.float32,
        )[:, None] * bm25

    manifest_by_id = {row["item_id"]: row for row in manifest}
    ocr_directory = project_path(args.ocr_json_dir)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    model = Qwen3VLReranker(
        model_name_or_path=str(project_path(args.model)),
        max_length=1536,
        min_pixels=32 * 32 * 4,
        max_pixels=384 * 384,
        dtype=torch.float16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    load_seconds = time.perf_counter() - load_started

    inference_started = time.perf_counter()
    results: list[dict[str, Any]] = []
    before_ranks: list[int] = []
    after_ranks: list[int] = []
    top_k = min(args.top_k, len(item_ids))
    instruction = (
        "Judge whether each personal document page or image matches the "
        "user's Chinese natural-language query."
    )

    for query_index, query in enumerate(queries):
        base_order = np.argsort(
            -candidate_scores[query_index]
        ).tolist()
        candidate_indices = base_order[:top_k]
        documents = []
        for item_index in candidate_indices:
            item_id = item_ids[item_index]
            row = manifest_by_id[item_id]
            document: dict[str, Any] = {
                "image": str(project_path(Path(row["source_path"])))
            }
            text_value = ocr_text(ocr_directory, item_id)
            if text_value:
                document["text"] = text_value
            documents.append(document)

        scores = model.process(
            {
                "instruction": instruction,
                "query": {"text": query["query"]},
                "documents": documents,
            }
        )
        candidate_order = np.argsort(-np.asarray(scores)).tolist()
        reranked_indices = [
            candidate_indices[index] for index in candidate_order
        ]
        candidate_set = set(candidate_indices)
        reranked_indices.extend(
            index for index in base_order if index not in candidate_set
        )
        expected_index = item_ids.index(query["expected_item_id"])
        before_rank = base_order.index(expected_index) + 1
        after_rank = reranked_indices.index(expected_index) + 1
        before_ranks.append(before_rank)
        after_ranks.append(after_rank)
        results.append(
            {
                "query_id": query["query_id"],
                "query": query["query"],
                "expected_item_id": query["expected_item_id"],
                "candidate_rank": before_rank,
                "reranker_rank": after_rank,
                "candidate_top_1_item_id": item_ids[base_order[0]],
                "reranker_top_1_item_id": item_ids[reranked_indices[0]],
                "reranked_top_k_item_ids": [
                    item_ids[index] for index in reranked_indices[:top_k]
                ],
                "reranked_top_k_scores": [
                    round(float(scores[index]), 8)
                    for index in candidate_order
                ],
            }
        )
        print(
            f"{query['query_id']}: candidate={before_rank}, "
            f"reranker={after_rank}"
        )

    torch.cuda.synchronize()
    payload = {
        "settings": {
            "model": "Qwen3-VL-Reranker-2B",
            "candidate_source": args.candidate_source,
            "top_k": top_k,
            "max_pixels": 384 * 384,
            "max_length": 1536,
        },
        "candidate_baseline": metrics(before_ranks),
        "reranker": metrics(after_ranks),
        "timings": {
            "load_seconds": round(load_seconds, 3),
            "inference_seconds": round(
                time.perf_counter() - inference_started, 3
            ),
        },
        "peak_reserved_gib": round(
            torch.cuda.max_memory_reserved() / 1024**3, 3
        ),
        "results": results,
    }
    output_path = project_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload["reranker"], ensure_ascii=False))
    print(output_path)


if __name__ == "__main__":
    main()
