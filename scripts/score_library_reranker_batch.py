"""Batch-score top candidates with Qwen3-VL-Reranker-2B."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_REPO = PROJECT_ROOT / "third_party/Qwen3-VL-Embedding"
for path in (PROJECT_ROOT, OFFICIAL_REPO):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.evaluate_library_retrieval import (  # noqa: E402
    align_archive,
    read_ocr_confidences,
    score_for_config,
)
from src.models.qwen3_vl_reranker import Qwen3VLReranker  # noqa: E402


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
        "--text-scores",
        type=Path,
        default=Path(
            "outputs/evaluation/library_retrieval/"
            "dev_v2_text_bm25_metadata_scores.npz"
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
        "--config",
        type=Path,
        default=Path(
            "outputs/evaluation/library_retrieval/"
            "selected_retrieval_config_v2_candidate.json"
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/user_library/manifest.jsonl"),
    )
    parser.add_argument(
        "--ocr-dir",
        type=Path,
        default=Path("outputs/user_library/ocr"),
    )
    parser.add_argument(
        "--metadata-index",
        type=Path,
        default=Path("outputs/user_library/metadata_index"),
    )
    parser.add_argument(
        "--ocr-summary",
        type=Path,
        default=Path("outputs/user_library/ocr/summary.csv"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/qwen3-vl-reranker-2b"),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "validation"),
        default=("train", "validation"),
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/evaluation/library_retrieval/"
            "dev_v2_reranker_scores.json"
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


def read_queries(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            row["query_id"]: row for row in csv.DictReader(handle)
        }


def ocr_text(ocr_dir: Path, item_id: str) -> str:
    for path in (
        ocr_dir / "overrides" / f"{item_id}.json",
        ocr_dir / "json" / f"{item_id}.json",
    ):
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            return "\n".join(
                str(value) for value in payload.get("rec_texts", [])
            )
    return ""


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3-VL reranking requires CUDA.")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive.")

    text_archive = np.load(project_path(args.text_scores))
    visual_archive = np.load(project_path(args.visual_scores))
    all_queries = read_queries(project_path(args.queries))
    selected_splits = set(args.splits)
    source_query_ids = text_archive["query_ids"].tolist()
    query_ids = [
        query_id
        for query_id in source_query_ids
        if all_queries[query_id]["split"] in selected_splits
    ]
    source_query_lookup = {
        query_id: index
        for index, query_id in enumerate(source_query_ids)
    }
    row_indices = [source_query_lookup[value] for value in query_ids]
    queries = [all_queries[value] for value in query_ids]
    item_ids = text_archive["item_ids"].tolist()
    dense = text_archive["dense_scores"][row_indices].astype(np.float32)
    bm25 = text_archive["bm25_scores"][row_indices].astype(np.float32)
    metadata = text_archive["metadata_scores"][row_indices].astype(
        np.float32
    )
    visual = align_archive(
        visual_archive,
        score_key="visual_scores",
        query_ids=query_ids,
        item_ids=item_ids,
    )
    confidences = read_ocr_confidences(
        project_path(args.ocr_summary), item_ids
    )
    frozen = json.loads(
        project_path(args.config).read_text(encoding="utf-8")
    )
    base_scores = score_for_config(
        frozen["ranker"],
        dense_raw=dense,
        bm25_raw=bm25,
        visual_raw=visual,
        metadata_raw=metadata,
        confidences=confidences,
        queries=queries,
    )

    manifest = {
        row["item_id"]: row
        for row in read_jsonl(project_path(args.manifest))
        if bool(row.get("search_enabled", True))
    }
    metadata_rows = {
        row["item_id"]: row["metadata_text"]
        for row in read_jsonl(
            project_path(args.metadata_index) / "metadata.jsonl"
        )
    }
    ocr_dir = project_path(args.ocr_dir)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print(
        f"Loading Qwen3-VL-Reranker-2B for {len(queries)} queries ..."
    )
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

    results: list[dict[str, Any]] = []
    inference_started = time.perf_counter()
    for position, (query, score_row) in enumerate(
        zip(queries, base_scores), start=1
    ):
        order = np.argsort(-score_row, kind="stable")[: args.top_k]
        candidate_ids = [item_ids[int(index)] for index in order]
        documents: list[dict[str, str]] = []
        for item_id in candidate_ids:
            row = manifest[item_id]
            combined_text = "\n".join(
                value
                for value in (
                    ocr_text(ocr_dir, item_id)[:1200],
                    metadata_rows.get(item_id, "")[:500],
                )
                if value
            )
            document: dict[str, str] = {
                "image": str(project_path(Path(row["source_path"])))
            }
            if combined_text:
                document["text"] = combined_text
            documents.append(document)
        scores = model.process(
            {
                "instruction": (
                    "Judge whether each document image, OCR text, and "
                    "real source metadata fully match the user's query. "
                    "A partial visual resemblance is not sufficient."
                ),
                "query": {"text": query["query"]},
                "documents": documents,
            }
        )
        values = [float(value) for value in scores]
        rerank_order = np.argsort(-np.asarray(values), kind="stable")
        results.append(
            {
                "query_id": query["query_id"],
                "split": query["split"],
                "candidate_item_ids": candidate_ids,
                "scores": [
                    round(value, 8) for value in values
                ],
                "reranked_item_ids": [
                    candidate_ids[int(index)] for index in rerank_order
                ],
                "top_score": round(
                    float(values[int(rerank_order[0])]), 8
                ),
                "top_margin": round(
                    float(
                        values[int(rerank_order[0])]
                        - values[int(rerank_order[1])]
                    )
                    if len(rerank_order) > 1
                    else float(values[int(rerank_order[0])]),
                    8,
                ),
            }
        )
        print(f"{position}/{len(queries)} {query['query_id']}")

    torch.cuda.synchronize()
    output_path = project_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "config": frozen["ranker"],
                "splits": list(args.splits),
                "query_count": len(queries),
                "top_k": args.top_k,
                "model": "Qwen3-VL-Reranker-2B",
                "load_seconds": round(load_seconds, 3),
                "inference_seconds": round(
                    time.perf_counter() - inference_started, 3
                ),
                "peak_reserved_gib": round(
                    torch.cuda.max_memory_reserved() / 1024**3, 3
                ),
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(output_path)


if __name__ == "__main__":
    main()
