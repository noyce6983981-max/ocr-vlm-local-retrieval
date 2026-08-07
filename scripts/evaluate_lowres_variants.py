"""Measure whether low-resolution OCR variants improve the q009 BGE rank."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from FlagEmbedding import BGEM3FlagModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate low-res OCR variants.")
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("data/evaluation/retrieval_queries.csv"),
    )
    parser.add_argument(
        "--ocr-dir",
        type=Path,
        default=Path("outputs/ocr_lowres_variants/json"),
    )
    parser.add_argument(
        "--ocr-summary",
        type=Path,
        default=Path("outputs/ocr_lowres_variants/summary.csv"),
    )
    parser.add_argument(
        "--text-scores",
        type=Path,
        default=Path("outputs/evaluation/text_score_matrix.npz"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/bge-m3"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/evaluation/lowres_variant_retrieval.json"),
    )
    return parser.parse_args()


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_query(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        row = next(row for row in csv.DictReader(handle) if row["query_id"] == "q009")
    return row["query"]


def read_variant_texts(path: Path) -> tuple[list[str], list[str]]:
    ids: list[str] = []
    texts: list[str] = []
    for json_path in sorted(path.glob("*.json")):
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        ids.append(json_path.stem)
        texts.append("\n".join(payload.get("rec_texts", [])))
    return ids, texts


def main() -> None:
    args = parse_args()
    query = read_query(project_path(args.queries))
    variant_ids, variant_texts = read_variant_texts(project_path(args.ocr_dir))
    if not variant_ids:
        raise ValueError("No OCR variant results found.")

    model = BGEM3FlagModel(
        str(project_path(args.model)),
        use_fp16=True,
        devices="cuda:0",
        batch_size=4,
        passage_max_length=512,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    embeddings = model.encode(
        [query, *variant_texts],
        batch_size=4,
        max_length=512,
    )["dense_vecs"]
    similarities = embeddings[1:] @ embeddings[0]

    archive = np.load(project_path(args.text_scores))
    query_ids = archive["query_ids"].tolist()
    item_ids = archive["item_ids"].tolist()
    q009_scores = archive["scores"][query_ids.index("q009")].copy()
    target_column = item_ids.index("pilot_009_lowres_ocr_poster")

    with project_path(args.ocr_summary).open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        summary = {
            row["item_id"]: row for row in csv.DictReader(handle)
        }

    results: list[dict[str, Any]] = []
    for variant_id, text, similarity in zip(
        variant_ids, variant_texts, similarities
    ):
        replaced_scores = q009_scores.copy()
        replaced_scores[target_column] = float(similarity)
        order = np.argsort(-replaced_scores)
        rank = int(np.where(order == target_column)[0][0]) + 1
        results.append(
            {
                "variant_id": variant_id,
                "mean_ocr_confidence": float(
                    summary[variant_id]["mean_confidence"]
                ),
                "text_box_count": int(summary[variant_id]["text_box_count"]),
                "bge_similarity": round(float(similarity), 6),
                "hypothetical_text_rank": rank,
                "ocr_text": text,
            }
        )

    report = {
        "query_id": "q009",
        "query": query,
        "selection_rule": (
            "Prefer human-correct OCR text; confidence and retrieval rank "
            "are supporting evidence only."
        ),
        "results": results,
    }
    output_path = project_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for row in results:
        print(
            f"{row['variant_id']}: conf={row['mean_ocr_confidence']:.4f}, "
            f"similarity={row['bge_similarity']:.4f}, "
            f"rank={row['hypothetical_text_rank']}"
        )
    print(f"Report: {output_path}")


if __name__ == "__main__":
    main()
