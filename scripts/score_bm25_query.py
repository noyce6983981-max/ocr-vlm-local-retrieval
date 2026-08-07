"""Score one arbitrary query against a compressed OCR BM25 index."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.bm25_retrieval import score_bm25  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library-revision", default="base")
    return parser.parse_args()


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    args = parse_args()
    query = " ".join(args.query.split())
    if not query:
        raise ValueError("Query cannot be empty.")
    manifest = read_jsonl(project_path(args.manifest))
    item_ids = [row["item_id"] for row in manifest]
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("Manifest contains duplicate item IDs.")
    item_columns = {item_id: index for index, item_id in enumerate(item_ids)}
    index_dir = project_path(args.index_dir)
    index_path = index_dir / "index.json.gz"
    metadata_path = index_dir / "metadata.jsonl"
    if not index_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Incomplete BM25 index: {index_dir}")

    started = time.perf_counter()
    with gzip.open(index_path, "rt", encoding="utf-8") as handle:
        index_payload = json.load(handle)
    metadata = read_jsonl(metadata_path)
    if len(metadata) != int(index_payload["document_count"]):
        raise ValueError("BM25 index and metadata are inconsistent.")
    chunk_scores = score_bm25(query, index_payload)
    document_scores = [0.0] * len(item_ids)
    for row, score in zip(metadata, chunk_scores):
        item_id = row["item_id"]
        if item_id not in item_columns:
            raise ValueError(
                f"BM25 index item is absent from manifest: {item_id}"
            )
        column = item_columns[item_id]
        document_scores[column] = max(document_scores[column], score)

    payload = {
        "query": query,
        "branch": "bm25",
        "library_revision": args.library_revision,
        "item_ids": item_ids,
        "scores": [round(float(score), 8) for score in document_scores],
        "matched_documents": sum(score > 0 for score in document_scores),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    output_path = project_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(output_path)


if __name__ == "__main__":
    main()
