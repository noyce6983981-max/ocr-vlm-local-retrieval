"""Build a compressed dependency-free BM25 index from OCR chunks."""

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

from scripts.bm25_retrieval import build_bm25_payload  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
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
    corpus_path = project_path(args.corpus)
    output_dir = project_path(args.output)
    rows = read_jsonl(corpus_path)
    if not rows:
        raise ValueError(f"Corpus is empty: {corpus_path}")

    started = time.perf_counter()
    payload = build_bm25_payload(
        [str(row["text"]) for row in rows],
        k1=args.k1,
        b=args.b,
    )
    elapsed_seconds = round(time.perf_counter() - started, 3)
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "index.json.gz"
    metadata_path = output_dir / "metadata.jsonl"
    config_path = output_dir / "config.json"
    with gzip.open(index_path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
    with metadata_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    config = {
        "corpus_path": corpus_path.relative_to(PROJECT_ROOT).as_posix(),
        "algorithm": payload["algorithm"],
        "document_count": payload["document_count"],
        "vocabulary_size": len(payload["postings"]),
        "k1": payload["k1"],
        "b": payload["b"],
        "build_seconds": elapsed_seconds,
    }
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "success",
                **config,
                "index_path": index_path.relative_to(
                    PROJECT_ROOT
                ).as_posix(),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
