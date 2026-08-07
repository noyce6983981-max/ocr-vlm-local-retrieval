"""Rerank mixed image-and-OCR candidates with Qwen3-VL-Reranker-2B."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_REPO = PROJECT_ROOT / "third_party/Qwen3-VL-Embedding"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(OFFICIAL_REPO) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_REPO))

from src.models.qwen3_vl_reranker import Qwen3VLReranker  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/qwen3-vl-reranker-2b"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library-revision", default="base")
    parser.add_argument(
        "--exploratory",
        action="store_true",
        help="Rank broad visual relevance instead of exact evidence.",
    )
    return parser.parse_args()


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    args = parse_args()
    query = " ".join(args.query.split())
    if not query:
        raise ValueError("Query cannot be empty.")
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3-VL reranking requires CUDA.")

    candidates_path = project_path(args.candidates)
    candidates: list[dict[str, Any]] = json.loads(
        candidates_path.read_text(encoding="utf-8")
    )
    if not candidates:
        raise ValueError("Candidate list cannot be empty.")

    documents = []
    for row in candidates:
        image_path = project_path(Path(row["source_path"]))
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        document: dict[str, Any] = {"image": str(image_path)}
        evidence_text = "\n".join(
            value
            for value in (
                str(row.get("ocr_text", "")).strip()[:1200],
                str(row.get("metadata_text", "")).strip()[:500],
            )
            if value
        )
        if evidence_text:
            document["text"] = evidence_text
        documents.append(document)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = Qwen3VLReranker(
        model_name_or_path=str(project_path(args.model)),
        max_length=1536,
        min_pixels=32 * 32 * 4,
        max_pixels=384 * 384,
        dtype=torch.float16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    load_seconds = time.perf_counter() - started
    inference_started = time.perf_counter()
    exploratory = bool(args.exploratory)
    instruction = (
        "Rank candidates by broad semantic, visual, or aesthetic "
        "relevance. The query is exploratory, so useful partial matches "
        "should remain visible."
        if exploratory
        else (
            "Judge whether each document image, OCR text, and real "
            "source metadata fully match the user's query. A partial "
            "visual resemblance is not sufficient."
        )
    )
    scores = model.process(
        {
            "instruction": instruction,
            "query": {"text": query},
            "documents": documents,
        }
    )
    torch.cuda.synchronize()
    payload = {
        "query": query,
        "branch": "reranker",
        "library_revision": args.library_revision,
        "item_ids": [row["item_id"] for row in candidates],
        "scores": [round(float(score), 8) for score in scores],
        "load_seconds": round(load_seconds, 3),
        "elapsed_seconds": round(
            time.perf_counter() - inference_started, 3
        ),
        "peak_reserved_gib": round(
            torch.cuda.max_memory_reserved() / 1024**3, 3
        ),
        "exploratory_query": exploratory,
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
