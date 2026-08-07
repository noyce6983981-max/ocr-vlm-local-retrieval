"""Convert PaddleOCR JSON results into chunked text-retrieval documents."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a chunked JSONL corpus from OCR batch results."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifest/pilot_manifest.jsonl"),
    )
    parser.add_argument(
        "--ocr-dir",
        type=Path,
        default=Path("outputs/ocr_batch/json"),
    )
    parser.add_argument(
        "--ocr-overrides-dir",
        type=Path,
        help=(
            "Optional directory of per-item OCR JSON files used only for "
            "retrieval; baseline OCR remains unchanged."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/pilot_corpus.jsonl"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("outputs/text_corpus/summary.json"),
    )
    parser.add_argument(
        "--chunk-chars",
        type=int,
        default=400,
        help="Approximate maximum OCR characters per chunk.",
    )
    parser.add_argument(
        "--overlap-boxes",
        type=int,
        default=2,
        help="Number of OCR boxes shared by adjacent chunks.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Discard OCR boxes below this recognition confidence.",
    )
    return parser.parse_args()


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return rows


def resolve_ocr_path(
    ocr_dir: Path,
    item_id: str,
    overrides_dir: Path | None = None,
) -> tuple[Path, bool]:
    if overrides_dir is not None:
        override_path = overrides_dir / f"{item_id}.json"
        if override_path.is_file():
            return override_path, True
    return ocr_dir / f"{item_id}.json", False


def make_chunks(
    boxes: list[dict[str, Any]], chunk_chars: int, overlap_boxes: int
) -> list[list[dict[str, Any]]]:
    chunks: list[list[dict[str, Any]]] = []
    cursor = 0

    while cursor < len(boxes):
        start = cursor
        chunk: list[dict[str, Any]] = []
        character_count = 0

        while cursor < len(boxes):
            text_length = len(boxes[cursor]["text"])
            if chunk and character_count + text_length > chunk_chars:
                break
            chunk.append(boxes[cursor])
            character_count += text_length
            cursor += 1

        chunks.append(chunk)
        if cursor >= len(boxes):
            break
        cursor = max(start + 1, cursor - overlap_boxes)

    return chunks


def main() -> None:
    args = parse_args()
    if args.chunk_chars <= 0:
        raise ValueError("--chunk-chars must be positive.")
    if args.overlap_boxes < 0:
        raise ValueError("--overlap-boxes cannot be negative.")
    if not 0.0 <= args.min_confidence <= 1.0:
        raise ValueError("--min-confidence must be between 0 and 1.")

    manifest_path = project_path(args.manifest)
    ocr_dir = project_path(args.ocr_dir)
    overrides_dir = (
        project_path(args.ocr_overrides_dir)
        if args.ocr_overrides_dir is not None
        else None
    )
    output_path = project_path(args.output)
    summary_path = project_path(args.summary)
    manifest = read_jsonl(manifest_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    corpus_rows: list[dict[str, Any]] = []
    documents_without_text: list[str] = []
    override_item_ids: list[str] = []

    for item in manifest:
        item_id = item["item_id"]
        ocr_path, used_override = resolve_ocr_path(
            ocr_dir, item_id, overrides_dir
        )
        if not ocr_path.is_file():
            raise FileNotFoundError(f"Missing OCR result: {ocr_path}")
        if used_override:
            override_item_ids.append(item_id)

        result = json.loads(ocr_path.read_text(encoding="utf-8"))
        texts = result.get("rec_texts", [])
        scores = result.get("rec_scores", [])
        if len(texts) != len(scores):
            raise ValueError(
                f"OCR text/score length mismatch for {item_id}: "
                f"{len(texts)} vs {len(scores)}"
            )

        boxes = [
            {"box_index": index, "text": str(text).strip(), "confidence": float(score)}
            for index, (text, score) in enumerate(zip(texts, scores))
            if str(text).strip() and float(score) >= args.min_confidence
        ]
        if not boxes:
            documents_without_text.append(item_id)
            continue

        for chunk_index, chunk in enumerate(
            make_chunks(boxes, args.chunk_chars, args.overlap_boxes)
        ):
            chunk_scores = [box["confidence"] for box in chunk]
            corpus_rows.append(
                {
                    "chunk_id": f"{item_id}_chunk_{chunk_index:03d}",
                    "item_id": item_id,
                    "display_name_zh": item.get("display_name_zh", ""),
                    "category": item.get("category", ""),
                    "source_path": item["source_path"],
                    "box_start": chunk[0]["box_index"],
                    "box_end": chunk[-1]["box_index"],
                    "mean_ocr_confidence": round(
                        statistics.fmean(chunk_scores), 6
                    ),
                    "text": "\n".join(box["text"] for box in chunk),
                }
            )

    with output_path.open("w", encoding="utf-8") as handle:
        for row in corpus_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "manifest_path": manifest_path.relative_to(PROJECT_ROOT).as_posix(),
        "ocr_dir": ocr_dir.relative_to(PROJECT_ROOT).as_posix(),
        "output_path": output_path.relative_to(PROJECT_ROOT).as_posix(),
        "document_count": len(manifest),
        "documents_with_text": len(manifest) - len(documents_without_text),
        "documents_without_text": documents_without_text,
        "ocr_override_item_ids": override_item_ids,
        "ocr_override_count": len(override_item_ids),
        "chunk_count": len(corpus_rows),
        "total_chunk_characters": sum(len(row["text"]) for row in corpus_rows),
        "settings": {
            "chunk_chars": args.chunk_chars,
            "overlap_boxes": args.overlap_boxes,
            "min_confidence": args.min_confidence,
        },
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        f"Corpus built: {len(corpus_rows)} chunks from "
        f"{summary['documents_with_text']}/{len(manifest)} text-bearing documents."
    )
    print(f"Corpus: {output_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
