"""Export a relevance-review bundle without ranker or author signals."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.blind_query_study import (  # noqa: E402
    load_protocol,
    read_candidates,
    verify_frozen_snapshot,
)


SAFE_QUERY_FIELDS = {"query_id", "query", "candidates"}
SAFE_CANDIDATE_FIELDS = {"item_id", "source_path", "ocr_text"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study-dir",
        type=Path,
        default=Path("data/evaluation/blind_study_v1"),
    )
    parser.add_argument(
        "--library-dir", type=Path, default=Path("outputs/user_library")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/evaluation/blind_study_v1/model_review_bundle.jsonl"
        ),
    )
    return parser.parse_args()


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_manifest(path: Path) -> dict[str, dict[str, Any]]:
    return {
        row["item_id"]: row
        for row in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def load_ocr_text(ocr_dir: Path, item_id: str) -> str:
    path = ocr_dir / f"{item_id}.json"
    if not path.is_file():
        return ""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    texts = [" ".join(str(value).split()) for value in payload.get("rec_texts", [])]
    return "\n".join(value for value in texts if value)[:6000]


def safe_review_row(
    query_row: dict[str, Any],
    candidate: dict[str, Any],
    manifest_by_id: dict[str, dict[str, Any]],
    ocr_loader: Callable[[str], str],
) -> dict[str, Any]:
    safe_candidates: list[dict[str, str]] = []
    for item_id in candidate["candidate_item_ids"]:
        item = manifest_by_id.get(item_id)
        if item is None:
            raise ValueError(f"Unknown candidate item: {item_id}")
        safe_candidates.append(
            {
                "item_id": item_id,
                "source_path": str(item["source_path"]),
                "ocr_text": ocr_loader(item_id),
            }
        )
    return {
        "query_id": str(query_row["query_id"]),
        "query": str(query_row["query"]),
        "candidates": safe_candidates,
    }


def write_bundle(
    study_dir: Path,
    library_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    protocol = load_protocol(study_dir)
    frozen = verify_frozen_snapshot(study_dir)
    candidates = read_candidates(study_dir)
    if set(candidates) != {row["query_id"] for row in frozen}:
        raise ValueError("Candidate pool is incomplete; refusing model-review export.")
    manifest_by_id = read_manifest(library_dir / "manifest.jsonl")
    ocr_dir = library_dir / "ocr/json"
    rows = [
        safe_review_row(
            row,
            candidates[row["query_id"]],
            manifest_by_id,
            lambda item_id: load_ocr_text(ocr_dir, item_id),
        )
        for row in frozen
    ]
    content = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in rows
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, output_path)
    bundle_sha256 = hashlib.sha256(output_path.read_bytes()).hexdigest()
    metadata = {
        "schema_version": 1,
        "query_count": len(rows),
        "candidate_count_per_query": sorted(
            {len(row["candidates"]) for row in rows}
        ),
        "query_set_sha256": protocol["query_set_sha256"],
        "bundle_sha256": bundle_sha256,
        "allowed_query_fields": sorted(SAFE_QUERY_FIELDS),
        "allowed_candidate_fields": sorted(SAFE_CANDIDATE_FIELDS),
        "excluded_signals": [
            "author answerability expectation",
            "system acceptance decision",
            "retrieval route",
            "scores",
            "ranks",
            "pool sources",
        ],
    }
    metadata_path = output_path.with_suffix(".protocol.json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    args = parse_args()
    metadata = write_bundle(
        project_path(args.study_dir),
        project_path(args.library_dir),
        project_path(args.output),
    )
    print(json.dumps(metadata, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
