"""Compact text, BM25, and visual indexes to active manifest pages only."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import faiss
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.bm25_retrieval import build_bm25_payload


DEFAULT_LIBRARY_DIR = PROJECT_ROOT / "outputs/user_library"
INDEX_NAMES = ("text_index", "bm25_index", "visual_index")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--library-dir",
        type=Path,
        default=DEFAULT_LIBRARY_DIR,
    )
    parser.add_argument(
        "--discard-backup",
        action="store_true",
        help=(
            "Delete the pre-compaction index backup after a successful swap. "
            "Use this when removed private content must not remain recoverable."
        ),
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def filter_metadata_indices(
    metadata: list[dict[str, Any]],
    active_item_ids: set[str],
) -> tuple[list[dict[str, Any]], np.ndarray]:
    keep_indices = np.array(
        [
            index
            for index, row in enumerate(metadata)
            if row["item_id"] in active_item_ids
        ],
        dtype=np.int64,
    )
    return (
        [metadata[int(index)] for index in keep_indices],
        keep_indices,
    )


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_config(path: Path, config: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_compacted_indexes(
    library_dir: Path,
    stage_root: Path,
    *,
    compacted_at: str,
) -> dict[str, Any]:
    manifest = read_jsonl(library_dir / "manifest.jsonl")
    active_rows = [
        row
        for row in manifest
        if bool(row.get("search_enabled", True))
    ]
    active_item_ids = {row["item_id"] for row in active_rows}
    if len(active_item_ids) != len(active_rows):
        raise ValueError("Active manifest contains duplicate item IDs.")

    for name in INDEX_NAMES:
        (stage_root / name).mkdir(parents=True, exist_ok=False)

    visual_dir = library_dir / "visual_index"
    visual_metadata = read_jsonl(visual_dir / "metadata.jsonl")
    visual_embeddings = np.load(visual_dir / "embeddings.npy")
    if len(visual_metadata) != len(visual_embeddings):
        raise ValueError("Visual index is inconsistent.")
    filtered_visual_metadata, visual_indices = filter_metadata_indices(
        visual_metadata,
        active_item_ids,
    )
    filtered_visual_ids = {
        row["item_id"] for row in filtered_visual_metadata
    }
    missing_visual_ids = sorted(active_item_ids - filtered_visual_ids)
    if missing_visual_ids:
        raise ValueError(
            "Active pages are missing visual embeddings: "
            + ", ".join(missing_visual_ids[:10])
        )
    filtered_visual_embeddings = np.ascontiguousarray(
        visual_embeddings[visual_indices],
        dtype=np.float32,
    )
    visual_stage = stage_root / "visual_index"
    np.save(
        visual_stage / "embeddings.npy",
        filtered_visual_embeddings,
    )
    write_jsonl(
        visual_stage / "metadata.jsonl",
        filtered_visual_metadata,
    )
    visual_config = load_config(visual_dir / "config.json")
    visual_config.update(
        {
            "image_count": len(filtered_visual_metadata),
            "dtype": str(filtered_visual_embeddings.dtype),
            "compacted_at": compacted_at,
            "active_manifest_count": len(active_rows),
            "filtered_inactive_count": (
                len(visual_metadata) - len(filtered_visual_metadata)
            ),
        }
    )
    write_config(visual_stage / "config.json", visual_config)

    text_dir = library_dir / "text_index"
    text_metadata = read_jsonl(text_dir / "metadata.jsonl")
    text_index_bytes = np.frombuffer(
        (text_dir / "index.faiss").read_bytes(),
        dtype=np.uint8,
    )
    text_index = faiss.deserialize_index(text_index_bytes)
    if text_index.ntotal != len(text_metadata):
        raise ValueError("Text index is inconsistent.")
    filtered_text_metadata, text_indices = filter_metadata_indices(
        text_metadata,
        active_item_ids,
    )
    text_vectors = text_index.reconstruct_n(
        0, text_index.ntotal
    ).astype(np.float32, copy=False)
    filtered_text_vectors = np.ascontiguousarray(
        text_vectors[text_indices],
        dtype=np.float32,
    )
    compacted_text_index = faiss.IndexFlatIP(text_index.d)
    compacted_text_index.add(filtered_text_vectors)
    text_stage = stage_root / "text_index"
    (text_stage / "index.faiss").write_bytes(
        faiss.serialize_index(compacted_text_index).tobytes()
    )
    write_jsonl(
        text_stage / "metadata.jsonl",
        filtered_text_metadata,
    )
    text_config = load_config(text_dir / "config.json")
    text_config.update(
        {
            "vector_count": len(filtered_text_metadata),
            "compacted_at": compacted_at,
            "active_manifest_count": len(active_rows),
            "filtered_inactive_chunk_count": (
                len(text_metadata) - len(filtered_text_metadata)
            ),
        }
    )
    write_config(text_stage / "config.json", text_config)

    bm25_dir = library_dir / "bm25_index"
    bm25_metadata = read_jsonl(bm25_dir / "metadata.jsonl")
    filtered_bm25_metadata, _ = filter_metadata_indices(
        bm25_metadata,
        active_item_ids,
    )
    bm25_config = load_config(bm25_dir / "config.json")
    bm25_payload = build_bm25_payload(
        [str(row["text"]) for row in filtered_bm25_metadata],
        k1=float(bm25_config.get("k1", 1.5)),
        b=float(bm25_config.get("b", 0.75)),
    )
    bm25_stage = stage_root / "bm25_index"
    with gzip.open(
        bm25_stage / "index.json.gz",
        "wt",
        encoding="utf-8",
    ) as handle:
        json.dump(
            bm25_payload,
            handle,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    write_jsonl(
        bm25_stage / "metadata.jsonl",
        filtered_bm25_metadata,
    )
    bm25_config.update(
        {
            "document_count": len(filtered_bm25_metadata),
            "vocabulary_size": len(bm25_payload["postings"]),
            "compacted_at": compacted_at,
            "active_manifest_count": len(active_rows),
            "filtered_inactive_chunk_count": (
                len(bm25_metadata) - len(filtered_bm25_metadata)
            ),
        }
    )
    write_config(bm25_stage / "config.json", bm25_config)

    return {
        "total_manifest_pages": len(manifest),
        "active_manifest_pages": len(active_rows),
        "inactive_manifest_pages": len(manifest) - len(active_rows),
        "visual_vectors_before": len(visual_metadata),
        "visual_vectors_after": len(filtered_visual_metadata),
        "text_chunks_before": len(text_metadata),
        "text_chunks_after": len(filtered_text_metadata),
        "bm25_documents_before": len(bm25_metadata),
        "bm25_documents_after": len(filtered_bm25_metadata),
        "bm25_vocabulary_after": len(bm25_payload["postings"]),
    }


def publish_compacted_indexes(
    library_dir: Path,
    *,
    now: datetime | None = None,
    keep_backup: bool = True,
) -> dict[str, Any]:
    library_dir = library_dir.resolve()
    if not (library_dir / "manifest.jsonl").is_file():
        raise FileNotFoundError(library_dir / "manifest.jsonl")
    for name in INDEX_NAMES:
        if not (library_dir / name).is_dir():
            raise FileNotFoundError(library_dir / name)

    timestamp = now or datetime.now(timezone.utc)
    compacted_at = timestamp.isoformat(timespec="seconds")
    suffix = timestamp.strftime("%Y%m%d_%H%M%S_%f")
    stage_root = library_dir / f".index_compaction_{suffix}"
    backup_root = (
        library_dir / "index_backups" / f"before_compaction_{suffix}"
    )
    stage_root.mkdir(parents=False, exist_ok=False)
    report = build_compacted_indexes(
        library_dir,
        stage_root,
        compacted_at=compacted_at,
    )
    backup_root.mkdir(parents=True, exist_ok=False)

    swapped: list[str] = []
    try:
        for name in INDEX_NAMES:
            current = library_dir / name
            backup = backup_root / name
            staged = stage_root / name
            os.replace(current, backup)
            os.replace(staged, current)
            swapped.append(name)
    except Exception:
        for name in reversed(swapped):
            current = library_dir / name
            backup = backup_root / name
            failed_stage = stage_root / name
            if current.exists():
                os.replace(current, failed_stage)
            if backup.exists():
                os.replace(backup, current)
        raise
    finally:
        if stage_root.exists() and not any(stage_root.iterdir()):
            stage_root.rmdir()

    if not keep_backup:
        shutil.rmtree(backup_root)

    report.update(
        {
            "status": "success",
            "compacted_at": compacted_at,
            "backup_path": str(backup_root) if keep_backup else None,
            "backup_discarded": not keep_backup,
        }
    )
    (library_dir / "index_compaction_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    args = parse_args()
    library_dir = (
        args.library_dir
        if args.library_dir.is_absolute()
        else PROJECT_ROOT / args.library_dir
    )
    report = publish_compacted_indexes(
        library_dir,
        keep_backup=not args.discard_backup,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
