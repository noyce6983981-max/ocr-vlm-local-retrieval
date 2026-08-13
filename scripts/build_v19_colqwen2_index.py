"""Build a crash-safe ColQwen2 multi-vector index for the local page library."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ocr_vlm_retrieval.runtime.cache import write_json_atomic  # noqa: E402
from ocr_vlm_retrieval.runtime.late_interaction import (  # noqa: E402
    atomic_save_embedding,
    exclusive_process_lock,
    load_embedding,
)

DEFAULT_LIBRARY = ROOT / "outputs/user_library"
DEFAULT_MODEL = ROOT / "models/colqwen2-v1.0-hf"
DEFAULT_INDEX = DEFAULT_LIBRARY / "colqwen2_v1_index"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(library_dir: Path) -> list[dict[str, Any]]:
    path = library_dir / "manifest.jsonl"
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    rows = [row for row in rows if row.get("search_enabled", True) is not False]
    item_ids = [str(row.get("item_id", "")) for row in rows]
    if not rows or any(not item_id for item_id in item_ids):
        raise ValueError("manifest must contain non-empty item IDs")
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("manifest item IDs must be unique")
    return rows


def resolve_source_path(row: dict[str, Any]) -> Path:
    path = Path(str(row["source_path"]))
    return path if path.is_absolute() else ROOT / path


def valid_existing_shard(path: Path, *, expected_dim: int = 128) -> bool:
    try:
        load_embedding(path, expected_dim=expected_dim)
    except (OSError, ValueError):
        return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library-dir", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--min-free-gib", type=float, default=5.0)
    parser.add_argument(
        "--throttle-ms",
        type=int,
        default=250,
        help="Cooling pause after each newly encoded page (default: 250 ms).",
    )
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    if args.checkpoint_every <= 0:
        raise ValueError("checkpoint-every must be positive")
    if args.throttle_ms < 0:
        raise ValueError("throttle-ms must not be negative")
    rows = read_manifest(args.library_dir)
    if args.max_items is not None:
        if args.max_items <= 0:
            raise ValueError("max-items must be positive")
        rows = rows[: args.max_items]
    manifest_path = args.library_dir / "manifest.jsonl"
    shards_dir = args.index_dir / "shards"
    receipt_path = args.index_dir / "index_receipt.json"

    import torch
    from PIL import Image
    from transformers import ColQwen2ForRetrieval, ColQwen2Processor

    if not torch.cuda.is_available():
        raise RuntimeError("ColQwen2 indexing requires CUDA")
    free_bytes, _ = torch.cuda.mem_get_info()
    free_gib = free_bytes / (1024**3)
    if free_gib < args.min_free_gib:
        raise RuntimeError(
            f"only {free_gib:.2f} GiB GPU memory is free; "
            f"at least {args.min_free_gib:.2f} GiB is required"
        )
    model = ColQwen2ForRetrieval.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
    ).eval()
    processor = ColQwen2Processor.from_pretrained(args.model, use_fast=True)
    started = time.perf_counter()
    encoded = 0
    reused = 0
    failures: list[dict[str, str]] = []

    def write_receipt(*, partial: bool) -> None:
        write_json_atomic(
            receipt_path,
            {
                "status": "partial" if partial else "complete",
                "method": "colqwen2_v1_multivector_page_index",
                "eligible_for_final_claim": False,
                "model_path": str(args.model.resolve()),
                "model_config_sha256": sha256_file(args.model / "config.json"),
                "manifest_sha256": sha256_file(manifest_path),
                "requested_item_count": len(rows),
                "scope": "search_enabled_pages_only",
                "indexed_item_ids": [str(row["item_id"]) for row in rows],
                "encoded_item_count": encoded,
                "reused_item_count": reused,
                "failed_item_count": len(failures),
                "embedding_dim": 128,
                "storage_dtype": "float16",
                "processor_use_fast": True,
                "throttle_ms": args.throttle_ms,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "failures": failures,
            },
        )

    for index, row in enumerate(rows, start=1):
        item_id = str(row["item_id"])
        shard_path = shards_dir / f"{item_id}.npy"
        if not args.force and valid_existing_shard(shard_path):
            reused += 1
        else:
            source_path = resolve_source_path(row)
            try:
                with Image.open(source_path) as source:
                    image = source.convert("RGB")
                inputs = processor(images=[image]).to(model.device)
                with torch.inference_mode():
                    embedding = model(**inputs).embeddings[0]
                atomic_save_embedding(
                    shard_path,
                    embedding.detach().to(torch.float16).cpu().numpy(),
                )
                encoded += 1
                del inputs, embedding
                if args.throttle_ms:
                    time.sleep(args.throttle_ms / 1000)
            except Exception as error:  # noqa: BLE001
                failures.append(
                    {
                        "item_id": item_id,
                        "error_type": type(error).__name__,
                        "message": str(error)[:500],
                    }
                )
            torch.cuda.empty_cache()
        if index % args.checkpoint_every == 0 or index == len(rows):
            print(
                f"[{index:04d}/{len(rows)}] encoded={encoded} reused={reused} "
                f"failed={len(failures)}",
                flush=True,
            )
            write_receipt(partial=index != len(rows))
    if failures:
        raise RuntimeError(f"indexing failed for {len(failures)} pages")
    return 0


def main() -> int:
    args = parse_args()
    lock_path = args.index_dir / ".colqwen2_gpu_job.lock"
    with exclusive_process_lock(lock_path):
        return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
