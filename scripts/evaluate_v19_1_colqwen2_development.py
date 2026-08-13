"""Evaluate ColQwen2 on V19.1 development only."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ocr_vlm_retrieval.runtime.cache import write_json_atomic  # noqa: E402
from ocr_vlm_retrieval.runtime.late_interaction import (  # noqa: E402
    exclusive_process_lock,
    load_embedding,
    positive_retrieval_summary,
)
from scripts.evaluate_v19_colqwen2_development import (  # noqa: E402
    maxsim_score_matrix,
)
from scripts.run_v19_downstream_retrieval_pilot import read_json  # noqa: E402

EVALUATION_ROOT = ROOT / "outputs/evaluation/v19_1/condition_completeness"
DEFAULT_ASSIGNMENTS = EVALUATION_ROOT / "development_assignments_machine.json"
DEFAULT_MODEL = ROOT / "models/colqwen2-v1.0-hf"
DEFAULT_INDEX = ROOT / "outputs/user_library/colqwen2_v1_index"
DEFAULT_OUTPUT = EVALUATION_ROOT / "development_colqwen2_machine.json"
DEVELOPMENT_SPLITS = frozenset(
    {
        "v19_1_machine_draft_development_only",
        "v19_1_human_reviewed_development_only",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--query-batch-size", type=int, default=4)
    parser.add_argument("--passage-batch-size", type=int, default=16)
    parser.add_argument("--stored-top-k", type=int, default=50)
    parser.add_argument("--min-free-gib", type=float, default=5.0)
    parser.add_argument("--throttle-ms", type=int, default=250)
    return parser.parse_args()


def _query_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    if payload.get("split") not in DEVELOPMENT_SPLITS:
        raise ValueError("ColQwen2 may read V19.1 development only")
    rows = [dict(row) for row in payload.get("assignments", [])]
    if len(rows) != 48:
        raise ValueError(f"expected 48 development rows, got {len(rows)}")
    return rows


def _by_stratum(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in results:
        if bool(row.get("gold_answerable")):
            groups[str(row.get("content_stratum", "unknown"))].append(row)
    return {
        stratum: positive_retrieval_summary(rows, cutoffs=(1, 3, 10, 20))
        for stratum, rows in sorted(groups.items())
    }


def run(args: argparse.Namespace) -> int:
    if args.query_batch_size <= 0 or args.passage_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    if args.stored_top_k <= 0 or args.throttle_ms < 0:
        raise ValueError("stored-top-k must be positive and throttle nonnegative")
    assignment_payload = read_json(args.assignments)
    queries = _query_rows(assignment_payload)
    split = str(assignment_payload["split"])
    receipt = read_json(args.index_dir / "index_receipt.json")
    if receipt.get("status") != "complete" or receipt.get("failed_item_count") != 0:
        raise ValueError("a complete failure-free ColQwen2 index is required")
    item_ids = [str(item_id) for item_id in receipt.get("indexed_item_ids", [])]
    if receipt.get("scope") != "search_enabled_pages_only" or len(item_ids) != int(
        receipt["requested_item_count"]
    ):
        raise ValueError("ColQwen2 index scope or item count is invalid")
    shard_by_id = {
        path.stem: path for path in (args.index_dir / "shards").glob("*.npy")
    }
    missing = [item_id for item_id in item_ids if item_id not in shard_by_id]
    if missing:
        raise ValueError(f"index missing {len(missing)} required shards")
    shard_paths = [shard_by_id[item_id] for item_id in item_ids]

    import torch
    from transformers import ColQwen2ForRetrieval, ColQwen2Processor

    if not torch.cuda.is_available():
        raise RuntimeError("ColQwen2 evaluation requires CUDA")
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
    ).eval()  # type: ignore[no-untyped-call]
    processor = ColQwen2Processor.from_pretrained(args.model, use_fast=True)
    started = time.perf_counter()
    query_vectors: list[Any] = []
    for row in queries:
        inputs = processor(text=[str(row["query"])]).to(model.device)
        with torch.inference_mode():
            embedding = model(**inputs).embeddings[0]
        query_vectors.append(embedding.detach().to(torch.bfloat16).cpu())
        del inputs, embedding
    all_scores = torch.empty((len(queries), len(item_ids)), dtype=torch.float32)
    for passage_start in range(0, len(shard_paths), args.passage_batch_size):
        passage_slice = shard_paths[
            passage_start : passage_start + args.passage_batch_size
        ]
        passages = [
            torch.from_numpy(load_embedding(path, expected_dim=128)).to(
                device=model.device, dtype=torch.bfloat16
            )
            for path in passage_slice
        ]
        for query_start in range(0, len(query_vectors), args.query_batch_size):
            query_slice = [
                embedding.to(model.device)
                for embedding in query_vectors[
                    query_start : query_start + args.query_batch_size
                ]
            ]
            with torch.inference_mode():
                block = maxsim_score_matrix(query_slice, passages).cpu()
            all_scores[
                query_start : query_start + len(query_slice),
                passage_start : passage_start + len(passages),
            ] = block
        completed = min(passage_start + len(passages), len(item_ids))
        print(f"scored {completed}/{len(item_ids)} pages", flush=True)
        del passages
        torch.cuda.empty_cache()
        if args.throttle_ms:
            time.sleep(args.throttle_ms / 1000)

    results: list[dict[str, Any]] = []
    for query_index, row in enumerate(queries):
        order = torch.argsort(all_scores[query_index], descending=True)
        top_indices = order[: args.stored_top_k].tolist()
        ranking = [item_ids[index] for index in top_indices]
        scores = [
            round(float(all_scores[query_index, index]), 6) for index in top_indices
        ]
        relevant = {str(value) for value in row["gold_relevant_item_ids"]}
        relevant_rank = next(
            (
                rank
                for rank, index in enumerate(order.tolist(), start=1)
                if item_ids[index] in relevant
            ),
            None,
        )
        results.append(
            {
                **row,
                "ranking_item_ids": ranking,
                "scores": scores,
                "relevant_rank": relevant_rank,
            }
        )
    summary = positive_retrieval_summary(results)
    payload = {
        "status": (
            "human_reviewed_development_diagnostic_only"
            if split == "v19_1_human_reviewed_development_only"
            else "machine_draft_development_diagnostic_only"
        ),
        "method": "colqwen2_v1_multivector_late_interaction",
        "split": split,
        "eligible_for_promotion": False,
        "holdout_opened": False,
        "source_assignment_sha256": hashlib.sha256(
            args.assignments.read_bytes()
        ).hexdigest(),
        "parameters": {
            "query_batch_size": args.query_batch_size,
            "passage_batch_size": args.passage_batch_size,
            "stored_top_k": args.stored_top_k,
            "throttle_ms": args.throttle_ms,
        },
        "summary": summary,
        "by_stratum": _by_stratum(results),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "results": results,
    }
    write_json_atomic(args.output, payload)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(payload["by_stratum"], ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    args = parse_args()
    lock_path = ROOT / "outputs/user_library/.v19_1_development_gpu_job.lock"
    with exclusive_process_lock(lock_path):
        return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
