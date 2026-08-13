"""Evaluate ColQwen2 retrieval on V19 reviewed development queries only."""

from __future__ import annotations

import argparse
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
    union_recall_summary,
)
from scripts.run_v19_downstream_retrieval_pilot import (  # noqa: E402
    ranking_ids,
)

DEFAULT_ASSIGNMENTS = (
    ROOT
    / "outputs/evaluation/v19/selective_intervention"
    / "development_route_assignments.json"
)
DEFAULT_RETRIEVAL_DIR = (
    ROOT / "outputs/evaluation/v19/selective_intervention/retrieval/baseline"
)
DEFAULT_MODEL = ROOT / "models/colqwen2-v1.0-hf"
DEFAULT_INDEX = ROOT / "outputs/user_library/colqwen2_v1_index"
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/evaluation/v19/selective_intervention"
    / "development_colqwen2_v1.json"
)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def build_query_rows(
    assignments_payload: Mapping[str, Any],
    *,
    retrieval_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if assignments_payload.get("split") != "v19_reviewed_development_only":
        raise ValueError("ColQwen2 development evaluation may not read the holdout")
    rows: list[dict[str, Any]] = []
    baseline: list[dict[str, Any]] = []
    for assignment in assignments_payload.get("assignments", []):
        query_id = str(assignment["query_id"])
        relevant = [
            str(item_id)
            for item_id in assignment.get("gold_relevant_item_ids", [])
            if str(item_id).strip()
        ]
        if bool(assignment.get("gold_answerable")) and not relevant:
            relevant = [str(assignment["source_item_id"])]
        common = {
            "query_id": query_id,
            "query": str(assignment["query"]),
            "query_role": str(assignment.get("query_role", "")),
            "content_stratum": str(assignment.get("content_stratum", "")),
            "gold_answerable": bool(assignment.get("gold_answerable")),
            "gold_relevant_item_ids": relevant,
        }
        rows.append(common)
        retrieval = read_json(retrieval_dir / f"{query_id}_v18_frozen.json")
        if retrieval.get("query") != assignment.get("query"):
            raise ValueError(f"retrieval query mismatch for {query_id}")
        baseline.append(
            {**common, "ranking_item_ids": ranking_ids(retrieval)}
        )
    if not rows:
        raise ValueError("no V19 development queries")
    return rows, baseline


def maxsim_score_matrix(
    query_embeddings: Sequence[Any],
    passage_embeddings: Sequence[Any],
) -> Any:
    """Return masked ColBERT MaxSim scores for one GPU-resident block."""

    import torch

    queries = torch.nn.utils.rnn.pad_sequence(
        list(query_embeddings), batch_first=True, padding_value=0
    )
    passages = torch.nn.utils.rnn.pad_sequence(
        list(passage_embeddings), batch_first=True, padding_value=0
    )
    passage_lengths = torch.tensor(
        [len(embedding) for embedding in passage_embeddings],
        device=passages.device,
    )
    passage_positions = torch.arange(passages.shape[1], device=passages.device)
    valid_passages = passage_positions[None, :] < passage_lengths[:, None]
    similarities = torch.einsum("bnd,csd->bcns", queries, passages)
    similarities = similarities.masked_fill(
        ~valid_passages[None, :, None, :],
        torch.finfo(similarities.dtype).min,
    )
    return similarities.amax(dim=3).sum(dim=2).float()


def summarize_by_stratum(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in results:
        if bool(row.get("gold_answerable")):
            groups[str(row.get("content_stratum", "unknown"))].append(row)
    return {
        stratum: positive_retrieval_summary(rows, cutoffs=(1, 3, 10, 20))
        for stratum, rows in sorted(groups.items())
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--retrieval-dir", type=Path, default=DEFAULT_RETRIEVAL_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--query-batch-size", type=int, default=4)
    parser.add_argument("--passage-batch-size", type=int, default=16)
    parser.add_argument("--stored-top-k", type=int, default=50)
    parser.add_argument("--min-free-gib", type=float, default=5.0)
    parser.add_argument("--throttle-ms", type=int, default=250)
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    if args.query_batch_size <= 0 or args.passage_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    if args.stored_top_k <= 0:
        raise ValueError("stored-top-k must be positive")
    if args.throttle_ms < 0:
        raise ValueError("throttle-ms must not be negative")
    receipt = read_json(args.index_dir / "index_receipt.json")
    if receipt.get("status") != "complete" or receipt.get("failed_item_count") != 0:
        raise ValueError("a complete failure-free ColQwen2 index is required")
    queries, baseline = build_query_rows(
        read_json(args.assignments), retrieval_dir=args.retrieval_dir
    )
    shard_paths = sorted((args.index_dir / "shards").glob("*.npy"))
    expected_count = int(receipt["requested_item_count"])
    if receipt.get("scope") != "search_enabled_pages_only":
        raise ValueError("index must contain search-enabled pages only")
    indexed_item_ids = [
        str(item_id) for item_id in receipt.get("indexed_item_ids", [])
    ]
    if len(indexed_item_ids) != expected_count:
        raise ValueError("index receipt item IDs do not match requested count")
    shard_by_id = {path.stem: path for path in shard_paths}
    missing = [item_id for item_id in indexed_item_ids if item_id not in shard_by_id]
    if missing:
        raise ValueError(f"index missing {len(missing)} required shards")
    shard_paths = [shard_by_id[item_id] for item_id in indexed_item_ids]

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
    ).eval()
    processor = ColQwen2Processor.from_pretrained(args.model, use_fast=True)
    started = time.perf_counter()
    query_vectors: list[Any] = []
    for row in queries:
        inputs = processor(text=[row["query"]]).to(model.device)
        with torch.inference_mode():
            embedding = model(**inputs).embeddings[0]
        query_vectors.append(embedding.detach().to(torch.bfloat16).cpu())
        del inputs, embedding
    item_ids = [path.stem for path in shard_paths]
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
        print(
            f"scored {min(passage_start + len(passages), len(item_ids))}"
            f"/{len(item_ids)} pages",
            flush=True,
        )
        del passages
        torch.cuda.empty_cache()
        if args.throttle_ms:
            time.sleep(args.throttle_ms / 1000)

    results: list[dict[str, Any]] = []
    for query_index, row in enumerate(queries):
        order = torch.argsort(all_scores[query_index], descending=True)
        top_indices = order[: args.stored_top_k].tolist()
        ranking = [item_ids[index] for index in top_indices]
        scores = [round(float(all_scores[query_index, index]), 6) for index in top_indices]
        relevant = set(row["gold_relevant_item_ids"])
        relevant_rank = next(
            (
                rank
                for rank, item_id in enumerate(
                    (item_ids[index] for index in order.tolist()), start=1
                )
                if item_id in relevant
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
    baseline_summary = positive_retrieval_summary(baseline)
    union = {
        f"union_at_{cutoff}": union_recall_summary(
            baseline, results, cutoff=cutoff
        )["union_recall"]
        for cutoff in (1, 3, 10, 20)
    }
    write_json_atomic(
        args.output,
        {
            "status": "development_diagnostic_only",
            "method": "colqwen2_v1_multivector_late_interaction",
            "split": "development_only",
            "eligible_for_final_claim": False,
            "metric_warning": (
                "Closed-set positive retrieval recall is not open-set end-to-end "
                "accuracy and is not directly comparable to ViDoRe nDCG."
            ),
            "parameters": {
                "query_batch_size": args.query_batch_size,
                "passage_batch_size": args.passage_batch_size,
                "stored_top_k": args.stored_top_k,
                "processor_use_fast": True,
                "throttle_ms": args.throttle_ms,
            },
            "summary": summary,
            "baseline_quality_hybrid_summary": baseline_summary,
            "baseline_colqwen2_union": union,
            "by_stratum": summarize_by_stratum(results),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "results": results,
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({"baseline_colqwen2_union": union}, indent=2))
    return 0


def main() -> int:
    args = parse_args()
    lock_path = args.index_dir / ".colqwen2_gpu_job.lock"
    with exclusive_process_lock(lock_path):
        return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
