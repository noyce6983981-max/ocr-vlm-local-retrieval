"""Evaluate full-query and condition-level ColQwen2 rank aggregation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ocr_vlm_retrieval.gating.ocr_literals_v19_2 import (  # noqa: E402
    extract_v19_2_literal_groups,
)
from ocr_vlm_retrieval.runtime.cache import write_json_atomic  # noqa: E402
from ocr_vlm_retrieval.runtime.condition_aggregation import (  # noqa: E402
    aggregate_rank_percentiles,
    ranked_indices,
)
from ocr_vlm_retrieval.runtime.late_interaction import (  # noqa: E402
    exclusive_process_lock,
    load_embedding,
    positive_retrieval_summary,
)
from scripts.evaluate_v19_colqwen2_development import (  # noqa: E402
    maxsim_score_matrix,
)
from scripts.run_v19_downstream_retrieval_pilot import read_json  # noqa: E402

EVALUATION_ROOT = ROOT / "outputs/evaluation/v19_2/automatic_optimization"
DEFAULT_ASSIGNMENTS = EVALUATION_ROOT / "development_assignments_machine.json"
DEFAULT_CONFIG = ROOT / "config/studies/v19_2_automatic_optimization.json"
DEFAULT_MODEL = ROOT / "models/colqwen2-v1.0-hf"
DEFAULT_INDEX = ROOT / "outputs/user_library/colqwen2_v1_index"
DEFAULT_OUTPUT = EVALUATION_ROOT / "development_condition_aggregation.json"
SPLIT = "v19_2_automatic_development_only"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--locked-method", choices=("", *(
        "full_query",
        "condition_min_rank",
        "condition_mean_rank",
        "full_25_condition_min_75",
        "full_25_condition_mean_75",
    )), default="")
    parser.add_argument("--stored-top-k", type=int, default=50)
    parser.add_argument("--query-batch-size", type=int, default=8)
    parser.add_argument("--passage-batch-size", type=int, default=16)
    parser.add_argument("--min-free-gib", type=float, default=5.0)
    return parser.parse_args()


def explicit_phrases(query: str) -> list[str]:
    groups = extract_v19_2_literal_groups(query)
    values = [group.label for group in groups if group.source == "explicit_required"]
    if len(values) != 2:
        raise ValueError("condition aggregation requires exactly two explicit phrases")
    return values


def _selection_key(
    method: str,
    summary: Mapping[str, Any],
    configured: Sequence[str],
) -> tuple[float, float, float, int]:
    return (
        -float(summary["recall_at_3"]),
        -float(summary["recall_at_1"]),
        -float(summary["mrr"]),
        configured.index(method),
    )


def run(args: argparse.Namespace) -> int:
    assignments_payload = read_json(args.assignments)
    config = read_json(args.config)
    if assignments_payload.get("split") != SPLIT:
        raise ValueError("condition aggregation may read V19.2 development only")
    if assignments_payload.get("human_review_used") is not False:
        raise ValueError("condition aggregation must not use human review")
    rows = [dict(row) for row in assignments_payload.get("assignments", [])]
    if len(rows) != 48:
        raise ValueError("expected 48 V19.2 development queries")
    configured = [str(value) for value in config["retrieval_aggregation_grid"]]
    methods = [args.locked_method] if args.locked_method else configured
    receipt = read_json(args.index_dir / "index_receipt.json")
    if receipt.get("status") != "complete" or receipt.get("failed_item_count") != 0:
        raise ValueError("a complete failure-free ColQwen2 index is required")
    item_ids = [str(value) for value in receipt.get("indexed_item_ids", [])]
    shard_by_id = {
        path.stem: path for path in (args.index_dir / "shards").glob("*.npy")
    }
    if len(item_ids) != 1482 or any(item_id not in shard_by_id for item_id in item_ids):
        raise ValueError("ColQwen2 index must contain all 1,482 pages")

    query_components: list[tuple[str, str, str]] = []
    for row in rows:
        query_id = str(row["query_id"])
        query_components.append((query_id, "full", str(row["query"])))
        for index, phrase in enumerate(explicit_phrases(str(row["query"])), start=1):
            query_components.append((query_id, f"condition_{index}", phrase))

    import torch
    from transformers import ColQwen2ForRetrieval, ColQwen2Processor

    if not torch.cuda.is_available():
        raise RuntimeError("condition aggregation evaluation requires CUDA")
    free_bytes, _ = torch.cuda.mem_get_info()
    if free_bytes / 1024**3 < args.min_free_gib:
        raise RuntimeError("insufficient free GPU memory")
    model = ColQwen2ForRetrieval.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
    ).eval()  # type: ignore[no-untyped-call]
    processor = ColQwen2Processor.from_pretrained(args.model, use_fast=True)
    started = time.perf_counter()
    query_vectors: list[Any] = []
    for start in range(0, len(query_components), args.query_batch_size):
        texts = [value[2] for value in query_components[start : start + args.query_batch_size]]
        inputs = processor(text=texts).to(model.device)
        with torch.inference_mode():
            embeddings = model(**inputs).embeddings
        query_vectors.extend(
            embedding.detach().to(torch.bfloat16).cpu() for embedding in embeddings
        )
        del inputs, embeddings
    all_scores = torch.empty((len(query_vectors), len(item_ids)), dtype=torch.float32)
    shard_paths = [shard_by_id[item_id] for item_id in item_ids]
    for passage_start in range(0, len(shard_paths), args.passage_batch_size):
        paths = shard_paths[passage_start : passage_start + args.passage_batch_size]
        passages = [
            torch.from_numpy(load_embedding(path, expected_dim=128)).to(
                model.device, dtype=torch.bfloat16
            )
            for path in paths
        ]
        for query_start in range(0, len(query_vectors), args.query_batch_size):
            queries = [
                value.to(model.device)
                for value in query_vectors[
                    query_start : query_start + args.query_batch_size
                ]
            ]
            with torch.inference_mode():
                block = maxsim_score_matrix(queries, passages).cpu()
            all_scores[
                query_start : query_start + len(queries),
                passage_start : passage_start + len(passages),
            ] = block
        del passages
        torch.cuda.empty_cache()

    component_index = {
        (query_id, component): index
        for index, (query_id, component, _) in enumerate(query_components)
    }
    method_results: dict[str, list[dict[str, Any]]] = {method: [] for method in methods}
    for row in rows:
        query_id = str(row["query_id"])
        full = all_scores[component_index[(query_id, "full")]].tolist()
        conditions = [
            all_scores[component_index[(query_id, f"condition_{index}")]].tolist()
            for index in (1, 2)
        ]
        relevant = {str(value) for value in row.get("gold_relevant_item_ids", [])}
        for method in methods:
            aggregated = aggregate_rank_percentiles(
                full, conditions, method=method
            )
            order = ranked_indices(aggregated)
            top_indices = order[: args.stored_top_k]
            ranking = [item_ids[index] for index in top_indices]
            relevant_rank = next(
                (
                    rank
                    for rank, index in enumerate(order, start=1)
                    if item_ids[index] in relevant
                ),
                None,
            )
            method_results[method].append(
                {
                    **row,
                    "ranking_item_ids": ranking,
                    "scores": [round(aggregated[index], 8) for index in top_indices],
                    "relevant_rank": relevant_rank,
                }
            )
    summaries = {
        method: positive_retrieval_summary(results)
        for method, results in method_results.items()
    }
    selected_method = min(
        methods,
        key=lambda method: _selection_key(method, summaries[method], configured),
    )
    payload = {
        "schema_version": 1,
        "status": (
            "locked_method_stress_diagnostic_only"
            if args.locked_method
            else "automatic_development_method_selected"
        ),
        "study_id": "v19-2-condition-query-aggregation",
        "split": SPLIT,
        "eligible_for_final_claim": False,
        "human_review_used": False,
        "future_holdout_opened": False,
        "source_assignment_sha256": hashlib.sha256(
            args.assignments.read_bytes()
        ).hexdigest(),
        "selection_rule": config["retrieval_aggregation_selection"],
        "evaluated_methods": methods,
        "selected_method": selected_method,
        "summaries": summaries,
        "summary": summaries[selected_method],
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "results": method_results[selected_method],
    }
    write_json_atomic(args.output, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "selected_method": selected_method,
                "summaries": summaries,
            },
            indent=2,
        )
    )
    return 0


def main() -> int:
    args = parse_args()
    lock_path = args.index_dir / ".v19_2_condition_aggregation_gpu_job.lock"
    with exclusive_process_lock(lock_path):
        return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
