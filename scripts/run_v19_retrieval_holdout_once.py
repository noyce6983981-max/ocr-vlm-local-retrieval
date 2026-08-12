"""Authorize, claim, and execute the one-shot V19 retrieval holdout."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ocr_vlm_retrieval.gating.candidate_verification import (  # noqa: E402
    FULL_QUERY_INSTRUCTION,
    load_ocr_lines,
)
from ocr_vlm_retrieval.gating.literal_evidence import (  # noqa: E402
    literal_override_is_eligible,
    select_literal_candidate,
)
from ocr_vlm_retrieval.gating.ocr_literals import (  # noqa: E402
    extract_ocr_literal_groups,
)
from ocr_vlm_retrieval.routing.rule_router import RuleRouter  # noqa: E402
from ocr_vlm_retrieval.runtime.cache import write_json_atomic  # noqa: E402
from ocr_vlm_retrieval.runtime.late_interaction import (  # noqa: E402
    exclusive_process_lock,
    load_embedding,
)
from scripts.benchmark_v19_colqwen2_warm_latency import (  # noqa: E402
    percentile,
)
from scripts.evaluate_v19_colqwen2_development import (  # noqa: E402
    maxsim_score_matrix,
)
from scripts.evaluate_v19_ocr_literal_override import (  # noqa: E402
    paired_family_bootstrap,
    positive_recall_at_k,
    summarize_by_stratum,
)
from scripts.freeze_v19_retrieval_method import (  # noqa: E402
    DEFAULT_OUTPUT as METHOD_LOCK,
)
from scripts.freeze_v19_retrieval_method import (  # noqa: E402
    read_json,
    sha256_file,
    verify_lock,
)
from scripts.run_v19_downstream_retrieval_pilot import (  # noqa: E402
    ranking_ids,
    run_baseline,
)
from scripts.score_v19_v18_l1_development import (  # noqa: E402
    l1_result,
    manifest_index,
    summarize,
)

PRIVATE_DIR = ROOT / "records/private/v19/selective_intervention"
REVIEWED_QUERIES = PRIVATE_DIR / "reviewed_queries.jsonl"
AUTHORIZATION = PRIVATE_DIR / "retrieval_holdout_authorization.json"
CLAIM = PRIVATE_DIR / "retrieval_holdout_claim.json"
PRIVATE_RECEIPT = PRIVATE_DIR / "retrieval_holdout_results_receipt.json"
OUTPUT_DIR = ROOT / "outputs/evaluation/v19/selective_intervention/holdout_once"
ASSIGNMENTS = OUTPUT_DIR / "holdout_assignments.json"
BASELINE_RETRIEVAL_DIR = OUTPUT_DIR / "retrieval/baseline"
BASELINE_SCORES = OUTPUT_DIR / "v18_l1_top3.json"
COLQ_SCORES = OUTPUT_DIR / "colqwen2_literal_eligible.json"
FINAL_RESULTS = OUTPUT_DIR / "v19_literal_evidence_results.json"
LIBRARY_DIR = ROOT / "outputs/user_library"
ATTRIBUTE_POLICY = ROOT / "config/v17_attribute_coverage.json"
OCR_ROOT = LIBRARY_DIR / "ocr/json"
COLQ_MODEL = ROOT / "models/colqwen2-v1.0-hf"
COLQ_INDEX = LIBRARY_DIR / "colqwen2_v1_index"
V18_RERANKER = ROOT / "models/qwen3-vl-reranker-2b"
GPU_LOCK = COLQ_INDEX / ".v19_retrieval_holdout_gpu.lock"
HOLDOUT_SPLIT = "v19_retrieval_holdout_once"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"line {line_number} must contain an object")
            rows.append(payload)
    return rows


def write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def validate_method_lock() -> dict[str, Any]:
    lock = read_json(METHOD_LOCK)
    if lock.get("status") != "locked_after_development_before_holdout":
        raise ValueError("V19 retrieval method is not frozen")
    if lock.get("holdout_scored") is not False:
        raise ValueError("method lock records an earlier holdout execution")
    errors = verify_lock(lock)
    if errors:
        raise ValueError("V19 retrieval method lock failed: " + "; ".join(errors))
    return lock


def build_authorization(basis: str) -> dict[str, Any]:
    validate_method_lock()
    if not basis.strip():
        raise ValueError("explicit user authorization basis is required")
    if CLAIM.exists() or PRIVATE_RECEIPT.exists() or FINAL_RESULTS.exists():
        raise FileExistsError("V19 retrieval holdout was already claimed or completed")
    return {
        "schema_version": 1,
        "study_id": "v19-colqwen2-literal-evidence-e2e",
        "status": "authorized_for_one_shot_retrieval_holdout",
        "authorized_at_utc": datetime.now(UTC).isoformat(),
        "authorization_basis": basis.strip(),
        "method_lock_sha256": sha256_file(METHOD_LOCK),
        "reviewed_query_sha256": sha256_file(REVIEWED_QUERIES),
        "holdout_query_count": 100,
        "authorized_methods": [
            "frozen_v18_L1_max_verifier_score",
            "colqwen2_top3_query_text_literal_evidence_override",
        ],
        "rerun_policy": "forbidden_after_completed_receipt",
        "holdout_scored": False,
    }


def authorize(basis: str) -> None:
    write_json_new(AUTHORIZATION, build_authorization(basis))
    print("V19 retrieval holdout authorized; scoring has not started.")


def validate_authorization() -> dict[str, Any]:
    authorization = read_json(AUTHORIZATION)
    if authorization.get("status") != ("authorized_for_one_shot_retrieval_holdout"):
        raise ValueError("V19 retrieval holdout is not authorized")
    if authorization.get("holdout_scored") is not False:
        raise ValueError("authorization records an earlier holdout run")
    if authorization.get("method_lock_sha256") != sha256_file(METHOD_LOCK):
        raise ValueError("authorization references a different method lock")
    if authorization.get("reviewed_query_sha256") != sha256_file(REVIEWED_QUERIES):
        raise ValueError("authorization references different reviewed queries")
    validate_method_lock()
    return authorization


def build_claim() -> dict[str, Any]:
    validate_authorization()
    return {
        "schema_version": 1,
        "study_id": "v19-colqwen2-literal-evidence-e2e",
        "status": "retrieval_holdout_claimed_incomplete",
        "claimed_at_utc": datetime.now(UTC).isoformat(),
        "authorization_sha256": sha256_file(AUTHORIZATION),
        "method_lock_sha256": sha256_file(METHOD_LOCK),
        "reviewed_query_sha256": sha256_file(REVIEWED_QUERIES),
        "holdout_query_count": 100,
        "holdout_scoring_opened": True,
        "holdout_execution_complete": False,
        "resume_policy": "partial checkpoints may resume until final receipt exists",
    }


def ensure_claim(*, resume: bool) -> dict[str, Any]:
    if PRIVATE_RECEIPT.exists() or FINAL_RESULTS.exists():
        raise FileExistsError("V19 retrieval holdout already has final results")
    if not CLAIM.exists():
        claim = build_claim()
        write_json_new(CLAIM, claim)
        print("V19 retrieval holdout claimed; scoring is now one-shot.")
        return claim
    if not resume:
        raise FileExistsError("holdout is already claimed; pass --resume to recover")
    claim = read_json(CLAIM)
    if claim.get("status") != "retrieval_holdout_claimed_incomplete":
        raise ValueError("existing claim is not resumable")
    if claim.get("method_lock_sha256") != sha256_file(METHOD_LOCK):
        raise ValueError("claim references a different method lock")
    return claim


def holdout_rows() -> list[dict[str, Any]]:
    rows = [
        row for row in read_jsonl(REVIEWED_QUERIES) if row.get("split") == "holdout"
    ]
    if len(rows) != 100 or len({str(row["query_id"]) for row in rows}) != 100:
        raise ValueError("holdout must contain 100 unique queries")
    families = Counter(str(row.get("family_id")) for row in rows)
    if len(families) != 25 or set(families.values()) != {4}:
        raise ValueError("holdout must contain 25 four-query families")
    roles = Counter(str(row.get("query_role")) for row in rows)
    if len(roles) != 4 or set(roles.values()) != {25}:
        raise ValueError("holdout roles must be balanced")
    return rows


def prepare_assignments(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    router = RuleRouter.legacy()
    assignments: list[dict[str, Any]] = []
    for row in rows:
        query = " ".join(str(row.get("query_text", "")).split())
        if not query:
            raise ValueError(f"empty query: {row.get('query_id')}")
        relevant = [
            str(item_id)
            for item_id in row.get("gold_relevant_item_ids", [])
            if str(item_id).strip()
        ]
        assignments.append(
            {
                "query_id": str(row["query_id"]),
                "family_id": str(row["family_id"]),
                "query": query,
                "query_role": str(row["query_role"]),
                "content_stratum": str(row["content_stratum"]),
                "gold_answerable": bool(row["gold_answerable"]),
                "gold_relevant_item_ids": relevant,
                "source_item_id": str(row["target_item_id"]),
                "neighbor_item_id": str(row["neighbor_item_id"]),
                "legacy_route": router.route(query).route,
            }
        )
    payload = {
        "status": "one_shot_holdout_claimed",
        "split": HOLDOUT_SPLIT,
        "query_count": len(assignments),
        "method_lock_sha256": sha256_file(METHOD_LOCK),
        "source_query_sha256": sha256_file(REVIEWED_QUERIES),
        "assignments": assignments,
    }
    write_json_atomic(ASSIGNMENTS, payload)
    return payload


def materialize_baseline_retrieval(
    assignments: Sequence[Mapping[str, Any]], *, timeout: int
) -> None:
    for index, assignment in enumerate(assignments, start=1):
        query_id = str(assignment["query_id"])
        print(f"[baseline retrieval {index:03d}/{len(assignments)}] {query_id}")
        run_baseline(
            dict(assignment),
            output_path=BASELINE_RETRIEVAL_DIR / f"{query_id}_v18_frozen.json",
            library_dir=LIBRARY_DIR,
            attribute_policy=ATTRIBUTE_POLICY,
            timeout=timeout,
        )


def build_baseline_tasks(
    assignments: Sequence[Mapping[str, Any]], *, top_k: int
) -> list[dict[str, Any]]:
    manifest = manifest_index(LIBRARY_DIR)
    tasks: list[dict[str, Any]] = []
    for assignment in assignments:
        query_id = str(assignment["query_id"])
        retrieval = read_json(BASELINE_RETRIEVAL_DIR / f"{query_id}_v18_frozen.json")
        if retrieval.get("query") != assignment.get("query"):
            raise ValueError(f"retrieval query mismatch for {query_id}")
        candidates: list[dict[str, Any]] = []
        for rank, item_id in enumerate(ranking_ids(retrieval)[:top_k], start=1):
            row = manifest.get(item_id)
            if row is None:
                raise ValueError(f"manifest missing {item_id}")
            source_path = ROOT / str(row["source_path"])
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            candidates.append(
                {
                    "item_id": item_id,
                    "retrieval_rank": rank,
                    "image_path": str(source_path.resolve()),
                }
            )
        tasks.append(
            {
                **dict(assignment),
                "candidates": candidates,
                "retrieval_latency_ms": round(
                    float(retrieval.get("timings", {}).get("total_seconds", 0.0))
                    * 1000,
                    3,
                ),
            }
        )
    return tasks


def score_v18_baseline(tasks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    previous: dict[str, Any] = {}
    if BASELINE_SCORES.exists():
        previous = read_json(BASELINE_SCORES)
        if previous.get("method_lock_sha256") != sha256_file(METHOD_LOCK):
            raise ValueError("baseline checkpoint references a different method lock")
    existing = {str(row["query_id"]): row for row in previous.get("results", [])}
    pending = [task for task in tasks if str(task["query_id"]) not in existing]
    if pending:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("V18 L1 holdout scoring requires CUDA")
        free_bytes, _ = torch.cuda.mem_get_info()
        if free_bytes / 1024**3 < 5.0:
            raise RuntimeError("at least 5 GiB free GPU memory is required")
        official_repo = ROOT / "third_party/Qwen3-VL-Embedding"
        if str(official_repo) not in sys.path:
            sys.path.insert(0, str(official_repo))
        from src.models.qwen3_vl_reranker import (  # type: ignore[import-untyped]
            Qwen3VLReranker,
        )

        with exclusive_process_lock(GPU_LOCK):
            torch.cuda.empty_cache()
            model = Qwen3VLReranker(
                model_name_or_path=str(V18_RERANKER.resolve()),
                max_length=1024,
                min_pixels=32 * 32 * 4,
                max_pixels=384 * 384,
                dtype=torch.float16,
                attn_implementation="sdpa",
                low_cpu_mem_usage=True,
            )
            for index, task in enumerate(pending, start=1):
                print(
                    f"[V18 L1 {index:03d}/{len(pending)}] {task['query_id']}",
                    flush=True,
                )
                started = time.perf_counter()
                scores = model.process(
                    {
                        "instruction": FULL_QUERY_INSTRUCTION,
                        "query": {"text": task["query"]},
                        "documents": [
                            {"image": candidate["image_path"]}
                            for candidate in task["candidates"]
                        ],
                    }
                )
                latency_ms = (time.perf_counter() - started) * 1000
                result = l1_result(task, scores, threshold=0.63)
                result.update(
                    {
                        "family_id": task["family_id"],
                        "retrieval_latency_ms": task["retrieval_latency_ms"],
                        "verifier_latency_ms": round(latency_ms, 3),
                    }
                )
                existing[str(task["query_id"])] = result
                ordered = [
                    existing[str(row["query_id"])]
                    for row in tasks
                    if str(row["query_id"]) in existing
                ]
                write_json_atomic(
                    BASELINE_SCORES,
                    {
                        "status": "partial_one_shot_holdout",
                        "split": HOLDOUT_SPLIT,
                        "method": "frozen_v18_L1_max_verifier_score",
                        "method_lock_sha256": sha256_file(METHOD_LOCK),
                        "parameters": {"top_k": 3, "threshold": 0.63},
                        "results": ordered,
                    },
                )
            del model
            torch.cuda.empty_cache()
    ordered = [existing[str(task["query_id"])] for task in tasks]
    payload = {
        "status": "complete_one_shot_holdout",
        "split": HOLDOUT_SPLIT,
        "method": "frozen_v18_L1_max_verifier_score",
        "method_lock_sha256": sha256_file(METHOD_LOCK),
        "parameters": {"top_k": 3, "threshold": 0.63},
        **summarize(ordered),
        "positive_recall_at_3": positive_recall_at_k(ordered, cutoff=3),
        "results": ordered,
    }
    write_json_atomic(BASELINE_SCORES, payload)
    return payload


def eligible_assignments(
    assignments: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for assignment in assignments:
        query = str(assignment["query"])
        groups = extract_ocr_literal_groups(query)
        if literal_override_is_eligible(query, groups):
            selected.append(dict(assignment))
    return selected


def score_colq_eligible(
    assignments: Sequence[Mapping[str, Any]], *, stored_top_k: int = 50
) -> dict[str, Any]:
    if COLQ_SCORES.exists():
        existing = read_json(COLQ_SCORES)
        if existing.get("status") == "complete_one_shot_holdout" and existing.get(
            "method_lock_sha256"
        ) == sha256_file(METHOD_LOCK):
            return existing
        raise ValueError("incomplete ColQ holdout checkpoint requires manual audit")
    selected = eligible_assignments(assignments)
    receipt = read_json(COLQ_INDEX / "index_receipt.json")
    if receipt.get("status") != "complete" or receipt.get("failed_item_count") != 0:
        raise ValueError("complete failure-free ColQwen2 index required")
    item_ids = [str(value) for value in receipt.get("indexed_item_ids", [])]
    shard_paths = [COLQ_INDEX / "shards" / f"{item_id}.npy" for item_id in item_ids]
    missing = [path for path in shard_paths if not path.is_file()]
    if missing:
        raise ValueError(f"ColQwen2 index is missing {len(missing)} shards")

    import torch
    from transformers import ColQwen2ForRetrieval, ColQwen2Processor

    if not torch.cuda.is_available():
        raise RuntimeError("ColQwen2 holdout scoring requires CUDA")
    free_bytes, _ = torch.cuda.mem_get_info()
    if free_bytes / 1024**3 < 5.0:
        raise RuntimeError("at least 5 GiB free GPU memory is required")
    with exclusive_process_lock(GPU_LOCK):
        model = ColQwen2ForRetrieval.from_pretrained(
            COLQ_MODEL,
            dtype=torch.bfloat16,
            device_map="cuda",
            attn_implementation="sdpa",
        ).eval()  # type: ignore[no-untyped-call]
        processor = ColQwen2Processor.from_pretrained(COLQ_MODEL, use_fast=True)
        passages = [
            torch.from_numpy(load_embedding(path, expected_dim=128)).to(
                device=model.device, dtype=torch.bfloat16
            )
            for path in shard_paths
        ]

        def score_one(
            query: str,
            loaded_model: Any = model,
            loaded_processor: Any = processor,
            loaded_passages: Sequence[Any] = passages,
        ) -> tuple[list[str], list[float], float]:
            started = time.perf_counter()
            inputs = loaded_processor(text=[query]).to(loaded_model.device)
            with torch.inference_mode():
                query_embedding = loaded_model(**inputs).embeddings[0]
                blocks = [
                    maxsim_score_matrix(
                        [query_embedding], loaded_passages[start : start + 16]
                    ).cpu()
                    for start in range(0, len(loaded_passages), 16)
                ]
            scores = torch.cat(blocks, dim=1)[0]
            values, indices = torch.topk(scores, k=min(stored_top_k, len(item_ids)))
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - started) * 1000
            ranking = [item_ids[int(index)] for index in indices.tolist()]
            ranked_scores = [round(float(value), 6) for value in values.tolist()]
            del inputs, query_embedding, blocks, scores, values, indices
            return ranking, ranked_scores, elapsed_ms

        if selected:
            score_one(str(selected[0]["query"]))
        results: list[dict[str, Any]] = []
        for index, assignment in enumerate(selected, start=1):
            print(
                f"[ColQ {index:03d}/{len(selected)}] {assignment['query_id']}",
                flush=True,
            )
            ranking, scores, latency_ms = score_one(str(assignment["query"]))
            results.append(
                {
                    "query_id": assignment["query_id"],
                    "ranking_item_ids": ranking,
                    "scores": scores,
                    "retrieval_latency_ms": round(latency_ms, 3),
                }
            )
        del model, processor, passages
        torch.cuda.empty_cache()
    payload = {
        "status": "complete_one_shot_holdout",
        "split": HOLDOUT_SPLIT,
        "method": "colqwen2_v1_multivector_late_interaction",
        "method_lock_sha256": sha256_file(METHOD_LOCK),
        "eligible_query_count": len(selected),
        "stored_top_k": stored_top_k,
        "results": results,
    }
    write_json_atomic(COLQ_SCORES, payload)
    return payload


def hard_negative_far(results: Sequence[Mapping[str, Any]]) -> float:
    rows = [
        row
        for row in results
        if row.get("query_role") == "single_condition_hard_negative"
    ]
    return sum(bool(row.get("accepted")) for row in rows) / len(rows)


def e2e_latency(row: Mapping[str, Any]) -> float:
    return float(row.get("retrieval_latency_ms", 0.0)) + float(
        row.get("verifier_latency_ms", 0.0)
    )


def finalize_results(
    assignments: Sequence[Mapping[str, Any]],
    baseline: Mapping[str, Any],
    colq: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_by_id = {str(row["query_id"]): row for row in baseline.get("results", [])}
    colq_by_id = {str(row["query_id"]): row for row in colq.get("results", [])}
    candidate_results: list[dict[str, Any]] = []
    baseline_latencies: list[float] = []
    candidate_latencies: list[float] = []
    decisions: list[dict[str, Any]] = []
    for assignment in assignments:
        query_id = str(assignment["query_id"])
        baseline_row = baseline_by_id[query_id]
        baseline_latency = e2e_latency(baseline_row)
        baseline_latencies.append(baseline_latency)
        query = str(assignment["query"])
        groups = extract_ocr_literal_groups(query)
        eligible = literal_override_is_eligible(query, groups)
        if not eligible:
            candidate_results.append({**baseline_row, "decision_source": "v18_l1"})
            candidate_latencies.append(baseline_latency)
            continue
        retrieval = colq_by_id.get(query_id)
        if retrieval is None:
            raise ValueError(f"eligible query missing ColQ result: {query_id}")
        item_ids = [str(value) for value in retrieval["ranking_item_ids"][:3]]
        literal_started = time.perf_counter()
        lines_by_item = {
            item_id: load_ocr_lines(
                OCR_ROOT / f"{item_id}.json", minimum_confidence=0.35
            )
            for item_id in item_ids
        }
        decision = select_literal_candidate(
            query, item_ids, lines_by_item, fuzzy_threshold=0.88
        )
        literal_latency_ms = (time.perf_counter() - literal_started) * 1000
        selected_item_id = decision["selected_item_id"]
        candidate_results.append(
            {
                **baseline_row,
                "candidate_item_ids": item_ids,
                "scores": [],
                "selected_item_id": selected_item_id,
                "selected_retrieval_rank": (
                    item_ids.index(str(selected_item_id)) + 1
                    if selected_item_id is not None
                    else None
                ),
                "accepted": bool(decision["accepted"]),
                "top_score": None,
                "decision_source": "query_text_literal_evidence_override",
                "colq_retrieval_latency_ms": retrieval["retrieval_latency_ms"],
                "literal_evidence_latency_ms": round(literal_latency_ms, 3),
            }
        )
        candidate_latencies.append(
            float(retrieval["retrieval_latency_ms"]) + literal_latency_ms
        )
        decisions.append(
            {
                "query_id": query_id,
                "override_eligible": True,
                **decision,
            }
        )

    baseline_rows = list(baseline_by_id.values())
    baseline_summary = summarize(baseline_rows)
    candidate_summary = summarize(candidate_results)
    baseline_summary["positive_recall_at_3"] = positive_recall_at_k(
        baseline_rows, cutoff=3
    )
    candidate_summary["positive_recall_at_3"] = positive_recall_at_k(
        candidate_results, cutoff=3
    )
    baseline_summary["hard_negative_far"] = hard_negative_far(baseline_rows)
    candidate_summary["hard_negative_far"] = hard_negative_far(candidate_results)
    delta = {
        key: round(float(candidate_summary[key]) - float(baseline_summary[key]), 8)
        for key in (
            "positive_selected_relevant",
            "positive_recall_at_3",
            "negative_correct_reject_rate",
            "end_to_end_accuracy",
            "false_accept_rate",
            "false_reject_rate",
            "hard_negative_far",
        )
    }
    baseline_p95 = percentile(baseline_latencies, 0.95)
    candidate_p95 = percentile(candidate_latencies, 0.95)
    latency_ratio = candidate_p95 / baseline_p95 if baseline_p95 else float("inf")
    gates = {
        "end_to_end_gain": delta["end_to_end_accuracy"] >= 0.05,
        "recall_at_3_gain": delta["positive_recall_at_3"] >= 0.05,
        "hard_negative_far_non_inferior": delta["hard_negative_far"] <= 0.0,
        "correct_rejection_non_inferior": (
            delta["negative_correct_reject_rate"] >= 0.0
        ),
        "warm_e2e_p95_ratio": latency_ratio <= 1.10,
        "runtime_eligibility_is_query_text_only": True,
    }
    payload = {
        "schema_version": 1,
        "study_id": "v19-colqwen2-literal-evidence-e2e",
        "status": "one_shot_holdout_complete",
        "split": HOLDOUT_SPLIT,
        "eligible_for_final_claim": all(gates.values()),
        "method_lock_sha256": sha256_file(METHOD_LOCK),
        "authorization_sha256": sha256_file(AUTHORIZATION),
        "claim_sha256_before_completion": sha256_file(CLAIM),
        "query_count": len(candidate_results),
        "override_eligible_query_count": len(decisions),
        "baseline": baseline_summary,
        "candidate": candidate_summary,
        "delta": delta,
        "paired_family_bootstrap": paired_family_bootstrap(
            baseline_rows, candidate_results
        ),
        "latency": {
            "scope": "warm method execution; model and index load excluded",
            "baseline_p50_ms": round(percentile(baseline_latencies, 0.50), 3),
            "baseline_p95_ms": round(baseline_p95, 3),
            "candidate_p50_ms": round(percentile(candidate_latencies, 0.50), 3),
            "candidate_p95_ms": round(candidate_p95, 3),
            "candidate_to_baseline_p95_ratio": round(latency_ratio, 6),
        },
        "release_gates": gates,
        "all_release_gates_passed": all(gates.values()),
        "baseline_by_stratum": summarize_by_stratum(baseline_rows),
        "candidate_by_stratum": summarize_by_stratum(candidate_results),
        "decisions": decisions,
        "results": candidate_results,
    }
    write_json_atomic(FINAL_RESULTS, payload)
    return payload


def complete_claim(result: Mapping[str, Any]) -> None:
    receipt = {
        "schema_version": 1,
        "study_id": result["study_id"],
        "status": "v19_retrieval_one_shot_holdout_complete",
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "eligible_for_final_claim": result["eligible_for_final_claim"],
        "method_lock_sha256": sha256_file(METHOD_LOCK),
        "authorization_sha256": sha256_file(AUTHORIZATION),
        "claim_sha256_before_completion": sha256_file(CLAIM),
        "final_results_sha256": sha256_file(FINAL_RESULTS),
        "query_count": result["query_count"],
        "baseline": result["baseline"],
        "candidate": result["candidate"],
        "delta": result["delta"],
        "paired_family_bootstrap": result["paired_family_bootstrap"],
        "latency": result["latency"],
        "release_gates": result["release_gates"],
        "all_release_gates_passed": result["all_release_gates_passed"],
    }
    write_json_new(PRIVATE_RECEIPT, receipt)
    claim = read_json(CLAIM)
    claim.update(
        {
            "status": "retrieval_holdout_claimed_complete",
            "holdout_execution_complete": True,
            "completed_at_utc": receipt["completed_at_utc"],
            "result_receipt_sha256": sha256_file(PRIVATE_RECEIPT),
        }
    )
    write_json_atomic(CLAIM, claim)


def execute(*, resume: bool, timeout: int) -> dict[str, Any]:
    ensure_claim(resume=resume)
    rows = holdout_rows()
    assignments_payload = prepare_assignments(rows)
    assignments = list(assignments_payload["assignments"])
    materialize_baseline_retrieval(assignments, timeout=timeout)
    tasks = build_baseline_tasks(assignments, top_k=3)
    baseline = score_v18_baseline(tasks)
    colq = score_colq_eligible(assignments)
    result = finalize_results(assignments, baseline, colq)
    complete_claim(result)
    print(
        "V19 retrieval one-shot holdout complete: "
        f"baseline={result['baseline']['end_to_end_accuracy']:.3f}, "
        f"candidate={result['candidate']['end_to_end_accuracy']:.3f}, "
        f"delta={result['delta']['end_to_end_accuracy']:+.3f}, "
        f"gates={result['all_release_gates_passed']}"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    authorization = subparsers.add_parser("authorize")
    authorization.add_argument("--basis", required=True)
    execution = subparsers.add_parser("execute")
    execution.add_argument("--resume", action="store_true")
    execution.add_argument("--timeout", type=int, default=300)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "authorize":
        authorize(args.basis)
        return 0
    execute(resume=args.resume, timeout=args.timeout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
