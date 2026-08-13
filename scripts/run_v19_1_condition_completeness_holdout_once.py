"""Authorize and execute the sealed V19.1 holdout exactly once."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
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
from ocr_vlm_retrieval.gating.literal_evidence_v19_1 import (  # noqa: E402
    select_v19_1_literal_candidate,
)
from ocr_vlm_retrieval.gating.ocr_literals import (  # noqa: E402
    OcrLiteralGroup,
)
from ocr_vlm_retrieval.runtime.cache import write_json_atomic  # noqa: E402
from ocr_vlm_retrieval.runtime.late_interaction import (  # noqa: E402
    exclusive_process_lock,
    load_embedding,
)
from scripts.benchmark_v19_colqwen2_warm_latency import percentile  # noqa: E402
from scripts.evaluate_v19_colqwen2_development import (  # noqa: E402
    maxsim_score_matrix,
)
from scripts.evaluate_v19_ocr_literal_override import (  # noqa: E402
    paired_family_bootstrap,
    positive_recall_at_k,
    summarize_by_stratum,
)
from scripts.freeze_v19_1_method import verify_lock  # noqa: E402
from scripts.run_v19_downstream_retrieval_pilot import (  # noqa: E402
    ranking_ids,
    read_json,
    run_baseline,
)
from scripts.score_v19_v18_l1_development import (  # noqa: E402
    l1_result,
    manifest_index,
    summarize,
)

STUDY_ID = "v19-1-condition-completeness-e2e"
HOLDOUT_SPLIT = "v19_1_human_reviewed_holdout_once"
PRIVATE_DIR = ROOT / "records/private/v19_1/condition_completeness"
SEALED_QUERIES = PRIVATE_DIR / "holdout_queries_human_reviewed_sealed.json"
REVIEWS = PRIVATE_DIR / "reviewer_01_holdout_query_reviews.jsonl"
METHOD_LOCK = ROOT / "config/studies/v19_1_condition_completeness_method_lock.json"
AUTHORIZATION = PRIVATE_DIR / "holdout_once_authorization.json"
CLAIM = PRIVATE_DIR / "holdout_once_claim.json"
RECEIPT = PRIVATE_DIR / "holdout_once_results_receipt.json"
OUTPUT_DIR = ROOT / "outputs/evaluation/v19_1/condition_completeness/holdout_once"
ASSIGNMENTS = OUTPUT_DIR / "holdout_assignments.json"
BASELINE_RETRIEVAL_DIR = OUTPUT_DIR / "retrieval/v18_frozen"
BASELINE_SCORES = OUTPUT_DIR / "v18_l1_top3.json"
COLQ_SCORES = OUTPUT_DIR / "colqwen2_all_queries.json"
FINAL_RESULTS = OUTPUT_DIR / "v19_1_condition_completeness_results.json"
LIBRARY_DIR = ROOT / "outputs/user_library"
ATTRIBUTE_POLICY = ROOT / "config/v17_attribute_coverage.json"
OCR_ROOT = LIBRARY_DIR / "ocr/json"
COLQ_MODEL = ROOT / "models/colqwen2-v1.0-hf"
COLQ_INDEX = LIBRARY_DIR / "colqwen2_v1_index"
V18_RERANKER = ROOT / "models/qwen3-vl-reranker-2b"
GPU_LOCK = LIBRARY_DIR / ".v19_1_holdout_gpu_job.lock"
EXECUTOR = Path(__file__).resolve()
ROLES = (
    "answerable_positive",
    "paraphrase_positive",
    "single_condition_hard_negative",
    "unanswerable_neighbor",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def validate_method_lock() -> dict[str, Any]:
    lock = read_json(METHOD_LOCK)
    if lock.get("status") != "locked_after_human_reviewed_development_before_holdout":
        raise ValueError("V19.1 method is not in its frozen pre-holdout state")
    if lock.get("holdout_scored") is not False:
        raise ValueError("method lock records an earlier holdout execution")
    errors = verify_lock(lock)
    if errors:
        raise ValueError("V19.1 method lock failed: " + "; ".join(errors))
    return lock


def sealed_payload() -> dict[str, Any]:
    payload = read_json(SEALED_QUERIES)
    if payload.get("status") != "human_reviewed_holdout_sealed_not_scored":
        raise ValueError("V19.1 holdout queries are not sealed")
    if payload.get("split") != "v19_1_human_reviewed_holdout_sealed":
        raise ValueError("unexpected V19.1 holdout split")
    if payload.get("retrieval_run") is not False:
        raise ValueError("sealed query payload records prior retrieval")
    if payload.get("candidate_scored") is not False:
        raise ValueError("sealed query payload records prior scoring")
    rows = list(payload.get("assignments", []))
    if len(rows) != 48 or len({str(row["query_id"]) for row in rows}) != 48:
        raise ValueError("holdout must contain 48 unique queries")
    families = Counter(str(row["family_id"]) for row in rows)
    roles = Counter(str(row["query_role"]) for row in rows)
    if len(families) != 12 or set(families.values()) != {4}:
        raise ValueError("holdout must contain 12 four-query families")
    if roles != Counter({role: 12 for role in ROLES}):
        raise ValueError("holdout query roles are not balanced")
    if sum(bool(row["gold_answerable"]) for row in rows) != 24:
        raise ValueError("holdout must contain 24 answerable queries")
    if any(row.get("review_status") != "human_approved_holdout_family" for row in rows):
        raise ValueError("every holdout query must be human reviewed")
    return payload


def build_authorization(basis: str) -> dict[str, Any]:
    lock = validate_method_lock()
    payload = sealed_payload()
    if not basis.strip():
        raise ValueError("explicit user authorization basis is required")
    if CLAIM.exists() or RECEIPT.exists() or FINAL_RESULTS.exists():
        raise FileExistsError("V19.1 holdout was already claimed or completed")
    return {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "status": "authorized_for_one_shot_v19_1_holdout",
        "authorized_at_utc": datetime.now(UTC).isoformat(),
        "authorization_basis": basis.strip(),
        "method_lock_sha256": sha256_file(METHOD_LOCK),
        "sealed_query_sha256": sha256_file(SEALED_QUERIES),
        "review_sha256": sha256_file(REVIEWS),
        "executor_sha256": sha256_file(EXECUTOR),
        "selection_commit_sha": lock["selection_commit_sha"],
        "executor_commit_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "holdout_family_count": payload["family_count"],
        "holdout_query_count": payload["query_count"],
        "authorized_methods": [
            "frozen_v18_L1_max_verifier_score",
            "frozen_v19_colqwen2_top3_literal_evidence",
            "v19_1_colqwen2_top3_complete_literal_evidence",
        ],
        "rerun_policy": "forbidden_after_completed_receipt",
        "resume_policy": "partial checkpoints may resume before final receipt",
        "holdout_scored": False,
    }


def authorize(basis: str) -> dict[str, Any]:
    payload = build_authorization(basis)
    write_json_new(AUTHORIZATION, payload)
    return payload


def validate_authorization() -> dict[str, Any]:
    payload = read_json(AUTHORIZATION)
    if payload.get("status") != "authorized_for_one_shot_v19_1_holdout":
        raise ValueError("V19.1 holdout is not authorized")
    checks = {
        "method_lock_sha256": sha256_file(METHOD_LOCK),
        "sealed_query_sha256": sha256_file(SEALED_QUERIES),
        "review_sha256": sha256_file(REVIEWS),
        "executor_sha256": sha256_file(EXECUTOR),
    }
    for field, expected in checks.items():
        if payload.get(field) != expected:
            raise ValueError(f"authorization hash mismatch: {field}")
    if payload.get("holdout_scored") is not False:
        raise ValueError("authorization records a prior holdout run")
    validate_method_lock()
    sealed_payload()
    return payload


def build_claim() -> dict[str, Any]:
    validate_authorization()
    return {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "status": "v19_1_holdout_claimed_incomplete",
        "claimed_at_utc": datetime.now(UTC).isoformat(),
        "authorization_sha256": sha256_file(AUTHORIZATION),
        "method_lock_sha256": sha256_file(METHOD_LOCK),
        "sealed_query_sha256": sha256_file(SEALED_QUERIES),
        "executor_sha256": sha256_file(EXECUTOR),
        "holdout_query_count": 48,
        "holdout_scoring_opened": True,
        "holdout_execution_complete": False,
        "resume_policy": "partial checkpoints may resume until final receipt exists",
    }


def ensure_claim(*, resume: bool) -> dict[str, Any]:
    if RECEIPT.exists() or FINAL_RESULTS.exists():
        raise FileExistsError("V19.1 holdout already has final results")
    if not CLAIM.exists():
        claim = build_claim()
        write_json_new(CLAIM, claim)
        print("V19.1 holdout claimed; final scoring is now one-shot.", flush=True)
        return claim
    if not resume:
        raise FileExistsError("holdout already claimed; pass --resume to recover")
    claim = read_json(CLAIM)
    expected = {
        "status": "v19_1_holdout_claimed_incomplete",
        "method_lock_sha256": sha256_file(METHOD_LOCK),
        "sealed_query_sha256": sha256_file(SEALED_QUERIES),
        "executor_sha256": sha256_file(EXECUTOR),
    }
    for field, value in expected.items():
        if claim.get(field) != value:
            raise ValueError(f"claim mismatch: {field}")
    return claim


def prepare_assignments() -> dict[str, Any]:
    source = sealed_payload()
    assignments = [dict(row) for row in source["assignments"]]
    payload = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "status": "one_shot_holdout_claimed",
        "split": HOLDOUT_SPLIT,
        "query_count": len(assignments),
        "method_lock_sha256": sha256_file(METHOD_LOCK),
        "source_query_sha256": sha256_file(SEALED_QUERIES),
        "assignments": assignments,
    }
    write_json_atomic(ASSIGNMENTS, payload)
    return payload


def materialize_baseline_retrieval(
    assignments: Sequence[Mapping[str, Any]], *, timeout: int
) -> None:
    for index, assignment in enumerate(assignments, start=1):
        query_id = str(assignment["query_id"])
        print(f"[baseline retrieval {index:02d}/{len(assignments)}] {query_id}")
        run_baseline(
            dict(assignment),
            output_path=BASELINE_RETRIEVAL_DIR / f"{query_id}_v18_frozen.json",
            library_dir=LIBRARY_DIR,
            attribute_policy=ATTRIBUTE_POLICY,
            timeout=timeout,
        )


def build_baseline_tasks(
    assignments: Sequence[Mapping[str, Any]], *, top_k: int = 3
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
            image_path = ROOT / str(row["source_path"])
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            candidates.append(
                {
                    "item_id": item_id,
                    "retrieval_rank": rank,
                    "image_path": str(image_path.resolve()),
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
    if len(tasks) != 48:
        raise ValueError(f"expected 48 holdout tasks, got {len(tasks)}")
    return tasks


def score_v18_baseline(tasks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    previous = read_json(BASELINE_SCORES) if BASELINE_SCORES.exists() else {}
    if previous and previous.get("method_lock_sha256") != sha256_file(METHOD_LOCK):
        raise ValueError("baseline checkpoint references another method lock")
    if previous and previous.get("source_query_sha256") != sha256_file(
        SEALED_QUERIES
    ):
        raise ValueError("baseline checkpoint references different sealed queries")
    existing = {str(row["query_id"]): row for row in previous.get("results", [])}
    pending = [row for row in tasks if str(row["query_id"]) not in existing]
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
        from src.models.qwen3_vl_reranker import (
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
                print(f"[V18 L1 {index:02d}/{len(pending)}] {task['query_id']}")
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
                result = l1_result(task, scores, threshold=0.63)
                result.update(
                    {
                        "family_id": task["family_id"],
                        "retrieval_latency_ms": task["retrieval_latency_ms"],
                        "verifier_latency_ms": round(
                            (time.perf_counter() - started) * 1000, 3
                        ),
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
                        "source_query_sha256": sha256_file(SEALED_QUERIES),
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
        "source_query_sha256": sha256_file(SEALED_QUERIES),
        "parameters": {"top_k": 3, "threshold": 0.63},
        "summary": summarize(ordered),
        "positive_recall_at_3": positive_recall_at_k(ordered, cutoff=3),
        "results": ordered,
    }
    write_json_atomic(BASELINE_SCORES, payload)
    return payload


def score_colq_all(
    assignments: Sequence[Mapping[str, Any]], *, stored_top_k: int = 50
) -> dict[str, Any]:
    previous = read_json(COLQ_SCORES) if COLQ_SCORES.exists() else {}
    if previous and previous.get("method_lock_sha256") != sha256_file(METHOD_LOCK):
        raise ValueError("ColQ checkpoint references another method lock")
    if previous and previous.get("source_query_sha256") != sha256_file(
        SEALED_QUERIES
    ):
        raise ValueError("ColQ checkpoint references different sealed queries")
    existing = {str(row["query_id"]): row for row in previous.get("results", [])}
    pending = [row for row in assignments if str(row["query_id"]) not in existing]
    if not pending:
        if previous.get("status") != "complete_one_shot_holdout":
            raise ValueError("complete ColQ rows have an invalid checkpoint status")
        return previous
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
            latency_ms = (time.perf_counter() - started) * 1000
            ranking = [item_ids[int(index)] for index in indices.tolist()]
            ranked_scores = [round(float(value), 6) for value in values.tolist()]
            del inputs, query_embedding, blocks, scores, values, indices
            return ranking, ranked_scores, latency_ms

        score_one(str(pending[0]["query"]))
        for index, assignment in enumerate(pending, start=1):
            print(f"[ColQ {index:02d}/{len(pending)}] {assignment['query_id']}")
            ranking, scores, latency_ms = score_one(str(assignment["query"]))
            existing[str(assignment["query_id"])] = {
                "query_id": assignment["query_id"],
                "ranking_item_ids": ranking,
                "scores": scores,
                "retrieval_latency_ms": round(latency_ms, 3),
            }
            ordered = [
                existing[str(row["query_id"])]
                for row in assignments
                if str(row["query_id"]) in existing
            ]
            write_json_atomic(
                COLQ_SCORES,
                {
                    "status": "partial_one_shot_holdout",
                    "split": HOLDOUT_SPLIT,
                    "method": "colqwen2_v1_multivector_late_interaction",
                    "method_lock_sha256": sha256_file(METHOD_LOCK),
                    "source_query_sha256": sha256_file(SEALED_QUERIES),
                    "query_count": len(ordered),
                    "stored_top_k": stored_top_k,
                    "results": ordered,
                },
            )
        del model, processor, passages
        torch.cuda.empty_cache()
    results = [existing[str(row["query_id"])] for row in assignments]
    payload = {
        "status": "complete_one_shot_holdout",
        "split": HOLDOUT_SPLIT,
        "method": "colqwen2_v1_multivector_late_interaction",
        "method_lock_sha256": sha256_file(METHOD_LOCK),
        "source_query_sha256": sha256_file(SEALED_QUERIES),
        "query_count": len(results),
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
    if not rows:
        raise ValueError("hard-negative rows are required")
    return sum(bool(row.get("accepted")) for row in rows) / len(rows)


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = summarize(rows)
    result["positive_recall_at_3"] = positive_recall_at_k(rows, cutoff=3)
    result["hard_negative_far"] = hard_negative_far(rows)
    return result


def _apply_literal_method(
    *,
    assignments: Sequence[Mapping[str, Any]],
    baseline: Mapping[str, Any],
    colq: Mapping[str, Any],
    method: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[float]]:
    baseline_by_id = {str(row["query_id"]): row for row in baseline["results"]}
    colq_by_id = {str(row["query_id"]): row for row in colq["results"]}
    results: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    latencies: list[float] = []
    for assignment in assignments:
        query_id = str(assignment["query_id"])
        query = str(assignment["query"])
        baseline_row = dict(baseline_by_id[query_id])
        baseline_latency = float(baseline_row.get("retrieval_latency_ms", 0.0)) + float(
            baseline_row.get("verifier_latency_ms", 0.0)
        )
        retrieval = colq_by_id[query_id]
        item_ids = [str(value) for value in retrieval["ranking_item_ids"][:3]]
        lines_by_item = {
            item_id: load_ocr_lines(OCR_ROOT / f"{item_id}.json", minimum_confidence=0.35)
            for item_id in item_ids
        }
        started = time.perf_counter()
        if method == "v19":
            raw_decision = select_literal_candidate(
                query, item_ids, lines_by_item, fuzzy_threshold=0.88
            )
            groups = tuple(
                OcrLiteralGroup(
                    label=str(group["label"]),
                    variants=tuple(str(value) for value in group["variants"]),
                    source=str(group["source"]),
                )
                for group in raw_decision["constraint_groups"]
            )
            eligible = literal_override_is_eligible(query, groups)
            decision = {"eligible": eligible, **raw_decision}
            source = "frozen_v19_literal_evidence"
        elif method == "v19_1":
            decision = select_v19_1_literal_candidate(
                query, item_ids, lines_by_item, fuzzy_threshold=0.88
            )
            eligible = bool(decision["eligible"])
            source = "v19_1_complete_literal_evidence"
        else:
            raise ValueError(f"unknown literal method: {method}")
        evidence_latency_ms = (time.perf_counter() - started) * 1000
        if not eligible:
            result = {**baseline_row, "decision_source": "v18_l1_fallback"}
            latency = baseline_latency
        else:
            selected = decision["selected_item_id"]
            result = {
                **baseline_row,
                "candidate_item_ids": item_ids,
                "scores": [],
                "selected_item_id": selected,
                "selected_retrieval_rank": (
                    item_ids.index(str(selected)) + 1 if selected is not None else None
                ),
                "accepted": bool(decision["accepted"]),
                "top_score": None,
                "decision_source": source,
                "colq_retrieval_latency_ms": retrieval["retrieval_latency_ms"],
                "literal_evidence_latency_ms": round(evidence_latency_ms, 3),
            }
            latency = float(retrieval["retrieval_latency_ms"]) + evidence_latency_ms
        results.append(result)
        latencies.append(latency)
        decisions.append(
            {
                "query_id": query_id,
                "query_role": assignment["query_role"],
                "gold_answerable": bool(assignment["gold_answerable"]),
                **decision,
            }
        )
    return results, decisions, latencies


def release_gates(
    baseline: Mapping[str, Any], predecessor: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, bool]:
    return {
        "e2e_gain_vs_v18_at_least_5pp": (
            float(candidate["end_to_end_accuracy"])
            - float(baseline["end_to_end_accuracy"])
            >= 0.05
        ),
        "recall_at_3_gain_vs_v18_at_least_5pp": (
            float(candidate["positive_recall_at_3"])
            - float(baseline["positive_recall_at_3"])
            >= 0.05
        ),
        "hard_negative_far_non_inferior_vs_v18": (
            float(candidate["hard_negative_far"])
            <= float(baseline["hard_negative_far"])
        ),
        "positive_selected_non_inferior_vs_v19": (
            float(candidate["positive_selected_relevant"])
            >= float(predecessor["positive_selected_relevant"])
        ),
        "e2e_improved_vs_v19": (
            float(candidate["end_to_end_accuracy"])
            > float(predecessor["end_to_end_accuracy"])
        ),
    }


def finalize_results(
    assignments: Sequence[Mapping[str, Any]],
    baseline: Mapping[str, Any],
    colq: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_rows = [dict(row) for row in baseline["results"]]
    predecessor_rows, predecessor_decisions, predecessor_latency = _apply_literal_method(
        assignments=assignments, baseline=baseline, colq=colq, method="v19"
    )
    candidate_rows, candidate_decisions, candidate_latency = _apply_literal_method(
        assignments=assignments, baseline=baseline, colq=colq, method="v19_1"
    )
    baseline_summary = _summary(baseline_rows)
    predecessor_summary = _summary(predecessor_rows)
    candidate_summary = _summary(candidate_rows)
    gates = release_gates(baseline_summary, predecessor_summary, candidate_summary)
    result = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "status": "one_shot_holdout_complete",
        "split": HOLDOUT_SPLIT,
        "eligible_for_final_claim": all(gates.values()),
        "method_lock_sha256": sha256_file(METHOD_LOCK),
        "authorization_sha256": sha256_file(AUTHORIZATION),
        "claim_sha256_before_completion": sha256_file(CLAIM),
        "sealed_query_sha256": sha256_file(SEALED_QUERIES),
        "executor_sha256": sha256_file(EXECUTOR),
        "query_count": len(candidate_rows),
        "baseline_v18": baseline_summary,
        "predecessor_v19": predecessor_summary,
        "candidate_v19_1": candidate_summary,
        "delta_v19_1_vs_v18": {
            key: round(float(candidate_summary[key]) - float(baseline_summary[key]), 8)
            for key in candidate_summary
            if isinstance(candidate_summary[key], (int, float))
            and key in baseline_summary
        },
        "delta_v19_1_vs_v19": {
            key: round(
                float(candidate_summary[key]) - float(predecessor_summary[key]), 8
            )
            for key in candidate_summary
            if isinstance(candidate_summary[key], (int, float))
            and key in predecessor_summary
        },
        "paired_family_bootstrap_vs_v18": paired_family_bootstrap(
            baseline_rows, candidate_rows
        ),
        "paired_family_bootstrap_vs_v19": paired_family_bootstrap(
            predecessor_rows, candidate_rows
        ),
        "latency": {
            "scope": "warm method execution; model and index load excluded",
            "v19_p50_ms": round(percentile(predecessor_latency, 0.50), 3),
            "v19_p95_ms": round(percentile(predecessor_latency, 0.95), 3),
            "v19_1_p50_ms": round(percentile(candidate_latency, 0.50), 3),
            "v19_1_p95_ms": round(percentile(candidate_latency, 0.95), 3),
        },
        "release_gates": gates,
        "all_release_gates_passed": all(gates.values()),
        "baseline_by_stratum": summarize_by_stratum(baseline_rows),
        "predecessor_by_stratum": summarize_by_stratum(predecessor_rows),
        "candidate_by_stratum": summarize_by_stratum(candidate_rows),
        "predecessor_decisions": predecessor_decisions,
        "candidate_decisions": candidate_decisions,
        "results": candidate_rows,
    }
    write_json_atomic(FINAL_RESULTS, result)
    return result


def complete_claim(result: Mapping[str, Any]) -> None:
    receipt = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "status": "v19_1_one_shot_holdout_complete",
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "eligible_for_final_claim": result["eligible_for_final_claim"],
        "method_lock_sha256": sha256_file(METHOD_LOCK),
        "authorization_sha256": sha256_file(AUTHORIZATION),
        "claim_sha256_before_completion": sha256_file(CLAIM),
        "sealed_query_sha256": sha256_file(SEALED_QUERIES),
        "executor_sha256": sha256_file(EXECUTOR),
        "final_results_sha256": sha256_file(FINAL_RESULTS),
        "query_count": result["query_count"],
        "baseline_v18": result["baseline_v18"],
        "predecessor_v19": result["predecessor_v19"],
        "candidate_v19_1": result["candidate_v19_1"],
        "delta_v19_1_vs_v18": result["delta_v19_1_vs_v18"],
        "delta_v19_1_vs_v19": result["delta_v19_1_vs_v19"],
        "release_gates": result["release_gates"],
        "all_release_gates_passed": result["all_release_gates_passed"],
    }
    write_json_new(RECEIPT, receipt)
    claim = read_json(CLAIM)
    claim.update(
        {
            "status": "v19_1_holdout_claimed_complete",
            "holdout_execution_complete": True,
            "completed_at_utc": receipt["completed_at_utc"],
            "result_receipt_sha256": sha256_file(RECEIPT),
        }
    )
    write_json_atomic(CLAIM, claim)


def execute(*, resume: bool, timeout: int) -> dict[str, Any]:
    ensure_claim(resume=resume)
    assignments_payload = prepare_assignments()
    assignments = list(assignments_payload["assignments"])
    materialize_baseline_retrieval(assignments, timeout=timeout)
    baseline = score_v18_baseline(build_baseline_tasks(assignments))
    colq = score_colq_all(assignments)
    result = finalize_results(assignments, baseline, colq)
    complete_claim(result)
    print(
        "V19.1 one-shot holdout complete: "
        f"V18={result['baseline_v18']['end_to_end_accuracy']:.3f}, "
        f"V19={result['predecessor_v19']['end_to_end_accuracy']:.3f}, "
        f"V19.1={result['candidate_v19_1']['end_to_end_accuracy']:.3f}, "
        f"gates={result['all_release_gates_passed']}",
        flush=True,
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
        payload = authorize(args.basis)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    execute(resume=args.resume, timeout=args.timeout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
