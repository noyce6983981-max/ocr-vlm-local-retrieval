"""Score the frozen V17 holdout Top-K without reading human judgments."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from ocr_vlm_retrieval.evaluation.protocol_lock import (
    file_sha256,
    validate_method_lock,
)
from ocr_vlm_retrieval.gating.attribute_coverage import (
    decompose_visual_query,
    load_attribute_policy,
)
from ocr_vlm_retrieval.gating.candidate_verification import (
    ATTRIBUTE_INSTRUCTION,
    FULL_QUERY_INSTRUCTION,
    build_ocr_evidence,
    load_ocr_lines,
    resolve_requirement_scores,
    verification_prompt_payload,
)
from ocr_vlm_retrieval.gating.contrastive_relations import (
    build_relation_counterfactual,
    counterfactual_prompt,
    relation_margin,
)
from scripts.score_v17_candidate_verification import (
    resume_results,
    write_json_atomic,
)

VERIFICATION_SCOPE = "v17_holdout_ranked_candidates_only"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def keyed_rows(
    rows: Iterable[Mapping[str, Any]], *, label: str
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for source in rows:
        query_id = str(source.get("query_id", "")).strip()
        if not query_id or query_id in result:
            raise ValueError(f"Invalid or duplicate {label} query_id: {query_id!r}")
        result[query_id] = dict(source)
    return result


def build_holdout_verification_tasks(
    packets: Iterable[Mapping[str, Any]],
    ranking_rows: Iterable[Mapping[str, Any]],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    packet_by_id = keyed_rows(packets, label="packet")
    ranking_by_id = keyed_rows(ranking_rows, label="ranking")
    if set(packet_by_id) != set(ranking_by_id):
        raise ValueError("Packet and ranking query IDs do not match")

    tasks: list[dict[str, Any]] = []
    for query_id in sorted(packet_by_id):
        packet = packet_by_id[query_id]
        metadata_by_id: dict[str, dict[str, Any]] = {}
        for source in packet.get("candidates", []):
            item_id = str(source.get("item_id", "")).strip()
            metadata = source.get("review_asset", source.get("review_metadata", {}))
            if not item_id or not isinstance(metadata, Mapping):
                raise ValueError(f"Invalid review candidate for {query_id}")
            metadata_by_id[item_id] = dict(metadata)
        ranking = ranking_by_id[query_id].get("ranking", [])
        if not isinstance(ranking, list) or len(ranking) < top_k:
            raise ValueError(f"Ranking {query_id} has fewer than {top_k} candidates")
        candidates: list[dict[str, Any]] = []
        for rank, source in enumerate(ranking[:top_k], start=1):
            item_id = str(source.get("item_id", "")).strip()
            if item_id not in metadata_by_id:
                raise ValueError(
                    f"Ranked candidate {item_id!r} for {query_id} is outside "
                    "the blinded review pool"
                )
            image_path = str(metadata_by_id[item_id].get("image_path", "")).strip()
            if not image_path:
                raise ValueError(f"Candidate {item_id} has no image_path")
            candidates.append(
                {
                    "item_id": item_id,
                    "image_path": image_path,
                    "retrieval_rank": rank,
                    "ranking_score": float(source.get("score", 0.0)),
                }
            )
        group_id = str(packet.get("group_id", "")).strip()
        query = str(packet.get("query", "")).strip()
        if not group_id or not query:
            raise ValueError(f"Packet metadata is incomplete for {query_id}")
        tasks.append(
            {
                "query_id": query_id,
                "query": query,
                "group_id": group_id,
                "candidates": candidates,
            }
        )
    return tasks


def validate_retrieval_lineage(
    receipt: Mapping[str, Any],
    *,
    current_lock_sha256: str,
    prior_lock_sha256: str,
    receipt_sha256: str,
    expected_receipt_sha256: str,
    ranking_sha256: str,
    expected_ranking_sha256: str,
) -> None:
    if receipt.get("executed_split") != "holdout":
        raise ValueError("Retrieval receipt is not for the holdout split")
    if receipt.get("judgments_read") is not False:
        raise ValueError("Retrieval receipt must certify judgments_read=false")
    if receipt.get("method_lock_sha256") not in {
        current_lock_sha256,
        prior_lock_sha256,
    }:
        raise ValueError("Retrieval receipt is not in the method-lock lineage")
    if receipt_sha256 != expected_receipt_sha256:
        raise ValueError("Retrieval receipt hash does not match the method lock")
    run = receipt.get("run_files", {}).get("v17_quality_hybrid", {})
    if not isinstance(run, Mapping) or run.get("sha256") != ranking_sha256:
        raise ValueError("V17 ranking hash does not match the retrieval receipt")
    if ranking_sha256 != expected_ranking_sha256:
        raise ValueError("V17 ranking hash does not match the method lock")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime_root = args.runtime_root.resolve()
    lock_path = PROJECT_ROOT / "config/v17_method_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    validation = validate_method_lock(
        lock,
        project_root=PROJECT_ROOT,
        runtime_root=runtime_root,
        require_clean=True,
    )
    if not validation["valid"]:
        raise ValueError(f"Method lock is invalid: {validation['errors']}")
    holdout_inputs = lock.get("holdout_inputs", {})
    if not isinstance(holdout_inputs, Mapping):
        raise ValueError("Method lock has no holdout input bindings")

    receipt_path = (
        runtime_root
        / "outputs/evaluation/v17/holdout/retrieval/"
        "holdout_retrieval_receipt.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    ranking_path = (
        runtime_root
        / "outputs/evaluation/v17/holdout/retrieval/runs/"
        "v17_quality_hybrid.jsonl"
    )
    receipt_hash = file_sha256(receipt_path)
    ranking_hash = file_sha256(ranking_path)
    validate_retrieval_lineage(
        receipt,
        current_lock_sha256=file_sha256(lock_path),
        prior_lock_sha256=str(lock.get("prior_lock_sha256", "")),
        receipt_sha256=receipt_hash,
        expected_receipt_sha256=str(
            holdout_inputs.get("retrieval_receipt_sha256", "")
        ),
        ranking_sha256=ranking_hash,
        expected_ranking_sha256=str(
            holdout_inputs.get("v17_candidate_ranking_sha256", "")
        ),
    )

    packets_path = (
        runtime_root
        / "data/evaluation/v17/human_study/holdout/review_packets.jsonl"
    )
    if file_sha256(packets_path) != holdout_inputs.get("review_packets_sha256"):
        raise ValueError("Holdout review packet hash does not match the lock")
    top_k = int(lock["inference"]["top_k"])
    tasks = build_holdout_verification_tasks(
        read_jsonl(packets_path), read_jsonl(ranking_path), top_k=top_k
    )
    if len(tasks) != int(lock["holdout"]["query_count"]):
        raise ValueError("Holdout verification task count does not match the lock")

    policy_path = PROJECT_ROOT / "config/v17_attribute_coverage.json"
    prompt_payload = verification_prompt_payload()
    if prompt_payload != lock.get("verification_prompt_text"):
        raise ValueError("Runtime verification prompts do not match the lock")
    inference = lock["inference"]
    model = lock["model"]
    run_identity = {
        "scope": VERIFICATION_SCOPE,
        "judgments_read": False,
        "query_count": len(tasks),
        "top_k": top_k,
        "model": model["name"],
        "model_path": model["path"],
        "model_weight_manifest_sha256": model["weight_manifest_sha256"],
        "verification_prompts": prompt_payload,
        "policy_sha256": file_sha256(policy_path),
        "packets_sha256": file_sha256(packets_path),
        "ranking_policy_sha256": lock["candidate_ranking_policy_sha256"],
        "ranking_sha256": ranking_hash,
        "retrieval_receipt_sha256": receipt_hash,
        "method_lock_sha256": file_sha256(lock_path),
        "inference": {
            "max_length": int(inference["max_length"]),
            "min_pixels": int(inference["min_pixels"]),
            "max_pixels": int(inference["max_pixels"]),
            "dtype": inference["dtype"],
            "attention": inference["attention"],
            "low_cpu_mem_usage": bool(inference["low_cpu_mem_usage"]),
        },
        "ocr_policy": {
            "root": inference["ocr_root"],
            "minimum_confidence": float(inference["ocr_minimum_confidence"]),
            "fuzzy_threshold": float(inference["ocr_fuzzy_threshold"]),
            "precedence": list(inference["ocr_precedence"]),
        },
    }
    output_path = runtime_root / lock["execution"]["verification"]
    results, previous_elapsed, previous_peak_gib = resume_results(
        output_path,
        run_identity=run_identity,
        ordered_query_ids=[str(task["query_id"]) for task in tasks],
    )
    pending_tasks = tasks[len(results) :]
    if not pending_tasks:
        print(output_path)
        return

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3-VL holdout verification requires CUDA")
    official_repo = runtime_root / "third_party/Qwen3-VL-Embedding"
    if str(official_repo) not in sys.path:
        sys.path.insert(0, str(official_repo))
    from src.models.qwen3_vl_reranker import Qwen3VLReranker

    policy = load_attribute_policy(policy_path)
    model_path = runtime_root / str(model["path"])
    ocr_root = runtime_root / str(inference["ocr_root"])
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    run_started = time.perf_counter()
    verifier = Qwen3VLReranker(
        model_name_or_path=str(model_path),
        max_length=int(inference["max_length"]),
        min_pixels=int(inference["min_pixels"]),
        max_pixels=int(inference["max_pixels"]),
        dtype=torch.float16,
        attn_implementation=str(inference["attention"]),
        low_cpu_mem_usage=bool(inference["low_cpu_mem_usage"]),
    )
    load_seconds = time.perf_counter() - run_started
    for index, task in enumerate(pending_tasks, start=len(results) + 1):
        print(f"[{index:02d}/{len(tasks)}] {task['query_id']}", flush=True)
        plan = decompose_visual_query(task["query"], policy)
        documents = [
            {
                "image": str(
                    (
                        runtime_root / Path(str(row["image_path"]))
                        if not Path(str(row["image_path"])).is_absolute()
                        else Path(str(row["image_path"]))
                    ).resolve()
                )
            }
            for row in task["candidates"]
        ]
        query_started = time.perf_counter()
        full_scores = verifier.process(
            {
                "instruction": FULL_QUERY_INSTRUCTION,
                "query": {"text": task["query"]},
                "documents": documents,
            }
        )
        scores_by_requirement = {
            requirement.requirement_id: verifier.process(
                {
                    "instruction": ATTRIBUTE_INSTRUCTION,
                    "query": {"text": requirement.prompt},
                    "documents": documents,
                }
            )
            for requirement in plan.requirements
        }
        counterfactual_rows: dict[str, dict[str, Any]] = {}
        for requirement in plan.requirements:
            if requirement.kind != "relation":
                continue
            counterfactual = build_relation_counterfactual(requirement.value)
            if counterfactual is None:
                continue
            negative_prompt = counterfactual_prompt(counterfactual)
            counterfactual_rows[requirement.requirement_id] = {
                **counterfactual.to_dict(),
                "negative_prompt": negative_prompt,
                "negative_scores": verifier.process(
                    {
                        "instruction": ATTRIBUTE_INSTRUCTION,
                        "query": {"text": negative_prompt},
                        "documents": documents,
                    }
                ),
                "absolute_threshold": requirement.threshold,
            }
        candidate_rows = []
        for candidate_index, candidate in enumerate(task["candidates"]):
            model_requirement_scores = {
                requirement_id: round(float(values[candidate_index]), 8)
                for requirement_id, values in scores_by_requirement.items()
            }
            ocr_lines = load_ocr_lines(
                ocr_root / f"{candidate['item_id']}.json",
                minimum_confidence=float(inference["ocr_minimum_confidence"]),
            )
            ocr_evidence = build_ocr_evidence(
                plan,
                ocr_lines,
                fuzzy_threshold=float(inference["ocr_fuzzy_threshold"]),
            )
            resolved_scores, score_sources = resolve_requirement_scores(
                model_requirement_scores, ocr_evidence
            )
            contrastive_evidence = []
            for requirement_id, row in counterfactual_rows.items():
                positive_score = float(model_requirement_scores[requirement_id])
                negative_score = float(row["negative_scores"][candidate_index])
                contrastive_evidence.append(
                    {
                        "requirement_id": requirement_id,
                        "positive_value": row["positive_value"],
                        "negative_value": row["negative_value"],
                        "positive_marker": row["positive_marker"],
                        "negative_marker": row["negative_marker"],
                        "negative_prompt": row["negative_prompt"],
                        "positive_score": round(positive_score, 8),
                        "negative_score": round(negative_score, 8),
                        "margin": round(
                            relation_margin(positive_score, negative_score), 8
                        ),
                        "absolute_threshold": row["absolute_threshold"],
                    }
                )
            candidate_rows.append(
                {
                    **candidate,
                    "full_query_score": round(float(full_scores[candidate_index]), 8),
                    "requirement_scores": model_requirement_scores,
                    "resolved_requirement_scores": resolved_scores,
                    "requirement_score_sources": score_sources,
                    "ocr_evidence": ocr_evidence,
                    "contrastive_relation_evidence": contrastive_evidence,
                }
            )
        results.append(
            {
                "query_id": task["query_id"],
                "query": task["query"],
                "group_id": task["group_id"],
                "attribute_plan": plan.to_dict(),
                "candidates": candidate_rows,
                "elapsed_seconds": round(time.perf_counter() - query_started, 3),
            }
        )
        write_json_atomic(
            output_path,
            {
                "schema_version": 2,
                "status": "partial" if index < len(tasks) else "complete",
                **run_identity,
                "completed_query_count": len(results),
                "load_seconds": round(load_seconds, 3),
                "elapsed_seconds": round(
                    previous_elapsed + time.perf_counter() - run_started, 3
                ),
                "peak_reserved_gib": max(
                    previous_peak_gib,
                    round(torch.cuda.max_memory_reserved() / 1024**3, 3),
                ),
                "results": results,
            },
        )
    print(output_path)


if __name__ == "__main__":
    main()
