"""Score V17 ranked candidates with candidate-level attribute verification.

This calibration-only scorer loads Qwen3-VL-Reranker once, verifies the full
query and every independently parsed necessary condition, and checkpoints
after each query.  It deliberately does not read relevance judgments, so its
scores cannot leak audited labels into inference.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_REPO = PROJECT_ROOT / "third_party/Qwen3-VL-Embedding"
PACKAGE_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

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


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            rows.append(payload)
    return rows


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def resume_results(
    output_path: Path,
    *,
    run_identity: Mapping[str, Any],
    ordered_query_ids: list[str],
) -> tuple[list[dict[str, Any]], float, float]:
    """Resume only an exact schema-v2 prefix from the same calibration run."""

    if not output_path.is_file():
        return [], 0.0, 0.0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0)) != 2:
        raise ValueError("Existing verification output is not resumable schema v2")
    for key, expected in run_identity.items():
        if payload.get(key) != expected:
            raise ValueError(f"Existing verification output mismatches {key}")
    results = payload.get("results", [])
    if not isinstance(results, list):
        raise ValueError("Existing verification results must be a list")
    completed_ids = [str(row.get("query_id", "")) for row in results]
    if completed_ids != ordered_query_ids[: len(completed_ids)]:
        raise ValueError("Existing verification results are not an ordered prefix")
    return (
        [dict(row) for row in results],
        float(payload.get("elapsed_seconds", 0.0)),
        float(payload.get("peak_reserved_gib", 0.0)),
    )


def keyed_rows(
    rows: Iterable[Mapping[str, Any]], *, label: str
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for source in rows:
        query_id = str(source.get("query_id", "")).strip()
        if not query_id:
            raise ValueError(f"Every {label} row needs query_id")
        if query_id in result:
            raise ValueError(f"Duplicate {label} row for {query_id}")
        result[query_id] = dict(source)
    return result


def merge_packet_extensions(
    packets: Iterable[Mapping[str, Any]],
    extensions: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Append audited calibration candidates without mutating the base pool."""

    base_rows = [copy.deepcopy(dict(row)) for row in packets]
    by_id = keyed_rows(base_rows, label="packet")
    for extension in extensions:
        query_id = str(extension.get("query_id", "")).strip()
        if query_id not in by_id:
            raise ValueError(f"Packet extension has unknown query_id {query_id!r}")
        if extension.get("split") != "calibration":
            raise ValueError("Packet extensions must be calibration-only")
        base = by_id[query_id]
        if str(extension.get("query", "")) != str(base.get("query", "")):
            raise ValueError(f"Packet extension query mismatch for {query_id}")
        candidates = base.get("candidates", [])
        extension_candidates = extension.get("candidates", [])
        if not isinstance(candidates, list) or not isinstance(
            extension_candidates, list
        ):
            raise ValueError("Packet candidates must be lists")
        known = {str(row.get("item_id", "")) for row in candidates}
        for candidate in extension_candidates:
            item_id = str(candidate.get("item_id", "")).strip()
            if not item_id or item_id in known:
                raise ValueError(
                    f"Packet extension candidate {item_id!r} is invalid or duplicated"
                )
            candidates.append(dict(candidate))
            known.add(item_id)
    return base_rows


def build_verification_tasks(
    packets: Iterable[Mapping[str, Any]],
    ranking_rows: Iterable[Mapping[str, Any]],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    """Join a blinded packet with a ranking without reading any labels."""

    if top_k < 1:
        raise ValueError("top_k must be positive")
    packet_by_id = keyed_rows(packets, label="packet")
    ranking_by_id = keyed_rows(ranking_rows, label="ranking")
    if set(packet_by_id) != set(ranking_by_id):
        raise ValueError("Packet and ranking query IDs do not match")
    tasks: list[dict[str, Any]] = []
    for query_id in sorted(packet_by_id):
        packet = packet_by_id[query_id]
        metadata_by_id = {
            str(row["item_id"]): dict(row.get("review_metadata", {}))
            for row in packet.get("candidates", [])
        }
        ranking = ranking_by_id[query_id].get("ranking", [])
        if not isinstance(ranking, list) or len(ranking) < top_k:
            raise ValueError(f"Ranking {query_id} has fewer than {top_k} candidates")
        candidates: list[dict[str, Any]] = []
        for row in ranking[:top_k]:
            item_id = str(row.get("item_id", "")).strip()
            if item_id not in metadata_by_id:
                raise ValueError(
                    f"Ranked candidate {item_id!r} for {query_id} is outside "
                    "the blinded review pool"
                )
            metadata = metadata_by_id[item_id]
            image_path = str(metadata.get("image_path", "")).strip()
            if not image_path:
                raise ValueError(f"Candidate {item_id} has no image_path")
            candidates.append(
                {
                    "item_id": item_id,
                    "image_path": image_path,
                    "ranking_score": float(row.get("score", 0.0)),
                }
            )
        tasks.append(
            {
                "query_id": query_id,
                "query": str(packet.get("query", "")).strip(),
                "group_id": str(packet.get("group_id", "")).strip(),
                "candidates": candidates,
            }
        )
    return tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--packets",
        type=Path,
        default=Path(
            "data/evaluation/v17/human_study/calibration/review_packets.jsonl"
        ),
    )
    parser.add_argument(
        "--ranking",
        type=Path,
        default=Path(
            "outputs/evaluation/v17/calibration/parser_v3/runs/v17_quality_hybrid.jsonl"
        ),
    )
    parser.add_argument(
        "--packet-extension",
        type=Path,
        default=Path(
            "data/evaluation/v17/human_study/calibration/"
            "top5_extension_review_packets.jsonl"
        ),
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("config/v17_attribute_coverage.json"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/qwen3-vl-reranker-2b"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/evaluation/v17/calibration/parser_v3/"
            "candidate_attribute_verification_top5_contrastive.json"
        ),
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-pixels", type=int, default=384 * 384)
    parser.add_argument(
        "--ocr-root",
        type=Path,
        default=Path("outputs/user_library/ocr/json"),
    )
    parser.add_argument("--ocr-min-confidence", type=float, default=0.35)
    parser.add_argument("--ocr-fuzzy-threshold", type=float, default=0.88)
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Ignore a partial schema-v2 checkpoint and start this output again.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Optional smoke-test query limit; omit for the complete "
            "calibration run."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    packets_path = project_path(args.packets)
    packet_extension_path = project_path(args.packet_extension)
    ranking_path = project_path(args.ranking)
    policy_path = project_path(args.policy)
    model_path = project_path(args.model)
    output_path = project_path(args.output)
    ocr_root = project_path(args.ocr_root)
    tasks = build_verification_tasks(
        merge_packet_extensions(
            read_jsonl(packets_path),
            read_jsonl(packet_extension_path),
        ),
        read_jsonl(ranking_path),
        top_k=args.top_k,
    )
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("limit must be positive")
        tasks = tasks[: args.limit]
    policy = load_attribute_policy(policy_path)

    run_identity = {
        "scope": "v17_calibration_ranked_candidates_only",
        "judgments_read": False,
        "query_count": len(tasks),
        "top_k": args.top_k,
        "model": "Qwen3-VL-Reranker-2B",
        "model_path": model_path.relative_to(PROJECT_ROOT).as_posix(),
        "verification_prompts": verification_prompt_payload(),
        "policy_sha256": file_sha256(policy_path),
        "packets_sha256": file_sha256(packets_path),
        "packet_extension_sha256": file_sha256(packet_extension_path),
        "ranking_sha256": file_sha256(ranking_path),
        "ocr_policy": {
            "root": args.ocr_root.as_posix(),
            "minimum_confidence": args.ocr_min_confidence,
            "fuzzy_threshold": args.ocr_fuzzy_threshold,
            "precedence": ["exact", "fuzzy", "visual_verifier_fallback"],
        },
    }
    if args.restart:
        results: list[dict[str, Any]] = []
        previous_elapsed = 0.0
        previous_peak_gib = 0.0
    else:
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
        raise RuntimeError("Qwen3-VL candidate verification requires CUDA")
    if str(OFFICIAL_REPO) not in sys.path:
        sys.path.insert(0, str(OFFICIAL_REPO))
    from src.models.qwen3_vl_reranker import Qwen3VLReranker

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    run_started = time.perf_counter()
    model = Qwen3VLReranker(
        model_name_or_path=str(model_path),
        max_length=args.max_length,
        min_pixels=32 * 32 * 4,
        max_pixels=args.max_pixels,
        dtype=torch.float16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    load_seconds = time.perf_counter() - run_started
    for index, task in enumerate(pending_tasks, start=len(results) + 1):
        print(f"[{index:02d}/{len(tasks)}] {task['query_id']}", flush=True)
        plan = decompose_visual_query(task["query"], policy)
        documents = [
            {"image": str(project_path(Path(row["image_path"])).resolve())}
            for row in task["candidates"]
        ]
        query_started = time.perf_counter()
        full_scores = model.process(
            {
                "instruction": FULL_QUERY_INSTRUCTION,
                "query": {"text": task["query"]},
                "documents": documents,
            }
        )
        scores_by_requirement: dict[str, list[float]] = {}
        for requirement in plan.requirements:
            scores_by_requirement[requirement.requirement_id] = model.process(
                {
                    "instruction": ATTRIBUTE_INSTRUCTION,
                    "query": {"text": requirement.prompt},
                    "documents": documents,
                }
            )
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
                "negative_scores": model.process(
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
            ocr_path = ocr_root / f"{candidate['item_id']}.json"
            ocr_lines = load_ocr_lines(
                ocr_path,
                minimum_confidence=args.ocr_min_confidence,
            )
            ocr_evidence = build_ocr_evidence(
                plan,
                ocr_lines,
                fuzzy_threshold=args.ocr_fuzzy_threshold,
            )
            resolved_scores, score_sources = resolve_requirement_scores(
                model_requirement_scores,
                ocr_evidence,
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
