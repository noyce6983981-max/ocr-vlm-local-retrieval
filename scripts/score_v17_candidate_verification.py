"""Score V17 ranked candidates with candidate-level attribute verification.

This calibration-only scorer loads Qwen3-VL-Reranker once, verifies the full
query and every independently parsed necessary condition, and checkpoints
after each query.  It deliberately does not read relevance judgments, so its
scores cannot leak audited labels into inference.
"""

from __future__ import annotations

import argparse
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
            "candidate_attribute_verification.json"
        ),
    )
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-pixels", type=int, default=384 * 384)
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
    ranking_path = project_path(args.ranking)
    policy_path = project_path(args.policy)
    model_path = project_path(args.model)
    output_path = project_path(args.output)
    tasks = build_verification_tasks(
        read_jsonl(packets_path),
        read_jsonl(ranking_path),
        top_k=args.top_k,
    )
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("limit must be positive")
        tasks = tasks[: args.limit]
    policy = load_attribute_policy(policy_path)

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
    results: list[dict[str, Any]] = []
    full_instruction = (
        "Verify exact full-query satisfaction in this one candidate image. "
        "Every necessary object, visual attribute, spatial or directional "
        "relation, and requested text condition must hold in the same image. "
        "A partial or near-neighbor match is negative."
    )
    attribute_instruction = (
        "Verify only the stated necessary visual condition in this candidate. "
        "For directional relations and attribute bindings, require the exact "
        "direction and the same referenced object; partial evidence is negative."
    )
    for index, task in enumerate(tasks, start=1):
        print(f"[{index:02d}/{len(tasks)}] {task['query_id']}", flush=True)
        plan = decompose_visual_query(task["query"], policy)
        documents = [
            {"image": str(project_path(Path(row["image_path"])).resolve())}
            for row in task["candidates"]
        ]
        query_started = time.perf_counter()
        full_scores = model.process(
            {
                "instruction": full_instruction,
                "query": {"text": task["query"]},
                "documents": documents,
            }
        )
        scores_by_requirement: dict[str, list[float]] = {}
        for requirement in plan.requirements:
            scores_by_requirement[requirement.requirement_id] = model.process(
                {
                    "instruction": attribute_instruction,
                    "query": {"text": requirement.prompt},
                    "documents": documents,
                }
            )
        candidate_rows = []
        for candidate_index, candidate in enumerate(task["candidates"]):
            candidate_rows.append(
                {
                    **candidate,
                    "full_query_score": round(float(full_scores[candidate_index]), 8),
                    "requirement_scores": {
                        requirement_id: round(float(values[candidate_index]), 8)
                        for requirement_id, values in scores_by_requirement.items()
                    },
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
                "status": "partial" if index < len(tasks) else "complete",
                "scope": "v17_calibration_ranked_candidates_only",
                "judgments_read": False,
                "query_count": len(tasks),
                "completed_query_count": len(results),
                "top_k": args.top_k,
                "model": "Qwen3-VL-Reranker-2B",
                "model_path": model_path.relative_to(PROJECT_ROOT).as_posix(),
                "policy_sha256": file_sha256(policy_path),
                "packets_sha256": file_sha256(packets_path),
                "ranking_sha256": file_sha256(ranking_path),
                "load_seconds": round(load_seconds, 3),
                "elapsed_seconds": round(time.perf_counter() - run_started, 3),
                "peak_reserved_gib": round(
                    torch.cuda.max_memory_reserved() / 1024**3, 3
                ),
                "results": results,
            },
        )
    print(output_path)


if __name__ == "__main__":
    main()
