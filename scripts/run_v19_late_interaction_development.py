"""Score only development-time guarded interventions with late interaction."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ocr_vlm_retrieval.gating.candidate_verification import (  # noqa: E402
    ATTRIBUTE_INSTRUCTION,
    FULL_QUERY_INSTRUCTION,
    build_ocr_evidence,
    load_ocr_lines,
    resolve_requirement_scores,
)
from ocr_vlm_retrieval.gating.attribute_coverage import (  # noqa: E402
    load_attribute_policy,
)
from ocr_vlm_retrieval.gating.condition_decomposition import (  # noqa: E402
    decompose_condition_query,
)
from ocr_vlm_retrieval.gating.contrastive_relations import (  # noqa: E402
    relation_margin,
)
from ocr_vlm_retrieval.gating.v18_contrastive_relations import (  # noqa: E402
    build_v18_relation_counterfactual,
    v18_counterfactual_prompt,
)
from ocr_vlm_retrieval.runtime.cache import write_json_atomic  # noqa: E402

DEFAULT_ASSIGNMENTS = (
    ROOT
    / "outputs/evaluation/v19/selective_intervention"
    / "development_route_assignments.json"
)
DEFAULT_BASELINE_DIR = (
    ROOT
    / "outputs/evaluation/v19/selective_intervention/retrieval/baseline"
)
DEFAULT_GUARDED_DIR = (
    ROOT
    / "outputs/evaluation/v19/selective_intervention/retrieval/guarded"
)
DEFAULT_LIBRARY = ROOT / "outputs/user_library"
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/evaluation/v19/selective_intervention"
    / "development_late_interaction.json"
)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ranking_rows(payload: Mapping[str, Any], top_k: int) -> list[dict[str, Any]]:
    rankings = payload.get("rankings", {})
    rows = rankings.get("quality_hybrid", []) if isinstance(rankings, Mapping) else []
    return [dict(row) for row in rows[:top_k] if row.get("item_id")]


def ocr_text(library_dir: Path, item_id: str) -> str:
    path = library_dir / "ocr/json" / f"{item_id}.json"
    if not path.is_file():
        return ""
    payload = read_json(path)
    return "\n".join(str(value) for value in payload.get("rec_texts", []))[:1200]


def build_tasks(
    route_payload: Mapping[str, Any],
    *,
    baseline_dir: Path,
    guarded_dir: Path,
    library_dir: Path,
    top_k: int,
) -> list[dict[str, Any]]:
    if route_payload.get("split") != "v19_reviewed_development_only":
        raise ValueError("late interaction may read development assignments only")
    tasks: list[dict[str, Any]] = []
    for assignment in route_payload.get("assignments", []):
        if not assignment.get("guarded_route_changed"):
            continue
        query_id = str(assignment["query_id"])
        baseline = read_json(baseline_dir / f"{query_id}_v18_frozen.json")
        guarded = read_json(guarded_dir / f"{query_id}_b21.json")
        candidates: list[dict[str, Any]] = []
        seen: set[str] = set()
        for source, payload in (("baseline", baseline), ("guarded", guarded)):
            for rank, row in enumerate(ranking_rows(payload, top_k), start=1):
                item_id = str(row["item_id"])
                if item_id in seen:
                    continue
                seen.add(item_id)
                image_path = ROOT / str(row["source_path"])
                if not image_path.is_file():
                    raise FileNotFoundError(image_path)
                candidates.append(
                    {
                        "item_id": item_id,
                        "image_path": str(image_path.resolve()),
                        "ocr_text": ocr_text(library_dir, item_id),
                        "first_source": source,
                        "source_rank": rank,
                    }
                )
        if not candidates:
            raise ValueError(f"no candidates for {query_id}")
        tasks.append(
            {
                "query_id": query_id,
                "query": str(assignment["query"]),
                "query_role": assignment.get("query_role"),
                "gold_answerable": assignment.get("gold_answerable"),
                "source_item_id": assignment.get("source_item_id"),
                "neighbor_item_id": assignment.get("neighbor_item_id"),
                "content_stratum": assignment.get("content_stratum"),
                "legacy_route": assignment.get("legacy_route"),
                "guarded_route": assignment.get("guarded_route"),
                "candidates": candidates,
            }
        )
    if not tasks:
        raise ValueError("no guarded interventions require verification")
    return tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--guarded-dir", type=Path, default=DEFAULT_GUARDED_DIR)
    parser.add_argument("--library-dir", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument(
        "--model", type=Path, default=ROOT / "models/qwen3-vl-reranker-2b"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--max-pixels", type=int, default=384 * 384)
    parser.add_argument(
        "--policy", type=Path, default=ROOT / "config/v17_attribute_coverage.json"
    )
    parser.add_argument(
        "--ocr-root", type=Path, default=ROOT / "outputs/user_library/ocr/json"
    )
    parser.add_argument("--ocr-min-confidence", type=float, default=0.35)
    parser.add_argument("--ocr-fuzzy-threshold", type=float, default=0.88)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("top-k must be positive")
    assignments_path = args.assignments.resolve()
    route_payload = read_json(assignments_path)
    tasks = build_tasks(
        route_payload,
        baseline_dir=args.baseline_dir.resolve(),
        guarded_dir=args.guarded_dir.resolve(),
        library_dir=args.library_dir.resolve(),
        top_k=args.top_k,
    )
    policy = load_attribute_policy(args.policy.resolve())
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("late interaction requires CUDA")
    official_repo = ROOT / "third_party/Qwen3-VL-Embedding"
    if str(official_repo) not in sys.path:
        sys.path.insert(0, str(official_repo))
    from src.models.qwen3_vl_reranker import Qwen3VLReranker

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = Qwen3VLReranker(
        model_name_or_path=str(args.model.resolve()),
        max_length=args.max_length,
        min_pixels=32 * 32 * 4,
        max_pixels=args.max_pixels,
        dtype=torch.float16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    load_seconds = time.perf_counter() - started
    results: list[dict[str, Any]] = []
    for index, task in enumerate(tasks, start=1):
        print(f"[{index:02d}/{len(tasks)}] {task['query_id']}", flush=True)
        documents = []
        for candidate in task["candidates"]:
            document: dict[str, Any] = {"image": candidate["image_path"]}
            if candidate["ocr_text"]:
                document["text"] = candidate["ocr_text"]
            documents.append(document)
        query_started = time.perf_counter()
        plan = decompose_condition_query(str(task["query"]), policy)
        scores = model.process(
            {
                "instruction": FULL_QUERY_INSTRUCTION,
                "query": {"text": task["query"]},
                "documents": documents,
            }
        )
        scores_by_requirement = {
            requirement.requirement_id: model.process(
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
            counterfactual = build_v18_relation_counterfactual(requirement.value)
            if counterfactual is None:
                continue
            negative_prompt = v18_counterfactual_prompt(counterfactual)
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
            }
        scored: list[dict[str, Any]] = []
        for candidate_index, candidate in enumerate(task["candidates"]):
            model_requirement_scores = {
                requirement_id: round(float(values[candidate_index]), 8)
                for requirement_id, values in scores_by_requirement.items()
            }
            ocr_lines = load_ocr_lines(
                args.ocr_root.resolve() / f"{candidate['item_id']}.json",
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
            contrastive_evidence: list[dict[str, Any]] = []
            for requirement_id, row in counterfactual_rows.items():
                positive_score = float(model_requirement_scores[requirement_id])
                negative_score = float(row["negative_scores"][candidate_index])
                contrastive_evidence.append(
                    {
                        "requirement_id": requirement_id,
                        "positive_value": row["positive_value"],
                        "negative_value": row["negative_value"],
                        "positive_score": round(positive_score, 8),
                        "negative_score": round(negative_score, 8),
                        "margin": round(
                            relation_margin(positive_score, negative_score), 8
                        ),
                    }
                )
            scored.append(
                {
                    **candidate,
                    "full_query_score": round(float(scores[candidate_index]), 8),
                    "requirement_scores": model_requirement_scores,
                    "resolved_requirement_scores": resolved_scores,
                    "requirement_score_sources": score_sources,
                    "ocr_evidence": ocr_evidence,
                    "contrastive_relation_evidence": contrastive_evidence,
                }
            )
        scored.sort(key=lambda row: (-row["full_query_score"], row["source_rank"]))
        results.append(
            {
                **{key: value for key, value in task.items() if key != "candidates"},
                "attribute_plan": plan.to_dict(),
                "elapsed_seconds": round(time.perf_counter() - query_started, 3),
                "candidates": scored,
            }
        )
        write_json_atomic(
            args.output.resolve(),
            {
                "schema_version": 1,
                "status": "partial" if index < len(tasks) else "complete",
                "study_id": "v19-selective-intervention-late-interaction",
                "split": "development_only",
                "eligible_for_final_claim": False,
                "assignments_sha256": file_sha256(assignments_path),
                "query_count": len(tasks),
                "completed_query_count": len(results),
                "top_k_per_route": args.top_k,
                "model": "Qwen3-VL-Reranker-2B",
                "load_seconds": round(load_seconds, 3),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "peak_reserved_gib": round(
                    torch.cuda.max_memory_reserved() / 1024**3, 3
                ),
                "results": results,
            },
        )
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
