"""Score frozen V18 L1 on V19.1 machine-draft development only."""

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
    FULL_QUERY_INSTRUCTION,
)
from ocr_vlm_retrieval.runtime.cache import write_json_atomic  # noqa: E402
from ocr_vlm_retrieval.runtime.late_interaction import (  # noqa: E402
    exclusive_process_lock,
)
from scripts.run_v19_downstream_retrieval_pilot import (  # noqa: E402
    ranking_ids,
    read_json,
)
from scripts.score_v19_v18_l1_development import (  # noqa: E402
    l1_result,
    manifest_index,
    summarize,
)

EVALUATION_ROOT = ROOT / "outputs/evaluation/v19_1/condition_completeness"
DEFAULT_ASSIGNMENTS = EVALUATION_ROOT / "development_assignments_machine.json"
DEFAULT_RETRIEVAL_DIR = EVALUATION_ROOT / "retrieval/development_v18_frozen"
DEFAULT_LIBRARY = ROOT / "outputs/user_library"
DEFAULT_OUTPUT = EVALUATION_ROOT / "development_v18_l1_top3_machine.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--retrieval-dir", type=Path, default=DEFAULT_RETRIEVAL_DIR)
    parser.add_argument("--library-dir", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument(
        "--model", type=Path, default=ROOT / "models/qwen3-vl-reranker-2b"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.63)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-pixels", type=int, default=384 * 384)
    return parser.parse_args()


def _source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_tasks(
    payload: Mapping[str, Any],
    *,
    retrieval_dir: Path,
    library_dir: Path,
    top_k: int,
) -> list[dict[str, Any]]:
    if payload.get("split") != "v19_1_machine_draft_development_only":
        raise ValueError("V18 L1 may score V19.1 machine-draft development only")
    manifest = manifest_index(library_dir)
    tasks: list[dict[str, Any]] = []
    for assignment in payload.get("assignments", []):
        query_id = str(assignment["query_id"])
        retrieval = read_json(retrieval_dir / f"{query_id}_v18_frozen.json")
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
        tasks.append({**dict(assignment), "candidates": candidates})
    if len(tasks) != 48:
        raise ValueError(f"expected 48 development tasks, got {len(tasks)}")
    return tasks


def run(args: argparse.Namespace) -> int:
    if args.top_k != 3 or args.threshold != 0.63:
        raise ValueError("frozen V18 L1 requires top_k=3 and threshold=0.63")
    assignment_payload = read_json(args.assignments)
    source_sha256 = _source_sha256(args.assignments)
    tasks = build_tasks(
        assignment_payload,
        retrieval_dir=args.retrieval_dir,
        library_dir=args.library_dir,
        top_k=args.top_k,
    )
    previous: dict[str, Any] = {}
    if args.output.is_file():
        previous = read_json(args.output)
        if previous.get("source_assignment_sha256") != source_sha256:
            raise ValueError("existing checkpoint references different assignments")
    existing = {str(row["query_id"]): row for row in previous.get("results", [])}
    pending = [task for task in tasks if str(task["query_id"]) not in existing]
    if not pending:
        print(json.dumps(summarize(list(existing.values())), indent=2))
        return 0

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("V18 L1 reproduction requires CUDA")
    official_repo = ROOT / "third_party/Qwen3-VL-Embedding"
    if str(official_repo) not in sys.path:
        sys.path.insert(0, str(official_repo))
    from src.models.qwen3_vl_reranker import Qwen3VLReranker  # type: ignore[import-not-found]

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
    inference_started = time.perf_counter()
    for index, task in enumerate(pending, start=1):
        print(f"[{index:02d}/{len(pending)}] {task['query_id']}", flush=True)
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
        existing[str(task["query_id"])] = l1_result(
            task, scores, threshold=args.threshold
        )
        ordered = [
            existing[str(row["query_id"])]
            for row in tasks
            if str(row["query_id"]) in existing
        ]
        write_json_atomic(
            args.output,
            {
                "status": (
                    "machine_draft_development_diagnostic_only"
                    if len(ordered) == len(tasks)
                    else "partial_machine_draft_development_diagnostic_only"
                ),
                "split": "v19_1_machine_draft_development_only",
                "eligible_for_promotion": False,
                "holdout_opened": False,
                "method": "frozen_v18_L1_max_verifier_score",
                "source_assignment_sha256": source_sha256,
                "parameters": {"top_k": args.top_k, "threshold": args.threshold},
                "summary": summarize(ordered),
                "load_seconds": round(load_seconds, 3),
                "inference_seconds": round(time.perf_counter() - inference_started, 3),
                "peak_reserved_gib": round(
                    torch.cuda.max_memory_reserved() / (1024**3), 3
                ),
                "results": ordered,
            },
        )
    print(json.dumps(summarize(list(existing.values())), indent=2))
    return 0


def main() -> int:
    args = parse_args()
    lock_path = args.library_dir / ".v19_1_development_gpu_job.lock"
    with exclusive_process_lock(lock_path):
        return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
