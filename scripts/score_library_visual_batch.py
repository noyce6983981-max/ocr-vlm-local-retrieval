"""Batch-score reviewed library queries with Qwen3-VL visual retrieval."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_REPO = PROJECT_ROOT / "third_party/Qwen3-VL-Embedding"
PACKAGE_ROOT = PROJECT_ROOT / "src"
if str(OFFICIAL_REPO) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_REPO))

from src.models.qwen3_vl_embedding import Qwen3VLEmbedder  # noqa: E402

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from ocr_vlm_retrieval.gating.attribute_coverage import (  # noqa: E402
    VISUAL_REQUIREMENT_KINDS,
    decompose_visual_query,
    load_attribute_policy,
)
from scripts.live_search import (  # noqa: E402
    library_revision,
    query_key,
    required_search_branches,
    resolve_search_intent,
    write_json_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path(
            "data/evaluation/public_dataset_1500_retrieval_queries_formal.csv"
        ),
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("outputs/user_library/visual_index"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/qwen3-vl-embedding-2b"),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=("train", "validation"),
    )
    parser.add_argument(
        "--required-review-status",
        default="已确认",
        help="Exact review_status required for every selected query.",
    )
    parser.add_argument(
        "--product-required-only",
        action="store_true",
        help="Encode only queries whose product route runs the visual branch.",
    )
    parser.add_argument(
        "--live-cache-dir",
        type=Path,
        default=None,
        help="Also materialize product-compatible visual caches.",
    )
    parser.add_argument(
        "--attribute-policy",
        type=Path,
        default=None,
        help=(
            "Optionally encode every compositional attribute requirement in "
            "the same model load and write V17-compatible visual caches."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/evaluation/library_retrieval/dev_visual_scores.npz"),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_queries(
    path: Path,
    splits: set[str],
    required_review_status: str = "已确认",
) -> list[dict[str, str]]:
    if path.suffix.lower() == ".jsonl":
        rows = [dict(row) for row in read_jsonl(path)]
    else:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("split") in splits]
    if not rows:
        raise ValueError(f"No queries found for splits: {sorted(splits)}")
    if any(row.get("review_status") != required_review_status for row in rows):
        raise ValueError(
            f"All scored queries must have review_status={required_review_status!r}."
        )
    return rows


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3-VL query encoding requires CUDA.")

    queries = read_queries(
        project_path(args.queries),
        set(args.splits),
        args.required_review_status,
    )
    product_intents = {
        row["query_id"]: resolve_search_intent(
            "quality_hybrid", " ".join(row["query"].split())
        )
        for row in queries
    }
    attribute_policy = None
    attribute_policy_sha256 = None
    attribute_plans: dict[str, Any] = {}
    if args.attribute_policy is not None:
        attribute_policy_path = project_path(args.attribute_policy)
        attribute_policy = load_attribute_policy(attribute_policy_path)
        attribute_policy_sha256 = hashlib.sha256(
            attribute_policy_path.read_bytes()
        ).hexdigest()
        attribute_plans = {
            row["query_id"]: decompose_visual_query(row["query"], attribute_policy)
            for row in queries
        }
    if args.product_required_only:
        queries = [
            row
            for row in queries
            if required_search_branches(
                "quality_hybrid",
                product_intents[row["query_id"]][0],
                product_intents[row["query_id"]][1],
            )["visual"]
        ]
        if not queries:
            raise ValueError("No selected query requires the visual branch.")
    index_dir = project_path(args.index_dir)
    metadata = read_jsonl(index_dir / "metadata.jsonl")
    image_vectors = np.load(index_dir / "embeddings.npy").astype(np.float32)
    if len(metadata) != len(image_vectors):
        raise ValueError("Visual embedding and metadata counts differ.")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print(f"Encoding {len(queries)} confirmed queries with Qwen3-VL-Embedding-2B ...")
    model = Qwen3VLEmbedder(
        model_name_or_path=str(project_path(args.model)),
        max_length=512,
        min_pixels=32 * 32 * 4,
        max_pixels=512 * 512,
        dtype=torch.float16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    started = time.perf_counter()
    model_inputs: list[dict[str, str]] = []
    request_keys: list[tuple[str, str | None]] = []
    for row in queries:
        query_id = row["query_id"]
        encoded_query = (
            product_intents[query_id][2] if args.product_required_only else row["query"]
        )
        model_inputs.append(
            {
                "text": encoded_query,
                "instruction": (
                    "Retrieve the image that best matches the user's "
                    "Chinese natural-language query."
                ),
            }
        )
        request_keys.append((query_id, None))
        plan = attribute_plans.get(query_id)
        if plan is None or not plan.compositional:
            continue
        for requirement in plan.requirements:
            if requirement.kind not in VISUAL_REQUIREMENT_KINDS:
                continue
            model_inputs.append(
                {
                    "text": requirement.prompt,
                    "instruction": (
                        "Score whether the image independently satisfies this "
                        "mandatory visual requirement. Do not reward a different "
                        "attribute from the original compound query."
                    ),
                }
            )
            request_keys.append((query_id, requirement.requirement_id))

    vectors: list[np.ndarray] = []
    for start in range(0, len(model_inputs), args.batch_size):
        embeddings = model.process(model_inputs[start : start + args.batch_size])
        vectors.append(embeddings.detach().float().cpu().numpy())
    torch.cuda.synchronize()
    request_matrix = np.ascontiguousarray(np.vstack(vectors), dtype=np.float32)
    request_scores = request_matrix @ image_vectors.T
    global_scores_by_id: dict[str, np.ndarray] = {}
    attribute_scores_by_id: dict[str, dict[str, np.ndarray]] = {}
    for request_index, (query_id, requirement_id) in enumerate(request_keys):
        if requirement_id is None:
            global_scores_by_id[query_id] = request_scores[request_index]
        else:
            attribute_scores_by_id.setdefault(query_id, {})[requirement_id] = (
                request_scores[request_index]
            )
    scores = np.vstack([global_scores_by_id[row["query_id"]] for row in queries])

    output_path = project_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        visual_scores=scores,
        query_ids=np.asarray([row["query_id"] for row in queries]),
        item_ids=np.asarray([row["item_id"] for row in metadata]),
        splits=np.asarray([row["split"] for row in queries]),
    )
    if args.live_cache_dir is not None:
        cache_dir = project_path(args.live_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        revision = library_revision(project_path(args.index_dir).parent)
        elapsed_per_query = round((time.perf_counter() - started) / len(queries), 3)
        for index_value, row in enumerate(queries):
            query = " ".join(row["query"].split())
            encoded_query = product_intents[row["query_id"]][2]
            key = query_key(query, revision)
            write_json_atomic(
                cache_dir / f"{key}_visual.json",
                {
                    "query": query,
                    "encoded_query": encoded_query,
                    "branch": "visual",
                    "library_revision": revision,
                    "item_ids": [row["item_id"] for row in metadata],
                    "scores": [round(float(score), 8) for score in scores[index_value]],
                    "attribute_policy_sha256": attribute_policy_sha256,
                    "attribute_plan": (
                        attribute_plans[row["query_id"]].to_dict()
                        if row["query_id"] in attribute_plans
                        else None
                    ),
                    "scores_by_requirement": {
                        requirement_id: [
                            round(float(value), 8) for value in requirement_scores
                        ]
                        for requirement_id, requirement_scores in (
                            attribute_scores_by_id.get(row["query_id"], {}).items()
                        )
                    },
                    "elapsed_seconds": elapsed_per_query,
                },
            )
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "splits": list(args.splits),
                "query_count": len(queries),
                "item_count": len(metadata),
                "encode_seconds": round(time.perf_counter() - started, 3),
                "peak_allocated_gib": round(
                    torch.cuda.max_memory_allocated() / 1024**3, 3
                ),
                "peak_reserved_gib": round(
                    torch.cuda.max_memory_reserved() / 1024**3, 3
                ),
                "product_required_only": args.product_required_only,
                "attribute_policy": (
                    project_path(args.attribute_policy)
                    .relative_to(PROJECT_ROOT)
                    .as_posix()
                    if args.attribute_policy is not None
                    else None
                ),
                "encoded_request_count": len(model_inputs),
                "live_cache_dir": (
                    project_path(args.live_cache_dir)
                    .relative_to(PROJECT_ROOT)
                    .as_posix()
                    if args.live_cache_dir is not None
                    else None
                ),
                "output": output_path.relative_to(PROJECT_ROOT).as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved {len(queries)} x {len(metadata)} visual scores: {output_path}")


if __name__ == "__main__":
    main()
