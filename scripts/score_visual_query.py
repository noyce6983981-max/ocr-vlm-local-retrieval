"""Score one arbitrary query against the Qwen3-VL image index."""

from __future__ import annotations

import argparse
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
if str(OFFICIAL_REPO) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_REPO))

from src.models.qwen3_vl_embedding import Qwen3VLEmbedder  # noqa: E402

PACKAGE_ROOT = PROJECT_ROOT / "src"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from ocr_vlm_retrieval.gating.attribute_coverage import (  # noqa: E402
    VISUAL_REQUIREMENT_KINDS,
    decompose_visual_query,
    load_attribute_policy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument(
        "--encoded-query",
        default=None,
        help="Optional expanded text to encode while preserving cache query.",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("artifacts/visual_index_dataset_v1"),
    )
    parser.add_argument(
        "--extra-index-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/qwen3-vl-embedding-2b"),
    )
    parser.add_argument(
        "--attribute-policy",
        type=Path,
        default=None,
        help=(
            "Optional V17 attribute policy. When supplied, global and "
            "attribute queries are encoded in the same model batch."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument("--library-revision", default="base")
    return parser.parse_args()


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    args = parse_args()
    query = args.query.strip()
    encoded_query = (
        args.encoded_query.strip()
        if args.encoded_query is not None
        else query
    )
    if not query:
        raise ValueError("Query cannot be empty.")
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3-VL query encoding requires CUDA.")

    attribute_policy: dict[str, Any] | None = None
    attribute_policy_sha256: str | None = None
    attribute_plan = None
    visual_requirements = []
    if args.attribute_policy is not None:
        policy_path = project_path(args.attribute_policy)
        attribute_policy = load_attribute_policy(policy_path)
        attribute_policy_sha256 = hashlib.sha256(
            policy_path.read_bytes()
        ).hexdigest()
        attribute_plan = decompose_visual_query(query, attribute_policy)
        if attribute_plan.compositional:
            visual_requirements = [
                row
                for row in attribute_plan.requirements
                if row.kind in VISUAL_REQUIREMENT_KINDS
            ]

    metadata: list[dict[str, Any]] = []
    vector_parts: list[np.ndarray] = []
    index_dirs = [project_path(args.index_dir)]
    if args.extra_index_dir is not None:
        index_dirs.append(project_path(args.extra_index_dir))
    for index_dir in index_dirs:
        metadata_path = index_dir / "metadata.jsonl"
        embeddings_path = index_dir / "embeddings.npy"
        if not metadata_path.is_file() and not embeddings_path.is_file():
            continue
        if not metadata_path.is_file() or not embeddings_path.is_file():
            raise ValueError(f"Incomplete visual index: {index_dir}")
        part_metadata = read_jsonl(metadata_path)
        part_vectors = np.load(embeddings_path).astype(np.float32)
        if len(part_metadata) != len(part_vectors):
            raise ValueError(f"Inconsistent visual index: {index_dir}")
        metadata.extend(part_metadata)
        vector_parts.append(part_vectors)
    if not vector_parts:
        raise FileNotFoundError("No visual index is available.")
    image_vectors = np.ascontiguousarray(
        np.vstack(vector_parts), dtype=np.float32
    )
    item_ids = [row["item_id"] for row in metadata]
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("Combined visual index contains duplicate item IDs.")

    torch.cuda.empty_cache()
    started = time.perf_counter()
    model = Qwen3VLEmbedder(
        model_name_or_path=str(project_path(args.model)),
        max_length=512,
        min_pixels=32 * 32 * 4,
        max_pixels=512 * 512,
        dtype=torch.float16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model_inputs = [
        {
            "text": encoded_query,
            "instruction": (
                "Retrieve the image that best matches the user's "
                "Chinese natural-language query."
            ),
        }
    ]
    model_inputs.extend(
        {
            "text": row.prompt,
            "instruction": (
                "Score whether the image independently satisfies this "
                "mandatory visual requirement. Do not reward a different "
                "attribute from the original compound query."
            ),
        }
        for row in visual_requirements
    )
    embeddings = model.process(model_inputs)
    torch.cuda.synchronize()
    query_matrix = embeddings.detach().float().cpu().numpy()
    query_vector = query_matrix[0]
    scores = image_vectors @ query_vector
    attribute_scores = (
        query_matrix[1:] @ image_vectors.T
        if visual_requirements
        else np.empty((0, len(item_ids)), dtype=np.float32)
    )

    payload = {
        "query": query,
        "encoded_query": encoded_query,
        "branch": "visual",
        "library_revision": args.library_revision,
        "item_ids": item_ids,
        "scores": [round(float(score), 8) for score in scores],
        "attribute_policy_sha256": attribute_policy_sha256,
        "attribute_plan": (
            attribute_plan.to_dict() if attribute_plan is not None else None
        ),
        "scores_by_requirement": {
            row.requirement_id: [
                round(float(value), 8)
                for value in attribute_scores[index]
            ]
            for index, row in enumerate(visual_requirements)
        },
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    output_path = project_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(output_path)


if __name__ == "__main__":
    main()
