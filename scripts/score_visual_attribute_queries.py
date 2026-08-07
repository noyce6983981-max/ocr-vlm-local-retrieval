"""Score V17 visual attributes in one Qwen3-VL model load.

The script intentionally keeps heavy model imports inside ``main`` so the CPU
test environment can import and validate the surrounding package without
installing PyTorch or model weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_REPO = PROJECT_ROOT / "third_party/Qwen3-VL-Embedding"
PACKAGE_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from ocr_vlm_retrieval.gating.attribute_coverage import (  # noqa: E402
    VISUAL_REQUIREMENT_KINDS,
    decompose_visual_query,
    load_attribute_policy,
)
from ocr_vlm_retrieval.runtime.cache import write_json_atomic  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("config/v17_attribute_coverage.json"),
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("artifacts/visual_index_dataset_v1"),
    )
    parser.add_argument("--extra-index-dir", type=Path, default=None)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/qwen3-vl-embedding-2b"),
    )
    parser.add_argument("--output", type=Path, required=True)
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
    query = " ".join(args.query.split())
    if not query:
        raise ValueError("Query cannot be empty.")

    policy_path = project_path(args.policy)
    policy = load_attribute_policy(policy_path)
    policy_sha256 = hashlib.sha256(policy_path.read_bytes()).hexdigest()
    plan = decompose_visual_query(query, policy)
    visual_requirements = [
        row for row in plan.requirements if row.kind in VISUAL_REQUIREMENT_KINDS
    ]
    output_path = project_path(args.output)
    if not plan.compositional or not visual_requirements:
        write_json_atomic(
            output_path,
            {
                "query": query,
                "library_revision": args.library_revision,
                "policy_sha256": policy_sha256,
                "plan": plan.to_dict(),
                "item_ids": [],
                "scores_by_requirement": {},
                "elapsed_seconds": 0.0,
            },
        )
        print(output_path)
        return

    import numpy as np
    import torch

    if str(OFFICIAL_REPO) not in sys.path:
        sys.path.insert(0, str(OFFICIAL_REPO))
    from src.models.qwen3_vl_embedding import Qwen3VLEmbedder

    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3-VL attribute encoding requires CUDA.")

    metadata: list[dict[str, Any]] = []
    vector_parts: list[Any] = []
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

    item_ids = [row["item_id"] for row in metadata]
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("Combined visual index contains duplicate item IDs.")
    image_vectors = np.ascontiguousarray(
        np.vstack(vector_parts), dtype=np.float32
    )

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
    embeddings = model.process(
        [
            {
                "text": row.prompt,
                "instruction": (
                    "Score whether the image independently satisfies this "
                    "mandatory visual requirement. Do not reward a different "
                    "attribute from the original compound query."
                ),
            }
            for row in visual_requirements
        ]
    )
    torch.cuda.synchronize()
    query_matrix = np.ascontiguousarray(
        embeddings.detach().float().cpu().numpy(), dtype=np.float32
    )
    scores = query_matrix @ image_vectors.T
    write_json_atomic(
        output_path,
        {
            "query": query,
            "library_revision": args.library_revision,
            "policy_sha256": policy_sha256,
            "plan": plan.to_dict(),
            "item_ids": item_ids,
            "scores_by_requirement": {
                row.requirement_id: [
                    round(float(value), 8) for value in scores[index]
                ]
                for index, row in enumerate(visual_requirements)
            },
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        },
    )
    print(output_path)


if __name__ == "__main__":
    main()
