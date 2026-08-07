"""Build calibration-only high-recall pools for corpus-answerability review."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from ocr_vlm_retrieval.evaluation.pooling import (
    DEFAULT_HIGH_RECALL_RUN_DEPTHS,
    HIGH_RECALL_EXPANSION_SOURCES,
    blind_corpus_answerability_packet,
    build_high_recall_answerability_pool,
)
from scripts.build_v17_human_pool import (
    parse_run,
    read_rows,
    write_jsonl_atomic,
)


def index_candidates(path: Path) -> dict[str, list[dict[str, Any]]]:
    indexed: dict[str, list[dict[str, Any]]] = {}
    for row in read_rows(path):
        query_id = str(row.get("query_id", "")).strip()
        candidates = row.get("candidates", row.get("ranking"))
        if not query_id or not isinstance(candidates, list):
            raise ValueError(f"Rows need query_id and candidates/ranking: {path}")
        if query_id in indexed:
            raise ValueError(f"Duplicate query {query_id} in {path}")
        indexed[query_id] = [dict(candidate) for candidate in candidates]
    return indexed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument(
        "--run",
        action="append",
        type=parse_run,
        required=True,
        help=(
            "Repeat RUN_ID=PATH for dense_text, bm25, global_visual, "
            "v16_quality_hybrid, v17_quality_hybrid, and candidate_verifier."
        ),
    )
    parser.add_argument(
        "--expansion",
        action="append",
        type=parse_run,
        default=[],
        help=(
            "Optional SOURCE=PATH for same_source, near_duplicate, "
            "category_neighbor, or counterfactual candidates."
        ),
    )
    parser.add_argument(
        "--split",
        choices=("calibration",),
        default="calibration",
        help="Holdout pooling is intentionally unavailable before method lock.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/evaluation/v17/human_study/calibration/corpus_answerability"
        ),
    )
    return parser.parse_args()


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    args = parse_args()
    queries = [
        row
        for row in read_rows(project_path(args.queries))
        if row.get("split") == args.split
    ]
    if not queries:
        raise ValueError("No calibration queries were found")

    run_paths = dict(args.run)
    if len(run_paths) != len(args.run):
        raise ValueError("Run IDs must be unique")
    missing = sorted(set(DEFAULT_HIGH_RECALL_RUN_DEPTHS) - set(run_paths))
    if missing:
        raise ValueError("Missing required high-recall runs: " + ", ".join(missing))
    run_rows = {
        run_id: index_candidates(path) for run_id, path in run_paths.items()
    }

    expansion_paths = dict(args.expansion)
    if len(expansion_paths) != len(args.expansion):
        raise ValueError("Expansion source IDs must be unique")
    unknown_expansions = sorted(
        set(expansion_paths) - set(HIGH_RECALL_EXPANSION_SOURCES)
    )
    if unknown_expansions:
        raise ValueError(
            "Unknown expansion sources: " + ", ".join(unknown_expansions)
        )
    expansion_rows = {
        source_id: index_candidates(path)
        for source_id, path in expansion_paths.items()
    }

    query_ids = {str(row["query_id"]) for row in queries}
    for run_id, indexed in run_rows.items():
        if set(indexed) != query_ids:
            raise ValueError(f"Run {run_id} does not match calibration query IDs")
    for source_id, indexed in expansion_rows.items():
        unknown = sorted(set(indexed) - query_ids)
        if unknown:
            raise ValueError(
                f"Expansion {source_id} has unknown query IDs: {unknown}"
            )

    audit_rows: list[dict[str, Any]] = []
    reviewer_rows: list[dict[str, Any]] = []
    for query in sorted(queries, key=lambda row: str(row["query_id"])):
        query_id = str(query["query_id"])
        audit = build_high_recall_answerability_pool(
            query_id,
            {
                run_id: indexed[query_id]
                for run_id, indexed in run_rows.items()
            },
            expansions={
                source_id: indexed.get(query_id, [])
                for source_id, indexed in expansion_rows.items()
            },
        )
        audit.update(
            {
                "query": str(query["query"]),
                "split": str(query["split"]),
                "group_id": str(query["group_id"]),
            }
        )
        reviewer = blind_corpus_answerability_packet(audit)
        reviewer["query"] = str(query["query"])
        audit_rows.append(audit)
        reviewer_rows.append(reviewer)

    output_dir = project_path(args.output_dir)
    audit_path = output_dir / "pool_audit.jsonl"
    review_path = output_dir / "review_packets.jsonl"
    write_jsonl_atomic(audit_path, audit_rows)
    write_jsonl_atomic(review_path, reviewer_rows)
    print(
        json.dumps(
            {
                "task_id": "corpus_answerability",
                "split": args.split,
                "query_count": len(queries),
                "audit_output": str(audit_path),
                "review_output": str(review_path),
                "holdout_read": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
