"""Build auditable and blinded Top-20-union pools for the V17 study.

Each input run is JSON or JSONL with one row per query::

    {"query_id": "q001", "ranking": [{"item_id": "page-1", "score": 0.8}]}

The audit output keeps run ranks and scores. The reviewer output removes them
and randomizes candidate order deterministically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from ocr_vlm_retrieval.evaluation.human_evaluation import (
    blind_candidate_pool,
    pool_ranked_runs,
)
from ocr_vlm_retrieval.evaluation.judgments import POOLED_RELEVANCE_TASK


def read_rows(path: Path) -> list[dict[str, Any]]:
    content = path.read_text(encoding="utf-8-sig").strip()
    if not content:
        return []
    if content.startswith("["):
        payload = json.loads(content)
        if not isinstance(payload, list):
            raise ValueError(f"Expected a JSON array: {path}")
        return [dict(row) for row in payload]
    if content.startswith("{") and "\n" not in content:
        payload = json.loads(content)
        if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
            return [dict(row) for row in payload["rows"]]
        return [dict(payload)]
    return [json.loads(line) for line in content.splitlines() if line.strip()]


def canonical_fingerprint(rows: list[dict[str, Any]]) -> str:
    canonical = [
        {
            "query_id": str(row["query_id"]),
            "query": " ".join(str(row["query"]).split()),
            "split": str(row["split"]),
            "group_id": str(row["group_id"]),
        }
        for row in sorted(rows, key=lambda value: str(value["query_id"]))
    ]
    return hashlib.sha256(
        json.dumps(
            canonical,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def study_fingerprint(
    query_fingerprint: str,
    runs: dict[str, dict[str, list[dict[str, Any]]]],
    *,
    top_per_run: int,
    pool_size: int | None,
    guaranteed_depth: int,
) -> str:
    material = {
        "query_fingerprint": query_fingerprint,
        "top_per_run": top_per_run,
        "pool_size": pool_size,
        "guaranteed_depth": guaranteed_depth,
        "ranked_item_ids": {
            run_id: {
                query_id: [str(row.get("item_id", "")) for row in ranking[:top_per_run]]
                for query_id, ranking in sorted(indexed.items())
            }
            for run_id, indexed in sorted(runs.items())
        },
    }
    return hashlib.sha256(
        json.dumps(
            material,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def candidate_pool_sha256(
    query_id: str,
    study_fingerprint_value: str,
    item_ids: list[str],
) -> str:
    material = {
        "task_id": POOLED_RELEVANCE_TASK,
        "query_id": query_id,
        "study_fingerprint": study_fingerprint_value,
        "item_ids": sorted(item_ids),
    }
    return hashlib.sha256(
        json.dumps(
            material,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def parse_run(value: str) -> tuple[str, Path]:
    run_id, separator, raw_path = value.partition("=")
    if not separator or not run_id.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("--run must use RUN_ID=PATH")
    path = Path(raw_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return run_id.strip(), path


def index_run(path: Path) -> dict[str, list[dict[str, Any]]]:
    indexed: dict[str, list[dict[str, Any]]] = {}
    for row in read_rows(path):
        query_id = str(row.get("query_id", "")).strip()
        ranking = row.get("ranking")
        if not query_id or not isinstance(ranking, list):
            raise ValueError(f"Each run row needs query_id and ranking array: {path}")
        if query_id in indexed:
            raise ValueError(f"Duplicate query {query_id} in {path}")
        indexed[query_id] = [dict(item) for item in ranking]
    return indexed


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_pools(
    queries: list[dict[str, Any]],
    runs: dict[str, dict[str, list[dict[str, Any]]]],
    *,
    top_per_run: int,
    pool_size: int | None = None,
    guaranteed_depth: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    query_fingerprint = canonical_fingerprint(queries)
    audit_rows: list[dict[str, Any]] = []
    reviewer_rows: list[dict[str, Any]] = []
    known_query_ids = {str(row["query_id"]) for row in queries}
    for run_id, indexed in runs.items():
        missing = sorted(known_query_ids - set(indexed))
        unknown = sorted(set(indexed) - known_query_ids)
        if missing or unknown:
            raise ValueError(
                f"Run {run_id} query mismatch; missing={missing}, unknown={unknown}"
            )
    if pool_size is not None and pool_size < 1:
        raise ValueError("pool_size must be positive when supplied")
    if guaranteed_depth < 0:
        raise ValueError("guaranteed_depth cannot be negative")
    if guaranteed_depth and pool_size is None:
        raise ValueError("guaranteed_depth requires a capped pool")
    fingerprint = study_fingerprint(
        query_fingerprint,
        runs,
        top_per_run=top_per_run,
        pool_size=pool_size,
        guaranteed_depth=guaranteed_depth,
    )
    for query in sorted(queries, key=lambda row: str(row["query_id"])):
        query_id = str(query["query_id"])
        pooled = pool_ranked_runs(
            {run_id: indexed[query_id] for run_id, indexed in runs.items()},
            top_per_run=top_per_run,
        )
        uncapped_candidate_count = len(pooled)
        if pool_size is not None:
            guaranteed = [
                row
                for row in pooled
                if any(
                    int(rank) <= guaranteed_depth for rank in row["run_ranks"].values()
                )
            ]
            if len(guaranteed) > pool_size:
                raise ValueError(
                    f"Guaranteed Top-{guaranteed_depth} union has "
                    f"{len(guaranteed)} candidates, exceeding pool_size={pool_size}"
                )
            guaranteed_ids = {str(row["item_id"]) for row in guaranteed}
            fillers = [
                row for row in pooled if str(row["item_id"]) not in guaranteed_ids
            ]
            pooled = (guaranteed + fillers)[:pool_size]
        pool_sha256 = candidate_pool_sha256(
            query_id,
            fingerprint,
            [str(row["item_id"]) for row in pooled],
        )
        shared = {
            "task_id": POOLED_RELEVANCE_TASK,
            "query_id": query_id,
            "query": str(query["query"]),
            "split": str(query["split"]),
            "group_id": str(query["group_id"]),
            "query_family": str(query.get("query_family", "")),
            "route": str(query.get("route", "")),
            "pool_sha256": pool_sha256,
        }
        audit_rows.append(
            {
                **shared,
                "study_fingerprint": fingerprint,
                "top_per_run": top_per_run,
                "pool_size": pool_size,
                "guaranteed_depth": guaranteed_depth,
                "run_ids": sorted(runs),
                "uncapped_candidate_count": uncapped_candidate_count,
                "candidate_count": len(pooled),
                "candidates": pooled,
            }
        )
        reviewer_rows.append(
            {
                **shared,
                "study_fingerprint": fingerprint,
                "candidate_count": len(pooled),
                "candidates": blind_candidate_pool(
                    query_id,
                    pooled,
                    study_fingerprint=fingerprint,
                ),
            }
        )
    return audit_rows, reviewer_rows, fingerprint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument(
        "--run",
        action="append",
        type=parse_run,
        required=True,
        help="Repeat RUN_ID=PATH for V16, V17, BM25, visual, and reranker runs.",
    )
    parser.add_argument("--top-per-run", type=int, default=20)
    parser.add_argument(
        "--pool-size",
        type=int,
        default=0,
        help="Cap the final cross-run RRF pool; zero keeps the full union.",
    )
    parser.add_argument(
        "--guaranteed-depth",
        type=int,
        default=0,
        help="Keep every run's Top-k union before filling the capped RRF pool.",
    )
    parser.add_argument(
        "--split",
        choices=("calibration", "holdout"),
        help="Optionally build a pool for one frozen split only.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/evaluation/v17/human_study"),
    )
    return parser.parse_args()


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    args = parse_args()
    queries = read_rows(project_path(args.queries))
    if args.split is not None:
        queries = [row for row in queries if row.get("split") == args.split]
        if not queries:
            raise ValueError(f"No queries found for split={args.split!r}")
    run_paths = dict(args.run)
    if len(run_paths) != len(args.run):
        raise ValueError("Run IDs must be unique")
    runs = {run_id: index_run(path) for run_id, path in run_paths.items()}
    audit, reviewer, fingerprint = build_pools(
        queries,
        runs,
        top_per_run=args.top_per_run,
        pool_size=args.pool_size or None,
        guaranteed_depth=args.guaranteed_depth,
    )
    output_dir = project_path(args.output_dir)
    write_jsonl_atomic(output_dir / "pool_audit.jsonl", audit)
    write_jsonl_atomic(output_dir / "review_packets.jsonl", reviewer)
    print(
        json.dumps(
            {
                "query_count": len(queries),
                "run_ids": sorted(runs),
                "study_fingerprint": fingerprint,
                "pool_size": args.pool_size or None,
                "guaranteed_depth": args.guaranteed_depth,
                "audit_output": str(output_dir / "pool_audit.jsonl"),
                "review_output": str(output_dir / "review_packets.jsonl"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
