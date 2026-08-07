"""Build a resumable, multi-method candidate pool for a frozen blind study."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.blind_query_study import (
    assert_library_unchanged,
    load_protocol,
    read_candidates,
    save_candidate,
    verify_frozen_snapshot,
)
from scripts.live_search import library_revision
DEFAULT_METHODS = ("quality_hybrid", "text", "visual")


def project_path(value: Path) -> Path:
    return value if value.is_absolute() else PROJECT_ROOT / value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study-dir",
        type=Path,
        default=Path("data/evaluation/blind_study_v1"),
    )
    parser.add_argument(
        "--library-dir", type=Path, default=Path("outputs/user_library")
    )
    parser.add_argument(
        "--methods", nargs="+", default=list(DEFAULT_METHODS)
    )
    parser.add_argument("--top-per-method", type=int, default=12)
    parser.add_argument("--pool-size", type=int, default=18)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most this many pending queries; zero means all.",
    )
    parser.add_argument("--timeout", type=int, default=300)
    return parser.parse_args()


def ranking_for_method(
    payload: dict[str, Any], method: str
) -> list[dict[str, Any]]:
    ranking = list(payload.get("rankings", {}).get(method, []))
    if ranking:
        return ranking
    return list(payload.get("low_confidence_rankings", {}).get(method, []))


def pool_payloads(
    payloads: dict[str, dict[str, Any]],
    *,
    methods: list[str],
    top_per_method: int,
    pool_size: int,
) -> dict[str, Any]:
    """Pool method rankings with RRF so no single route owns the pool."""
    if not methods or any(method not in payloads for method in methods):
        raise ValueError("每种候选池方法都必须有检索结果。")
    if top_per_method < 1 or pool_size < 1:
        raise ValueError("候选池大小必须为正数。")
    first = payloads[methods[0]]
    versions = {
        (
            payload.get("search_policy_version"),
            payload.get("retrieval_config_revision"),
            payload.get("library_revision"),
        )
        for payload in payloads.values()
    }
    if len(versions) != 1:
        raise ValueError("多路候选的检索策略或资料库版本不一致。")
    pooled: dict[str, dict[str, Any]] = {}
    for method in methods:
        for rank, row in enumerate(
            ranking_for_method(payloads[method], method)[:top_per_method],
            start=1,
        ):
            item_id = str(row.get("item_id", "")).strip()
            if not item_id:
                continue
            entry = pooled.setdefault(
                item_id,
                {
                    "item_id": item_id,
                    "rrf_pool_score": 0.0,
                    "best_rank": rank,
                    "pool_sources": [],
                    "ranks": {},
                    "scores": {},
                },
            )
            entry["rrf_pool_score"] += 1.0 / (60.0 + rank)
            entry["best_rank"] = min(int(entry["best_rank"]), rank)
            entry["pool_sources"].append(method)
            entry["ranks"][method] = rank
            entry["scores"][method] = round(float(row.get("score", 0.0)), 6)
    ordered = sorted(
        pooled.values(),
        key=lambda row: (
            -float(row["rrf_pool_score"]),
            int(row["best_rank"]),
            str(row["item_id"]),
        ),
    )[:pool_size]
    for row in ordered:
        row["rrf_pool_score"] = round(float(row["rrf_pool_score"]), 8)
    primary = payloads.get("quality_hybrid", first)
    decision = primary.get("acceptance", {}).get("quality_hybrid", {})
    return {
        "candidate_item_ids": [row["item_id"] for row in ordered],
        "candidate_results": ordered,
        "system_accepted": bool(decision.get("accepted", False)),
        "acceptance_reason": str(decision.get("reason", "")),
        "retrieval_route": primary.get("retrieval_route", "unknown"),
        "query_mode": primary.get("query_mode", "unknown"),
        "search_policy_version": first.get("search_policy_version"),
        "retrieval_config_revision": first.get("retrieval_config_revision"),
        "pool_methods": methods,
    }


def blinded_candidate_order(
    item_ids: list[str],
    *,
    query_id: str,
    query_set_sha256: str,
) -> list[str]:
    """Return a reproducible assessor order that does not reveal system rank."""
    return sorted(
        item_ids,
        key=lambda item_id: hashlib.sha256(
            f"{query_set_sha256}\0{query_id}\0{item_id}".encode("utf-8")
        ).hexdigest(),
    )


def run_live_search(
    query: str,
    *,
    method: str,
    library_dir: Path,
    output_path: Path,
    timeout: int,
) -> dict[str, Any]:
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/live_search.py"),
            query,
            "--library-dir",
            str(library_dir),
            "--output",
            str(output_path),
            "--method",
            method,
            "--rerank-top-k",
            "0",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout)[-1800:]
        raise RuntimeError(f"{method} 候选生成失败：{detail}")
    return json.loads(output_path.read_text(encoding="utf-8"))


def build_pending_candidates(
    study_dir: Path,
    library_dir: Path,
    *,
    methods: list[str],
    top_per_method: int,
    pool_size: int,
    limit: int,
    timeout: int,
) -> dict[str, Any]:
    protocol = load_protocol(study_dir)
    frozen = verify_frozen_snapshot(study_dir)
    revision = library_revision(library_dir)
    assert_library_unchanged(protocol, revision)
    existing = read_candidates(study_dir)
    pending = [row for row in frozen if row["query_id"] not in existing]
    if limit > 0:
        pending = pending[:limit]
    processed: list[str] = []
    with tempfile.TemporaryDirectory(prefix="blind-candidates-") as temp_dir:
        temp_root = Path(temp_dir)
        for pending_index, row in enumerate(pending, start=1):
            started = time.perf_counter()
            payloads: dict[str, dict[str, Any]] = {}
            for method in methods:
                payloads[method] = run_live_search(
                    row["query"],
                    method=method,
                    library_dir=library_dir,
                    output_path=temp_root / f"{row['query_id']}_{method}.json",
                    timeout=timeout,
                )
            candidate = pool_payloads(
                payloads,
                methods=methods,
                top_per_method=top_per_method,
                pool_size=pool_size,
            )
            candidate["query_id"] = row["query_id"]
            blind_order = blinded_candidate_order(
                candidate["candidate_item_ids"],
                query_id=row["query_id"],
                query_set_sha256=protocol["query_set_sha256"],
            )
            result_by_id = {
                result["item_id"]: result
                for result in candidate["candidate_results"]
            }
            candidate["candidate_item_ids"] = blind_order
            candidate["candidate_results"] = [
                result_by_id[item_id] for item_id in blind_order
            ]
            candidate["assessor_order"] = "sha256_blinded_v1"
            save_candidate(
                study_dir,
                candidate,
                current_library_revision=revision,
            )
            processed.append(row["query_id"])
            print(
                f"[{pending_index}/{len(pending)}] {row['query_id']} "
                f"stored in {time.perf_counter() - started:.1f}s",
                flush=True,
            )
    return {
        "processed_query_ids": processed,
        "processed_count": len(processed),
        "candidate_count": len(read_candidates(study_dir)),
        "frozen_query_count": len(frozen),
        "remaining_count": len(frozen) - len(read_candidates(study_dir)),
    }


def main() -> None:
    args = parse_args()
    summary = build_pending_candidates(
        project_path(args.study_dir),
        project_path(args.library_dir),
        methods=list(args.methods),
        top_per_method=args.top_per_method,
        pool_size=args.pool_size,
        limit=args.limit,
        timeout=args.timeout,
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
