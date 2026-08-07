"""Capture compact route-gate features from a frozen live-search protocol."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def split_ids(value: str) -> set[str]:
    return {item_id.strip() for item_id in value.split(";") if item_id.strip()}


def top_values(rows: list[dict[str, Any]], field: str) -> tuple[float, float]:
    values = sorted(
        (float(row.get(field, 0.0)) for row in rows), reverse=True
    )
    if not values:
        return 0.0, 0.0
    return values[0], values[1] if len(values) > 1 else 0.0


def summarize(
    query_row: dict[str, str], payload: dict[str, Any]
) -> dict[str, Any]:
    ranking = list(payload["rankings"]["quality_hybrid"])
    top_ten = ranking[:10]
    top = top_ten[0] if top_ten else {}
    relevant = split_ids(query_row.get("relevant_item_ids", ""))
    def rank_for(method: str) -> int | None:
        return next(
            (
                index
                for index, row in enumerate(
                    payload["rankings"].get(method, []), start=1
                )
                if row.get("item_id") in relevant
            ),
            None,
        )

    relevant_rank = rank_for("quality_hybrid")
    dense_top, dense_second = top_values(top_ten, "raw_text_score")
    visual_top, visual_second = top_values(top_ten, "raw_visual_score")
    bm25_top, bm25_second = top_values(top_ten, "bm25_raw_score")
    metadata_top, _ = top_values(top_ten, "raw_metadata_score")
    return {
        "query_id": query_row["query_id"],
        "query": query_row["query"],
        "is_no_answer": str(parse_bool(query_row.get("is_no_answer", ""))).lower(),
        "reviewer_type": query_row.get("reviewer_type", ""),
        "query_type": query_row.get("query_type", ""),
        "split": query_row.get("split", ""),
        "category": query_row.get("category", ""),
        "route": payload.get("retrieval_route", ""),
        "search_policy_version": payload.get("search_policy_version", ""),
        "accepted": str(
            bool(payload["acceptance"]["quality_hybrid"]["accepted"])
        ).lower(),
        "relevant_rank": relevant_rank or "",
        "relevant_rank_text": rank_for("text") or "",
        "relevant_rank_visual": rank_for("visual") or "",
        "relevant_rank_bm25": rank_for("bm25") or "",
        "relevant_rank_rrf": rank_for("rrf") or "",
        "relevant_rank_quality_rrf": rank_for("quality_rrf") or "",
        "relevant_rank_adaptive": rank_for("adaptive") or "",
        "top_item_id": top.get("item_id", ""),
        "top_raw_text": float(top.get("raw_text_score", 0.0)),
        "top_raw_visual": float(top.get("raw_visual_score", 0.0)),
        "top_bm25_raw": float(top.get("bm25_raw_score", 0.0)),
        "top_raw_metadata": float(top.get("raw_metadata_score", 0.0)),
        "max10_raw_text": dense_top,
        "margin10_raw_text": dense_top - dense_second,
        "max10_raw_visual": visual_top,
        "margin10_raw_visual": visual_top - visual_second,
        "max10_bm25_raw": bm25_top,
        "margin10_bm25_raw": bm25_top - bm25_second,
        "max10_raw_metadata": metadata_top,
        "aligned_text_bm25_count": sum(
            float(row.get("raw_text_score", 0.0)) >= 0.5
            and float(row.get("bm25_raw_score", 0.0)) >= 4.0
            for row in top_ten
        ),
        "aligned_visual_text_count": sum(
            float(row.get("raw_visual_score", 0.0)) >= 0.45
            and float(row.get("raw_text_score", 0.0)) >= 0.5
            for row in top_ten
        ),
        "exact_topic_count": len(
            payload.get("exact_topic_evidence_item_ids", [])
        ),
        "composite_visual": str(
            bool(payload.get("composite_visual_query", False))
        ).lower(),
        "color_intent": payload.get("color_intent") or "",
    }


def capture(
    query_file: Path,
    output: Path,
    *,
    library_dir: Path,
    expected_policy: int | None,
) -> list[dict[str, Any]]:
    with query_file.open("r", encoding="utf-8-sig", newline="") as handle:
        queries = list(csv.DictReader(handle))
    rows: list[dict[str, Any]] = []
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="route_gate_", dir=output.parent
    ) as temporary_dir:
        runtime_path = Path(temporary_dir) / "runtime.json"
        for index, query_row in enumerate(queries, start=1):
            command = [
                sys.executable,
                str(PROJECT_ROOT / "scripts/live_search.py"),
                query_row["query"],
                "--output",
                str(runtime_path),
                "--method",
                "quality_hybrid",
                "--library-dir",
                str(library_dir),
            ]
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=300,
                check=False,
            )
            if completed.returncode:
                raise RuntimeError(
                    f"live_search failed for {query_row['query_id']}: "
                    f"{completed.stderr[-2000:]}"
                )
            payload = json.loads(runtime_path.read_text(encoding="utf-8"))
            if (
                expected_policy is not None
                and int(payload["search_policy_version"]) != expected_policy
            ):
                raise ValueError(
                    f"Expected policy V{expected_policy}, got "
                    f"V{payload['search_policy_version']}"
                )
            rows.append(summarize(query_row, payload))
            runtime_path.unlink(missing_ok=True)
            print(
                f"[{index:02d}/{len(queries):02d}] {query_row['query_id']}",
                flush=True,
            )
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--library-dir", type=Path, default=PROJECT_ROOT / "outputs/user_library"
    )
    parser.add_argument("--expected-policy", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = capture(
        args.queries,
        args.output,
        library_dir=args.library_dir,
        expected_policy=args.expected_policy,
    )
    print(f"captured={len(rows)}")
    print(args.output)


if __name__ == "__main__":
    main()
