from __future__ import annotations

"""Evaluate the shipped live-search path on a frozen query split.

The runner invokes ``scripts/live_search.py`` exactly as the product does,
keeps only compact per-query evidence, and deletes the large runtime payload
after every query.  It is intentionally an evaluator, not a parameter tuner.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = (
    PROJECT_ROOT
    / "data/evaluation/live_search_frozen_holdout_v1.json"
)


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def split_item_ids(value: str) -> set[str]:
    return {
        item_id.strip()
        for item_id in re.split(r"[;,|]", value or "")
        if item_id.strip()
    }


def read_frozen_queries(
    path: Path,
    *,
    split: str,
    required_review_status: str,
) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected: list[dict[str, Any]] = []
    for row in rows:
        if row.get("split", "").strip() != split:
            continue
        if row.get("review_status", "").strip() != required_review_status:
            continue
        selected.append(
            {
                "query_id": row["query_id"].strip(),
                "query": row["query"].strip(),
                "query_type": row.get("query_type", "unknown").strip()
                or "unknown",
                "is_no_answer": parse_bool(row.get("is_no_answer", False)),
                "relevant_item_ids": split_item_ids(
                    row.get("relevant_item_ids", "")
                ),
            }
        )
    selected.sort(key=lambda row: row["query_id"])
    if not selected:
        raise ValueError(f"No eligible queries found for split={split!r}")
    invalid = [
        row["query_id"]
        for row in selected
        if not row["is_no_answer"] and not row["relevant_item_ids"]
    ]
    if invalid:
        raise ValueError(
            "Answerable queries lack relevant_item_ids: " + ", ".join(invalid)
        )
    return selected


def apply_judgment_overrides(
    rows: list[dict[str, Any]],
    overrides: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Apply explicit human judgments without mutating the frozen source CSV."""
    override_by_id = {row["query_id"]: row for row in overrides}
    if len(override_by_id) != len(overrides):
        raise ValueError("Judgment overrides contain duplicate query IDs")
    known_ids = {row["query_id"] for row in rows}
    unknown = sorted(set(override_by_id) - known_ids)
    if unknown:
        raise ValueError("Unknown judgment override IDs: " + ", ".join(unknown))
    revised: list[dict[str, Any]] = []
    for source in rows:
        row = {**source, "relevant_item_ids": set(source["relevant_item_ids"])}
        override = override_by_id.get(row["query_id"])
        if override is not None:
            if "is_no_answer" in override:
                row["is_no_answer"] = bool(override["is_no_answer"])
            if "replace_relevant_item_ids" in override:
                row["relevant_item_ids"] = set(
                    override["replace_relevant_item_ids"]
                )
            row["relevant_item_ids"].update(
                override.get("add_relevant_item_ids", [])
            )
            if row["is_no_answer"]:
                row["relevant_item_ids"] = set()
            row["judgment_override"] = override.get("decision", "human_review")
        revised.append(row)
    invalid = [
        row["query_id"]
        for row in revised
        if not row["is_no_answer"] and not row["relevant_item_ids"]
    ]
    if invalid:
        raise ValueError(
            "Overrides left answerable queries without relevant items: "
            + ", ".join(invalid)
        )
    return revised


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def query_set_fingerprint(rows: Iterable[dict[str, Any]]) -> str:
    canonical = [
        {
            "query_id": row["query_id"],
            "query": row["query"],
            "query_type": row["query_type"],
            "is_no_answer": row["is_no_answer"],
            "relevant_item_ids": sorted(row["relevant_item_ids"]),
        }
        for row in rows
    ]
    encoded = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def relevant_rank(
    ranking: list[dict[str, Any]], relevant_item_ids: set[str]
) -> int | None:
    for index, item in enumerate(ranking, start=1):
        if item.get("item_id") in relevant_item_ids:
            return index
    return None


def ndcg_at_k(
    ranking: list[dict[str, Any]], relevant_item_ids: set[str], k: int
) -> float:
    if not relevant_item_ids:
        return 0.0
    observed = [
        1.0 if item.get("item_id") in relevant_item_ids else 0.0
        for item in ranking[:k]
    ]
    dcg = sum(
        relevance / math.log2(index + 2)
        for index, relevance in enumerate(observed)
    )
    ideal_count = min(len(relevant_item_ids), k)
    ideal = sum(1.0 / math.log2(index + 2) for index in range(ideal_count))
    return dcg / ideal if ideal else 0.0


def classify_errors(record: dict[str, Any]) -> list[str]:
    if record["is_no_answer"]:
        return ["false_accept"] if record["accepted"] else []
    errors: list[str] = []
    if not record["accepted"]:
        errors.append("false_reject")
    rank = record["relevant_rank"]
    if rank is None:
        errors.append("recall_miss_top_100")
    elif rank > 10:
        errors.append("recall_miss_top_10")
    elif rank > 1:
        errors.append("ranking_error_top_1")
    return errors


def summarize_live_payload(
    row: dict[str, Any],
    payload: dict[str, Any],
    *,
    method: str,
    wall_seconds: float,
) -> dict[str, Any]:
    ranking = list(payload.get("rankings", {}).get(method, []))
    decision = payload.get("acceptance", {}).get(method, {})
    accepted = bool(decision.get("accepted", False))
    rank = (
        None
        if row["is_no_answer"]
        else relevant_rank(ranking, row["relevant_item_ids"])
    )
    record = {
        "query_id": row["query_id"],
        "query": row["query"],
        "query_type": row["query_type"],
        "is_no_answer": row["is_no_answer"],
        "relevant_item_ids": sorted(row["relevant_item_ids"]),
        "accepted": accepted,
        "acceptance_reason": decision.get("reason", ""),
        "relevant_rank": rank,
        "ranker_ndcg_at_10": (
            None
            if row["is_no_answer"]
            else round(ndcg_at_k(ranking, row["relevant_item_ids"], 10), 6)
        ),
        "top_5_item_ids": [item.get("item_id") for item in ranking[:5]],
        "stored_result_count": len(ranking),
        "retrieval_route": payload.get("retrieval_route"),
        "query_mode": payload.get("query_mode"),
        "executed_branches": payload.get("executed_branches", {}),
        "component_cache_hits": payload.get("component_cache_hits", {}),
        "search_policy_version": payload.get("search_policy_version"),
        "retrieval_config_revision": payload.get(
            "retrieval_config_revision"
        ),
        "library_revision": payload.get("library_revision"),
        "internal_seconds": float(
            payload.get("timings", {}).get("total_seconds", 0.0)
        ),
        "wall_seconds": round(wall_seconds, 6),
    }
    component_values = [
        value
        for value in record["component_cache_hits"].values()
        if isinstance(value, bool)
    ]
    if component_values and all(component_values):
        record["component_cache_state"] = "full_hit"
    elif any(component_values):
        record["component_cache_state"] = "partial_hit"
    else:
        record["component_cache_state"] = "cold"
    record["errors"] = classify_errors(record)
    return record


def aggregate_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [row for row in records if not row["is_no_answer"]]
    no_answer = [row for row in records if row["is_no_answer"]]

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def recall(rows: list[dict[str, Any]], k: int, *, gated: bool) -> float:
        if not rows:
            return 0.0
        hits = 0
        for row in rows:
            rank = row["relevant_rank"]
            hits += int(
                rank is not None
                and rank <= k
                and (row["accepted"] or not gated)
            )
        return hits / len(rows)

    ranker_rr = [
        0.0 if row["relevant_rank"] is None else 1.0 / row["relevant_rank"]
        for row in answerable
    ]
    product_rr = [
        value if row["accepted"] else 0.0
        for row, value in zip(answerable, ranker_rr)
    ]
    ranker_ndcg = [float(row["ranker_ndcg_at_10"]) for row in answerable]
    product_ndcg = [
        value if row["accepted"] else 0.0
        for row, value in zip(answerable, ranker_ndcg)
    ]
    correct = [
        (not row["accepted"])
        if row["is_no_answer"]
        else (
            row["accepted"]
            and row["relevant_rank"] is not None
            and row["relevant_rank"] == 1
        )
        for row in records
    ]
    component_hits = [
        value
        for row in records
        for value in row.get("component_cache_hits", {}).values()
        if isinstance(value, bool)
    ]
    wall = [float(row["wall_seconds"]) for row in records]
    internal = [float(row["internal_seconds"]) for row in records]
    cache_state_counts = Counter(
        row.get("component_cache_state", "unknown") for row in records
    )
    latency_by_cache_state = {}
    for state in sorted(cache_state_counts):
        state_rows = [
            row
            for row in records
            if row.get("component_cache_state") == state
        ]
        state_wall = [float(row["wall_seconds"]) for row in state_rows]
        latency_by_cache_state[state] = {
            "query_count": len(state_rows),
            "wall_p50_seconds": round(float(percentile(state_wall, 0.5)), 6),
            "wall_p95_seconds": round(float(percentile(state_wall, 0.95)), 6),
        }
    metrics = {
        "query_count": len(records),
        "answerable_count": len(answerable),
        "no_answer_count": len(no_answer),
        "ranker_recall_at_1": recall(answerable, 1, gated=False),
        "ranker_recall_at_3": recall(answerable, 3, gated=False),
        "ranker_recall_at_5": recall(answerable, 5, gated=False),
        "ranker_recall_at_10": recall(answerable, 10, gated=False),
        "ranker_mrr": mean(ranker_rr),
        "ranker_ndcg_at_10": mean(ranker_ndcg),
        "answerable_acceptance_rate": mean(
            [float(row["accepted"]) for row in answerable]
        ),
        "no_answer_rejection_accuracy": mean(
            [float(not row["accepted"]) for row in no_answer]
        ),
        "false_accept_rate": mean(
            [float(row["accepted"]) for row in no_answer]
        ),
        "product_recall_at_1": recall(answerable, 1, gated=True),
        "product_recall_at_3": recall(answerable, 3, gated=True),
        "product_recall_at_5": recall(answerable, 5, gated=True),
        "product_recall_at_10": recall(answerable, 10, gated=True),
        "product_mrr": mean(product_rr),
        "product_ndcg_at_10": mean(product_ndcg),
        "end_to_end_top1_accuracy": mean([float(value) for value in correct]),
        "component_cache_hit_rate": mean(
            [float(value) for value in component_hits]
        ),
        "component_cache_state_counts": dict(sorted(cache_state_counts.items())),
        "latency_by_component_cache_state": latency_by_cache_state,
        "latency_wall_p50_seconds": percentile(wall, 0.5),
        "latency_wall_p95_seconds": percentile(wall, 0.95),
        "latency_internal_p50_seconds": percentile(internal, 0.5),
        "latency_internal_p95_seconds": percentile(internal, 0.95),
    }
    return {
        key: round(value, 6) if isinstance(value, float) else value
        for key, value in metrics.items()
    }


def metrics_by_query_type(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["query_type"]].append(record)
    return {
        query_type: aggregate_metrics(grouped[query_type])
        for query_type in sorted(grouped)
    }


def write_json_atomic(path: Path, payload: Any) -> None:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--max-queries",
        type=int,
        help="Diagnostic-only prefix; reports created with this flag are not formal.",
    )
    return parser.parse_args()


def run_query(
    row: dict[str, Any],
    *,
    protocol: dict[str, Any],
    runtime_path: Path,
) -> dict[str, Any]:
    command = live_search_command(row, protocol, runtime_path)
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=int(protocol.get("query_timeout_seconds", 240)),
            check=False,
        )
        wall_seconds = time.perf_counter() - started
        if completed.returncode != 0:
            stdout_tail = completed.stdout[-2000:]
            stderr_tail = completed.stderr[-4000:]
            raise RuntimeError(
                f"live_search failed for {row['query_id']} "
                f"(exit {completed.returncode})\n"
                f"stdout tail:\n{stdout_tail}\n"
                f"stderr tail:\n{stderr_tail}"
            )
        payload = json.loads(runtime_path.read_text(encoding="utf-8"))
        return summarize_live_payload(
            row,
            payload,
            method=protocol["method"],
            wall_seconds=wall_seconds,
        )
    finally:
        runtime_path.unlink(missing_ok=True)


def live_search_command(
    row: dict[str, Any],
    protocol: dict[str, Any],
    runtime_path: Path,
) -> list[str]:
    method = protocol["method"]
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/live_search.py"),
        row["query"],
        "--output",
        str(runtime_path),
        "--method",
        method,
        "--library-dir",
        str(project_path(protocol["library_dir"])),
        "--rerank-top-k",
        str(int(protocol.get("rerank_top_k", 0))),
    ]
    if bool(protocol.get("force_components", False)):
        command.append("--force")
    return command


def main() -> None:
    args = parse_args()
    protocol_path = project_path(args.protocol)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    query_path = project_path(protocol["query_file"])
    rows = read_frozen_queries(
        query_path,
        split=protocol["split"],
        required_review_status=protocol["required_review_status"],
    )
    overrides_path: Path | None = None
    if protocol.get("judgment_overrides_file"):
        overrides_path = project_path(protocol["judgment_overrides_file"])
        overrides_payload = json.loads(
            overrides_path.read_text(encoding="utf-8")
        )
        rows = apply_judgment_overrides(
            rows,
            list(overrides_payload["overrides"]),
        )
    formal = args.max_queries is None
    if args.max_queries is not None:
        if args.max_queries < 1:
            raise ValueError("--max-queries must be positive")
        rows = rows[: args.max_queries]
    expected_count = int(protocol["query_count"])
    if formal and len(rows) != expected_count:
        raise ValueError(
            f"Frozen query count changed: expected {expected_count}, got {len(rows)}"
        )
    expected_fingerprint = protocol["query_set_sha256"]
    actual_fingerprint = query_set_fingerprint(rows)
    if formal and actual_fingerprint != expected_fingerprint:
        raise ValueError(
            "Frozen query set fingerprint changed; refusing formal evaluation"
        )

    output_path = project_path(args.output or protocol["output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(
        prefix="live_blind_", dir=output_path.parent
    ) as runtime_dir:
        runtime_root = Path(runtime_dir)
        for index, row in enumerate(rows, start=1):
            record = run_query(
                row,
                protocol=protocol,
                runtime_path=runtime_root / f"{row['query_id']}.json",
            )
            records.append(record)
            rank_text = (
                "NA"
                if record["is_no_answer"]
                else str(record["relevant_rank"])
            )
            print(
                f"[{index:02d}/{len(rows):02d}] {record['query_id']} "
                f"route={record['retrieval_route']} "
                f"accepted={record['accepted']} rank={rank_text} "
                f"wall={record['wall_seconds']:.2f}s",
                flush=True,
            )

    policy_versions = sorted(
        {
            record["search_policy_version"]
            for record in records
            if record["search_policy_version"] is not None
        }
    )
    expected_policy = int(protocol["expected_search_policy_version"])
    if policy_versions != [expected_policy]:
        raise ValueError(
            f"Expected search policy V{expected_policy}, got {policy_versions}"
        )
    config_revisions = sorted(
        {
            record["retrieval_config_revision"]
            for record in records
            if record["retrieval_config_revision"]
        }
    )
    expected_config_revision = protocol.get(
        "expected_retrieval_config_revision"
    )
    if (
        expected_config_revision is not None
        and config_revisions != [str(expected_config_revision)]
    ):
        raise ValueError(
            "Expected retrieval config revision "
            f"{expected_config_revision}, got {config_revisions}"
        )
    library_revisions = sorted(
        {
            record["library_revision"]
            for record in records
            if record["library_revision"]
        }
    )
    error_counts = Counter(
        error for record in records for error in record["errors"]
    )
    report = {
        "benchmark": {
            "name": protocol["name"],
            "formal": formal,
            "seal_status": protocol["seal_status"],
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_path": protocol_path.relative_to(PROJECT_ROOT).as_posix(),
            "protocol_sha256": sha256_file(protocol_path),
            "query_file": query_path.relative_to(PROJECT_ROOT).as_posix(),
            "query_file_sha256": sha256_file(query_path),
            "query_set_sha256": actual_fingerprint,
            "judgment_overrides_file": (
                overrides_path.relative_to(PROJECT_ROOT).as_posix()
                if overrides_path is not None
                else None
            ),
            "judgment_overrides_sha256": (
                sha256_file(overrides_path)
                if overrides_path is not None
                else None
            ),
            "adjudicated_after_system_review": bool(
                protocol.get("adjudicated_after_system_review", False)
            ),
            "split": protocol["split"],
            "method": protocol["method"],
            "rerank_top_k": int(protocol.get("rerank_top_k", 0)),
            "force_components": bool(protocol.get("force_components", False)),
            "limitations": protocol.get("limitations", []),
        },
        "runtime": {
            "python": sys.executable,
            "search_policy_versions": policy_versions,
            "retrieval_config_revisions": config_revisions,
            "library_revisions": library_revisions,
        },
        "metrics": aggregate_metrics(records),
        "metrics_by_query_type": metrics_by_query_type(records),
        "error_summary": dict(sorted(error_counts.items())),
        "details": records,
    }
    write_json_atomic(output_path, report)
    print(output_path)


if __name__ == "__main__":
    main()
