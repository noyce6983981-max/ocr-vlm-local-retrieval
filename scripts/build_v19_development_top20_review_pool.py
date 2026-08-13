"""Build a blinded Top-20 multi-relevance review pool for V19 development."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts.run_v19_downstream_retrieval_pilot import ranking_ids  # noqa: E402

DEFAULT_ASSIGNMENTS = (
    ROOT
    / "outputs/evaluation/v19/selective_intervention"
    / "development_route_assignments.json"
)
DEFAULT_BASELINE_DIR = (
    ROOT / "outputs/evaluation/v19/selective_intervention/retrieval/baseline"
)
DEFAULT_GUARDED_DIR = (
    ROOT / "outputs/evaluation/v19/selective_intervention/retrieval/guarded"
)
DEFAULT_LIBRARY = ROOT / "outputs/user_library"
DEFAULT_OUTPUT = (
    ROOT
    / "records/private/v19/selective_intervention"
    / "development_top20_review_packets.jsonl"
)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
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


def manifest_index(library_dir: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    with (library_dir / "manifest.jsonl").open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            item_id = str(row.get("item_id", ""))
            if item_id:
                result[item_id] = row
    return result


def stable_blind_order(query_id: str, item_ids: Sequence[str]) -> list[str]:
    return sorted(
        item_ids,
        key=lambda item_id: hashlib.sha256(
            f"v19-development-top20\0{query_id}\0{item_id}".encode()
        ).hexdigest(),
    )


def pool_item_ids(
    baseline: Mapping[str, Any],
    guarded: Mapping[str, Any] | None,
    *,
    pool_size: int,
) -> list[str]:
    ordered: list[str] = []
    for payload in (baseline, guarded):
        if payload is None:
            continue
        for item_id in ranking_ids(dict(payload))[:pool_size]:
            if item_id not in ordered:
                ordered.append(item_id)
    return ordered[:pool_size]


def build_packets(
    assignments_payload: Mapping[str, Any],
    *,
    baseline_payloads: Mapping[str, Mapping[str, Any]],
    guarded_payloads: Mapping[str, Mapping[str, Any]],
    manifest: Mapping[str, Mapping[str, Any]],
    pool_size: int = 20,
) -> list[dict[str, Any]]:
    if assignments_payload.get("split") != "v19_reviewed_development_only":
        raise ValueError("Top-20 review pool may contain development only")
    if pool_size != 20:
        raise ValueError("V19 review protocol requires pool_size=20")
    packets: list[dict[str, Any]] = []
    for assignment in assignments_payload.get("assignments", []):
        query_id = str(assignment["query_id"])
        if query_id not in baseline_payloads:
            raise ValueError(f"missing baseline payload for {query_id}")
        item_ids = pool_item_ids(
            baseline_payloads[query_id],
            guarded_payloads.get(query_id),
            pool_size=pool_size,
        )
        blind_ids = stable_blind_order(query_id, item_ids)
        candidates = []
        for item_id in blind_ids:
            if item_id not in manifest:
                raise ValueError(f"manifest missing {item_id}")
            source_path = str(manifest[item_id].get("source_path", ""))
            candidates.append(
                {
                    "item_id": item_id,
                    "review_asset": {"image_path": source_path},
                }
            )
        pool_sha256 = hashlib.sha256(
            json.dumps(
                {"query_id": query_id, "item_ids": sorted(item_ids)},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        packets.append(
            {
                "task_id": "v19_development_top20_pooled_relevance",
                "split": "development",
                "eligible_for_final_claim": False,
                "query_id": query_id,
                "family_id": assignment.get("family_id"),
                "query": assignment["query"],
                "query_role": assignment.get("query_role"),
                "content_stratum": assignment.get("content_stratum"),
                "candidate_count": len(candidates),
                "pool_sha256": pool_sha256,
                "candidates": candidates,
            }
        )
    return packets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--guarded-dir", type=Path, default=DEFAULT_GUARDED_DIR)
    parser.add_argument("--library-dir", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    assignments = read_json(args.assignments)
    baseline = {
        str(row["query_id"]): read_json(
            args.baseline_dir / f"{row['query_id']}_v18_frozen.json"
        )
        for row in assignments.get("assignments", [])
    }
    guarded = {
        str(row["query_id"]): read_json(
            args.guarded_dir / f"{row['query_id']}_b21.json"
        )
        for row in assignments.get("assignments", [])
        if (args.guarded_dir / f"{row['query_id']}_b21.json").is_file()
    }
    packets = build_packets(
        assignments,
        baseline_payloads=baseline,
        guarded_payloads=guarded,
        manifest=manifest_index(args.library_dir),
    )
    write_jsonl_atomic(args.output, packets)
    print(f"V19 development Top-20 packets: {len(packets)}")
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
