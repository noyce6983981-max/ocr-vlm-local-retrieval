"""Run frozen V16/V17 retrieval only for the V17 calibration split.

This entry point deliberately refuses every non-calibration row. It produces
standardized ranking files for blinded Top-20 pooling and an auditable receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.live_search import library_revision  # noqa: E402

RUN_METHODS = {
    "text": ("v16", "text"),
    "bm25": ("v16", "bm25"),
    "global_visual": ("v16", "visual"),
    "v16_quality_hybrid": ("v16", "quality_hybrid"),
    "v17_quality_hybrid": ("v17", "quality_hybrid"),
    "v17_attribute_coverage": ("v17", "attribute_coverage"),
}


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
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
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("data/evaluation/v17/frozen_query_set/frozen_queries.jsonl"),
    )
    parser.add_argument(
        "--freeze-receipt",
        type=Path,
        default=Path("data/evaluation/v17/frozen_query_set/freeze_receipt.json"),
    )
    parser.add_argument(
        "--attribute-policy",
        type=Path,
        default=Path("config/v17_attribute_coverage.json"),
    )
    parser.add_argument(
        "--library-dir", type=Path, default=Path("outputs/user_library")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/evaluation/v17/calibration"),
    )
    parser.add_argument("--expected-count", type=int, default=40)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the frozen calibration scope without executing retrieval.",
    )
    parser.add_argument(
        "--reuse-raw",
        action="store_true",
        help="Rebuild standardized runs from already validated raw payloads.",
    )
    return parser.parse_args()


def ranking_for(payload: dict[str, Any], method: str) -> list[dict[str, Any]]:
    ranking = list(payload.get("rankings", {}).get(method, []))
    if not ranking:
        ranking = list(payload.get("low_confidence_rankings", {}).get(method, []))
    normalized: list[dict[str, Any]] = []
    for row in ranking:
        item_id = str(row.get("item_id", "")).strip()
        if not item_id:
            continue
        normalized.append(
            {
                "item_id": item_id,
                "score": float(row.get("score", 0.0)),
                "title": row.get("display_name_zh") or row.get("source_file_name"),
                "source_name": row.get("public_source_name") or row.get("source"),
                "source_relpath": row.get("public_source_file"),
                "page_number": row.get("page_number"),
                "image_path": row.get("source_path"),
            }
        )
    return normalized


def validate_raw_payload(payload: dict[str, Any], *, query: str, version: str) -> None:
    if payload.get("query") != query:
        raise ValueError(f"Raw {version} payload does not match its frozen query.")
    if version == "v16" and payload.get("attribute_coverage_active"):
        raise ValueError("A V16 raw payload unexpectedly enables attribute coverage.")
    if version == "v17" and not payload.get("attribute_coverage_active"):
        raise ValueError("A V17 raw payload is missing active attribute coverage.")


def run_live_search(
    query: str,
    *,
    version: str,
    library_dir: Path,
    attribute_policy: Path,
    output_path: Path,
    timeout: int,
) -> dict[str, Any]:
    refresh_path = output_path.with_name(
        f".{output_path.stem}.refresh.{os.getpid()}.json"
    )
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/live_search.py"),
        query,
        "--library-dir",
        str(library_dir),
        "--output",
        str(refresh_path),
        "--method",
        "quality_hybrid",
        "--rerank-top-k",
        "0",
    ]
    if version == "v17":
        command.extend(["--attribute-policy", str(attribute_policy)])
    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"{version} retrieval failed for {query!r}:\n"
                f"stdout={completed.stdout}\nstderr={completed.stderr}"
            )
        os.replace(refresh_path, output_path)
        return json.loads(output_path.read_text(encoding="utf-8"))
    finally:
        refresh_path.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    query_path = project_path(args.queries)
    receipt_path = project_path(args.freeze_receipt)
    policy_path = project_path(args.attribute_policy)
    library_dir = project_path(args.library_dir).resolve()
    output_dir = project_path(args.output_dir)

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "query_wording_frozen_relevance_not_yet_judged":
        raise ValueError("V17 query wording is not in the required frozen state.")
    all_queries = read_jsonl(query_path)
    calibration = [row for row in all_queries if row.get("split") == "calibration"]
    holdout = [row for row in all_queries if row.get("split") == "holdout"]
    if len(calibration) != args.expected_count:
        raise ValueError(
            f"Expected {args.expected_count} calibration rows, got {len(calibration)}"
        )
    if len(holdout) + len(calibration) != len(all_queries):
        raise ValueError("Frozen query set contains an unknown split.")
    if any(row.get("review_status") != "human_query_approved" for row in calibration):
        raise ValueError("Every calibration query must be human_query_approved.")
    if receipt.get("query_set_sha256") is None:
        raise ValueError("Freeze receipt has no query-set fingerprint.")

    scope = {
        "status": "validated_no_retrieval" if args.dry_run else "ready",
        "executed_split": "calibration",
        "calibration_query_count": len(calibration),
        "holdout_query_count_not_executed": len(holdout),
        "query_set_sha256": receipt["query_set_sha256"],
    }
    if args.dry_run:
        print(json.dumps(scope, ensure_ascii=False))
        return

    raw_dir = output_dir / "raw"
    payloads: dict[str, dict[str, dict[str, Any]]] = {"v16": {}, "v17": {}}
    for index, row in enumerate(calibration, start=1):
        query_id = str(row["query_id"])
        print(f"[{index:02d}/{len(calibration)}] {query_id}", flush=True)
        for version in ("v16", "v17"):
            raw_path = raw_dir / f"{query_id}_{version}.json"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            if args.reuse_raw:
                if not raw_path.is_file():
                    raise FileNotFoundError(f"Missing raw payload: {raw_path}")
                payload = json.loads(raw_path.read_text(encoding="utf-8"))
            else:
                payload = run_live_search(
                    str(row["query"]),
                    version=version,
                    library_dir=library_dir,
                    attribute_policy=policy_path,
                    output_path=raw_path,
                    timeout=args.timeout,
                )
            validate_raw_payload(payload, query=str(row["query"]), version=version)
            payloads[version][query_id] = payload

    run_dir = output_dir / "runs"
    run_paths: dict[str, Path] = {}
    for run_id, (version, method) in RUN_METHODS.items():
        rows = [
            {
                "query_id": str(query["query_id"]),
                "ranking": ranking_for(
                    payloads[version][str(query["query_id"])], method
                ),
            }
            for query in calibration
        ]
        path = run_dir / f"{run_id}.jsonl"
        write_jsonl_atomic(path, rows)
        run_paths[run_id] = path

    run_receipt = {
        **scope,
        "status": "calibration_retrieval_complete_relevance_not_yet_judged",
        "library_revision": library_revision(library_dir),
        "frozen_queries_sha256": file_sha256(query_path),
        "freeze_receipt_sha256": file_sha256(receipt_path),
        "attribute_policy_sha256": file_sha256(policy_path),
        "run_files": {
            run_id: {
                "path": path.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": file_sha256(path),
            }
            for run_id, path in run_paths.items()
        },
        "relevance_review_complete": False,
    }
    write_json_atomic(output_dir / "calibration_retrieval_receipt.json", run_receipt)
    print(json.dumps(run_receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
