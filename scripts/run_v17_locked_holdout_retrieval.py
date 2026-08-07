"""Generate label-blind V17 holdout retrieval runs for human review."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from ocr_vlm_retrieval.evaluation.protocol_lock import (
    create_one_shot_receipt,
    file_sha256,
    validate_method_lock,
)
from scripts.live_search import library_revision
from scripts.run_v17_calibration_retrieval import (
    RUN_METHODS,
    ranking_for,
    validate_raw_payload,
    write_json_atomic,
    write_jsonl_atomic,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def run_live_search(
    query: str,
    *,
    version: str,
    runtime_root: Path,
    library_dir: Path,
    attribute_policy: Path,
    output_path: Path,
    timeout: int,
) -> dict[str, Any]:
    refresh = output_path.with_name(
        f".{output_path.stem}.refresh.{os.getpid()}.json"
    )
    command = [
        str(runtime_root / ".venv/Scripts/python.exe"),
        str(runtime_root / "scripts/live_search.py"),
        query,
        "--library-dir",
        str(library_dir),
        "--output",
        str(refresh),
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
            cwd=runtime_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
        if completed.returncode:
            raise RuntimeError(
                f"{version} retrieval failed for {query!r}:\n"
                f"stdout={completed.stdout}\nstderr={completed.stderr}"
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(refresh, output_path)
        return json.loads(output_path.read_text(encoding="utf-8"))
    finally:
        refresh.unlink(missing_ok=True)


def validate_runtime_code_mirror(
    lock: dict[str, Any], runtime_root: Path
) -> None:
    paths = {
        *lock["locked_files"],
        "scripts/live_search.py",
        "scripts/demo_backend.py",
        "scripts/bm25_retrieval.py",
        "scripts/color_retrieval.py",
        "scripts/evaluate_library_retrieval.py",
        "scripts/query_routing.py",
        "scripts/retrieval_rejection.py",
        "scripts/score_text_query.py",
        "scripts/score_bm25_query.py",
        "scripts/score_visual_query.py",
        "scripts/score_visual_attribute_queries.py",
        "config/selected_retrieval_config_v16.json",
    }
    for relative in sorted(paths):
        public_path = PROJECT_ROOT / relative
        runtime_path = runtime_root / relative
        if not public_path.is_file() or not runtime_path.is_file():
            raise FileNotFoundError(f"Missing retrieval dependency: {relative}")
        if file_sha256(public_path) != file_sha256(runtime_path):
            raise ValueError(
                f"Runtime code mirror differs from public code: {relative}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=40)
    parser.add_argument("--timeout", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime_root = args.runtime_root.resolve()
    lock_path = PROJECT_ROOT / "config/v17_method_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    validation = validate_method_lock(
        lock,
        project_root=PROJECT_ROOT,
        runtime_root=runtime_root,
        require_clean=False,
    )
    if not validation["valid"]:
        raise ValueError(f"Method lock is invalid: {validation['errors']}")
    validate_runtime_code_mirror(lock, runtime_root)

    final_receipt = runtime_root / lock["execution"]["receipt"]
    final_output = runtime_root / lock["execution"]["output"]
    if final_receipt.exists() or final_output.exists():
        raise FileExistsError("Final V17 evaluation already has a receipt or output")

    query_path = runtime_root / lock["query_set_path"]
    queries = [row for row in read_jsonl(query_path) if row.get("split") == "holdout"]
    if len(queries) != args.expected_count:
        raise ValueError(f"Expected {args.expected_count} holdout queries")
    if any(row.get("review_status") != "human_query_approved" for row in queries):
        raise ValueError("Every holdout query must be human_query_approved")

    output_dir = runtime_root / "outputs/evaluation/v17/holdout/retrieval"
    receipt_path = output_dir / "holdout_retrieval_receipt.json"
    if receipt_path.exists():
        raise FileExistsError("Holdout retrieval receipt already exists")
    raw_dir = output_dir / "raw"
    library_dir = runtime_root / "outputs/user_library"
    attribute_policy = runtime_root / "config/v17_attribute_coverage.json"
    payloads: dict[str, dict[str, dict[str, Any]]] = {"v16": {}, "v17": {}}
    for index, row in enumerate(queries, start=1):
        query_id = str(row["query_id"])
        print(f"[{index:02d}/{len(queries)}] {query_id}", flush=True)
        for version in ("v16", "v17"):
            raw_path = raw_dir / f"{query_id}_{version}.json"
            if raw_path.is_file():
                payload = json.loads(raw_path.read_text(encoding="utf-8"))
            else:
                payload = run_live_search(
                    str(row["query"]),
                    version=version,
                    runtime_root=runtime_root,
                    library_dir=library_dir,
                    attribute_policy=attribute_policy,
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
            for query in queries
        ]
        path = run_dir / f"{run_id}.jsonl"
        write_jsonl_atomic(path, rows)
        run_paths[run_id] = path

    receipt = {
        "schema_version": 1,
        "status": "holdout_retrieval_complete_human_review_not_started",
        "executed_split": "holdout",
        "query_count": len(queries),
        "judgments_read": False,
        "method_lock_sha256": file_sha256(lock_path),
        "frozen_queries_sha256": file_sha256(query_path),
        "library_revision": library_revision(library_dir),
        "run_files": {
            run_id: {"path": str(path), "sha256": file_sha256(path)}
            for run_id, path in run_paths.items()
        },
        "human_review_complete": False,
    }
    create_one_shot_receipt(receipt_path, receipt)
    write_json_atomic(output_dir / "holdout_retrieval_summary.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
