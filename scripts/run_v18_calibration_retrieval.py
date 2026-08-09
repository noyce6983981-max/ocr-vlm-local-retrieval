"""Run multi-route retrieval for the isolated V18 calibration scope only."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from ocr_vlm_retrieval.studies.query_split import (  # noqa: E402
    query_set_fingerprint,
)
from scripts.live_search import library_revision  # noqa: E402
from scripts.run_v17_calibration_retrieval import (  # noqa: E402
    file_sha256,
    ranking_for,
    read_jsonl,
    run_live_search,
    write_json_atomic,
    write_jsonl_atomic,
)

RUN_METHODS = {
    "text": ("v16", "text"),
    "bm25": ("v16", "bm25"),
    "global_visual": ("v16", "visual"),
    "non_attribute_quality_hybrid": ("v16", "quality_hybrid"),
    "v17_quality_hybrid": ("v17", "quality_hybrid"),
    "v17_attribute_coverage": ("v17", "attribute_coverage"),
}


def ranking_for_v18(
    payload: Mapping[str, Any], method: str
) -> list[dict[str, Any]]:
    """Export a complete study ranking across confidence partitions.

    The interactive search payload separates displayable and low-confidence
    candidates.  V17's frozen exporter intentionally selects only one
    partition, which can leave a human-review pool with fewer than 20 unique
    candidates.  V18 keeps the frozen order within each partition, appends the
    low-confidence tail, and removes duplicate items without reading labels.
    """

    ranked = list(payload.get("rankings", {}).get(method, []))
    ranked.extend(payload.get("low_confidence_rankings", {}).get(method, []))
    merged: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for row in ranked:
        item_id = str(row.get("item_id", "")).strip()
        if not item_id or item_id in seen:
            continue
        seen.add(item_id)
        merged.append(row)
    return ranking_for({"rankings": {method: merged}}, method)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def validate_v18_raw_payload(
    payload: Mapping[str, Any], *, query: str, version: str
) -> None:
    """Validate retrieval identity without assuming the V17 parser fires.

    V18 deliberately includes natural OCR-heavy compound queries that the
    frozen V17 visual parser can classify as non-compositional.  Treating that
    legacy-parser miss as a runner failure would selectively remove the exact
    failure mode under study.  The activation miss is therefore retained and
    reported in the retrieval receipt instead.
    """

    if payload.get("query") != query:
        raise ValueError(f"Raw {version} payload does not match its frozen query")
    active = bool(payload.get("attribute_coverage_active"))
    if version == "v16" and active:
        raise ValueError("A V16 raw payload unexpectedly enables attribute coverage")
    if version not in {"v16", "v17"}:
        raise ValueError(f"Unsupported retrieval version: {version}")
    quality_ranking = payload.get("rankings", {}).get("quality_hybrid", [])
    low_quality_ranking = payload.get("low_confidence_rankings", {}).get(
        "quality_hybrid", []
    )
    if not quality_ranking and not low_quality_ranking:
        raise ValueError(f"Raw {version} payload has no quality-hybrid candidates")


def validate_calibration_scope(
    queries: Sequence[Mapping[str, Any]],
    scope_receipt: Mapping[str, Any],
    *,
    expected_count: int,
    expected_methods_sha256: str,
) -> None:
    if len(queries) != expected_count:
        raise ValueError(
            f"Expected {expected_count} V18 calibration queries, got {len(queries)}"
        )
    if any(row.get("split") != "calibration" for row in queries):
        raise ValueError("V18 retrieval input must contain calibration rows only")
    if any(row.get("review_status") != "human_query_approved" for row in queries):
        raise ValueError("every V18 calibration query must be human approved")
    if len({str(row.get("query_id", "")) for row in queries}) != len(queries):
        raise ValueError("V18 calibration query IDs must be unique")
    if scope_receipt.get("status") != "prepared":
        raise ValueError("V18 calibration scope is not prepared")
    if scope_receipt.get("exported_split") != "calibration":
        raise ValueError("V18 scope receipt does not isolate calibration")
    if int(scope_receipt.get("calibration_query_count", 0)) != expected_count:
        raise ValueError("V18 scope receipt calibration count does not match")
    if int(scope_receipt.get("holdout_query_count_not_exported", 0)) < 1:
        raise ValueError("V18 scope receipt does not attest a sealed holdout")
    if scope_receipt.get("retrieval_executed") is not False:
        raise ValueError("V18 calibration scope already records retrieval execution")
    if scope_receipt.get("holdout_results_opened") is not False:
        raise ValueError("V18 holdout access invariant is not satisfied")
    if scope_receipt.get("v17_artifacts_modified") is not False:
        raise ValueError("V17 read-only invariant is not satisfied")
    if str(scope_receipt.get("methods_sha256", "")) != expected_methods_sha256:
        raise ValueError("V18 method specification hash does not match the scope")
    if str(scope_receipt.get("calibration_query_set_sha256", "")) != (
        query_set_fingerprint(queries)
    ):
        raise ValueError("V18 calibration query fingerprint does not match")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--scope-receipt", type=Path, required=True)
    parser.add_argument(
        "--methods",
        type=Path,
        default=PROJECT_ROOT / "config/studies/v18_methods.json",
    )
    parser.add_argument(
        "--attribute-policy",
        type=Path,
        default=PROJECT_ROOT / "config/v17_attribute_coverage.json",
    )
    parser.add_argument("--library-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=80)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--dry-run", action="store_true")
    raw_mode = parser.add_mutually_exclusive_group()
    raw_mode.add_argument(
        "--reuse-raw",
        action="store_true",
        help="Require and reuse every existing raw payload without retrieval.",
    )
    raw_mode.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing validated raw payloads and retrieve only missing ones.",
    )
    return parser.parse_args()


def obtain_raw_payload(
    *,
    raw_path: Path,
    query: str,
    version: str,
    library_dir: Path,
    policy_path: Path,
    timeout: int,
    reuse_raw: bool,
    resume: bool,
) -> dict[str, Any]:
    if reuse_raw and not raw_path.is_file():
        raise FileNotFoundError(f"Missing raw payload: {raw_path}")
    if reuse_raw or (resume and raw_path.is_file()):
        payload = read_json(raw_path)
    else:
        payload = run_live_search(
            query,
            version=version,
            library_dir=library_dir,
            attribute_policy=policy_path,
            output_path=raw_path,
            timeout=timeout,
        )
    try:
        validate_v18_raw_payload(payload, query=query, version=version)
    except ValueError as error:
        if "has no quality-hybrid candidates" not in str(error):
            raise
        payload = run_study_route_fallback(
            query,
            version=version,
            library_dir=library_dir,
            attribute_policy=policy_path,
            output_path=raw_path,
            timeout=timeout,
        )
        validate_v18_raw_payload(payload, query=query, version=version)
    return payload


def run_study_route_fallback(
    query: str,
    *,
    version: str,
    library_dir: Path,
    attribute_policy: Path,
    output_path: Path,
    timeout: int,
) -> dict[str, Any]:
    """Retrieve all branches when the frozen entity route returns no pool."""

    refresh_path = output_path.with_name(
        f".{output_path.stem}.v18-fallback.{os.getpid()}.json"
    )
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/v18_live_search.py"),
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
                f"V18 study fallback failed for {query!r}:\n"
                f"stdout={completed.stdout}\nstderr={completed.stderr}"
            )
        payload = read_json(refresh_path)
        payload["v18_study_route_fallback"] = {
            "reason": "frozen_entity_route_returned_no_candidates",
            "original_route": "entity_exact",
            "forced_route": "mixed",
            "labels_read": False,
        }
        write_json_atomic(output_path, payload)
        return payload
    finally:
        refresh_path.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    queries_path = args.queries.resolve()
    scope_receipt_path = args.scope_receipt.resolve()
    methods_path = args.methods.resolve()
    policy_path = args.attribute_policy.resolve()
    library_dir = args.library_dir.resolve()
    output_dir = args.output_dir.resolve()

    methods = read_json(methods_path)
    base_ranking = str(
        methods.get("shared_candidate_policy", {}).get("base_ranking", "")
    )
    if base_ranking not in RUN_METHODS:
        raise ValueError("V18 base ranking is not produced by this runner")
    queries = read_jsonl(queries_path)
    scope_receipt = read_json(scope_receipt_path)
    validate_calibration_scope(
        queries,
        scope_receipt,
        expected_count=args.expected_count,
        expected_methods_sha256=file_sha256(methods_path),
    )
    scope = {
        "status": "validated_no_retrieval" if args.dry_run else "ready",
        "executed_split": "calibration",
        "calibration_query_count": len(queries),
        "holdout_query_count_not_loaded": int(
            scope_receipt["holdout_query_count_not_exported"]
        ),
        "calibration_query_set_sha256": scope_receipt[
            "calibration_query_set_sha256"
        ],
        "methods_sha256": file_sha256(methods_path),
        "base_ranking": base_ranking,
        "holdout_results_opened": False,
        "v17_artifacts_modified": False,
    }
    if args.dry_run:
        print(json.dumps(scope, ensure_ascii=False, sort_keys=True))
        return

    raw_dir = output_dir / "raw"
    payloads: dict[str, dict[str, dict[str, Any]]] = {"v16": {}, "v17": {}}
    for index, row in enumerate(queries, start=1):
        query_id = str(row["query_id"])
        print(f"[{index:02d}/{len(queries)}] {query_id}", flush=True)
        for version in ("v16", "v17"):
            raw_path = raw_dir / f"{query_id}_{version}.json"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            payload = obtain_raw_payload(
                raw_path=raw_path,
                query=str(row["query"]),
                version=version,
                library_dir=library_dir,
                policy_path=policy_path,
                timeout=args.timeout,
                reuse_raw=args.reuse_raw,
                resume=args.resume,
            )
            payloads[version][query_id] = payload

    run_dir = output_dir / "runs"
    run_paths: dict[str, Path] = {}
    for run_id, (version, method) in RUN_METHODS.items():
        rows = [
            {
                "query_id": str(query["query_id"]),
                "ranking": ranking_for_v18(
                    payloads[version][str(query["query_id"])], method
                ),
            }
            for query in queries
        ]
        path = run_dir / f"{run_id}.jsonl"
        write_jsonl_atomic(path, rows)
        run_paths[run_id] = path

    receipt = {
        **scope,
        "status": "calibration_retrieval_complete_relevance_not_yet_judged",
        "library_revision": library_revision(library_dir),
        "calibration_query_file_sha256": file_sha256(queries_path),
        "scope_receipt_sha256": file_sha256(scope_receipt_path),
        "attribute_policy_sha256": file_sha256(policy_path),
        "run_files": {
            run_id: {"path": str(path), "sha256": file_sha256(path)}
            for run_id, path in sorted(run_paths.items())
        },
        "legacy_attribute_coverage": {
            "active_query_count": sum(
                bool(payloads["v17"][str(query["query_id"])].get(
                    "attribute_coverage_active"
                ))
                for query in queries
            ),
            "inactive_query_ids": [
                str(query["query_id"])
                for query in queries
                if not payloads["v17"][str(query["query_id"])].get(
                    "attribute_coverage_active"
                )
            ],
            "interpretation": (
                "Retained legacy V17 parser misses; V18 does not drop or "
                "rewrite natural compound queries based on parser activation."
            ),
        },
        "study_route_fallback_query_ids": [
            str(query["query_id"])
            for query in queries
            if payloads["v16"][str(query["query_id"])].get(
                "v18_study_route_fallback"
            )
            or payloads["v17"][str(query["query_id"])].get(
                "v18_study_route_fallback"
            )
        ],
        "relevance_review_complete": False,
        "holdout_retrieval_executed": False,
    }
    write_json_atomic(output_dir / "calibration_retrieval_receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
