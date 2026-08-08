"""Validate and prefill machine-authored study query drafts for human review."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from ocr_vlm_retrieval.studies.query_split import (  # noqa: E402
    ALLOWED_CHANGED_KINDS_BY_STRATUM,
    validate_authored_pair,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(dict(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _index_unique(
    rows: Sequence[Mapping[str, Any]], *, label: str
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        authoring_id = str(row.get("authoring_id", "")).strip()
        if not authoring_id or authoring_id in indexed:
            raise ValueError(f"{label} need unique non-empty authoring IDs")
        indexed[authoring_id] = row
    return indexed


def _normalized_query(value: str) -> str:
    return " ".join(value.split()).casefold()


def build_draft_submissions(
    queue: Sequence[Mapping[str, Any]],
    submissions: Sequence[Mapping[str, Any]],
    drafts: Sequence[Mapping[str, Any]],
    *,
    drafted_at: str,
) -> list[dict[str, Any]]:
    """Merge drafts without approving them or overwriting approved human work."""

    queue_by_id = _index_unique(queue, label="queue rows")
    submission_by_id = _index_unique(submissions, label="submission rows")
    draft_by_id = _index_unique(drafts, label="draft rows")

    unknown_submissions = sorted(set(submission_by_id) - set(queue_by_id))
    if unknown_submissions:
        raise ValueError(f"submissions contain unknown IDs: {unknown_submissions}")

    approved_ids = {
        authoring_id
        for authoring_id, row in submission_by_id.items()
        if str(row.get("review_action", "")) == "approve"
    }
    pending_ids = set(queue_by_id) - approved_ids
    if set(draft_by_id) != pending_ids:
        missing = sorted(pending_ids - set(draft_by_id))
        unexpected = sorted(set(draft_by_id) - pending_ids)
        raise ValueError(
            f"draft coverage mismatch; missing={missing}, unexpected={unexpected}"
        )

    seen_queries: dict[str, str] = {}
    for authoring_id in sorted(approved_ids):
        submission = submission_by_id[authoring_id]
        for field in ("positive_query", "hard_negative_query"):
            normalized = _normalized_query(str(submission.get(field, "")))
            if normalized:
                seen_queries[normalized] = f"{authoring_id}:{field}"

    merged: list[dict[str, Any]] = []
    for source in queue:
        authoring_id = str(source["authoring_id"])
        if authoring_id in approved_ids:
            merged.append(dict(submission_by_id[authoring_id]))
            continue

        draft = draft_by_id[authoring_id]
        stratum = str(source.get("stratum", ""))
        changed_kind = str(draft.get("changed_condition_kind", "")).strip()
        if changed_kind not in ALLOWED_CHANGED_KINDS_BY_STRATUM.get(stratum, set()):
            raise ValueError(
                f"changed condition {changed_kind!r} is invalid for "
                f"{authoring_id} ({stratum})"
            )
        positive = " ".join(str(draft.get("positive_query", "")).split())
        negative = " ".join(str(draft.get("hard_negative_query", "")).split())
        validate_authored_pair(
            positive,
            negative,
            language_target=str(source.get("language_target", "")),
            changed_condition_kind=changed_kind,
            single_condition_confirmed=True,
        )
        for field, value in (
            ("positive_query", positive),
            ("hard_negative_query", negative),
        ):
            normalized = _normalized_query(value)
            if normalized in seen_queries:
                raise ValueError(
                    f"duplicate normalized query at {authoring_id}:{field}; "
                    f"already used by {seen_queries[normalized]}"
                )
            seen_queries[normalized] = f"{authoring_id}:{field}"

        merged.append(
            {
                "authoring_id": authoring_id,
                "review_action": "codex_draft",
                "positive_query": positive,
                "hard_negative_query": negative,
                "changed_condition_kind": changed_kind,
                "single_condition_confirmed": False,
                "reviewer_id": "codex_draft",
                "notes": str(draft.get("notes", "")).strip(),
                "draft_author": str(draft.get("draft_author", "codex")).strip()
                or "codex",
                "draft_version": int(draft.get("draft_version", 1)),
                "drafted_at": drafted_at,
            }
        )
    return merged


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    authoring_root = (
        args.runtime_root.resolve() / "data/evaluation/v18/query_authoring"
    )
    queue_path = authoring_root / "authoring_queue.jsonl"
    submissions_path = authoring_root / "authoring_submissions.jsonl"
    drafts_path = authoring_root / "codex_drafts_v1.jsonl"
    receipt_path = authoring_root / "codex_draft_receipt.json"
    drafted_at = datetime.now(UTC).isoformat(timespec="seconds")
    archive_stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    archive_path = (
        authoring_root
        / "snapshots"
        / f"authoring_submissions.before_codex_drafts.{archive_stamp}.jsonl"
    )

    queue = read_jsonl(queue_path)
    submissions = read_jsonl(submissions_path)
    drafts = read_jsonl(drafts_path)
    merged = build_draft_submissions(
        queue,
        submissions,
        drafts,
        drafted_at=drafted_at,
    )
    draft_count = sum(row.get("review_action") == "codex_draft" for row in merged)
    approved_count = sum(row.get("review_action") == "approve" for row in merged)
    if args.dry_run:
        print(
            json.dumps(
                {"approved": approved_count, "codex_draft": draft_count},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return

    write_jsonl_atomic(archive_path, submissions)
    write_jsonl_atomic(submissions_path, merged)
    write_json_atomic(
        receipt_path,
        {
            "action": "prefill_codex_query_drafts",
            "approved_preserved": approved_count,
            "before_snapshot_path": str(archive_path),
            "before_snapshot_sha256": sha256_file(archive_path),
            "codex_drafts_prefilled": draft_count,
            "drafts_path": str(drafts_path),
            "drafts_sha256": sha256_file(drafts_path),
            "queue_sha256": sha256_file(queue_path),
            "submissions_sha256": sha256_file(submissions_path),
            "written_at": drafted_at,
        },
    )
    print(
        json.dumps(
            {"approved": approved_count, "codex_draft": draft_count},
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
