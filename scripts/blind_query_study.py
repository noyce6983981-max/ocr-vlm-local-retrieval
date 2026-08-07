"""State and audit utilities for independently authored blind queries.

The collection phase deliberately stores queries without running retrieval.
Only an immutable frozen snapshot may receive system-generated candidates and
human relevance judgments.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
DEFAULT_TARGET_COUNT = 60
DEFAULT_MINIMUM_COUNT = 50
DEFAULT_MAXIMUM_COUNT = 100
DEFAULT_MINIMUM_NO_ANSWER_PROBES = 10
VALID_STATUSES = {"collecting", "frozen", "reviewing", "completed"}
VALID_EXPECTATIONS = {"answerable", "unsure", "no_answer_probe"}
VALID_REVIEW_DECISIONS = {"answerable", "no_answer", "excluded"}
VALID_QUERY_ORIGINS = {"user_authored", "assistant_generated"}

SUBMISSION_FIELDS = [
    "query_id",
    "query",
    "expected_answerability",
    "authored_at",
]
REVIEW_FIELDS = [
    "query_id",
    "decision",
    "relevant_item_ids",
    "evidence_scope",
    "reviewer_type",
    "reviewer_id",
    "review_confidence",
    "human_notes",
    "reviewed_at",
]


def utc_now(now: datetime | None = None) -> str:
    value = now or datetime.now(timezone.utc)
    return value.isoformat(timespec="seconds")


def normalize_query(value: str) -> str:
    return " ".join(str(value).split())


def split_item_ids(value: str | Iterable[str]) -> list[str]:
    if isinstance(value, str):
        values = re.split(r"[;,|]", value)
    else:
        values = [str(item_id) for item_id in value]
    result: list[str] = []
    seen: set[str] = set()
    for item_id in values:
        normalized = item_id.strip()
        if normalized and normalized not in seen:
            result.append(normalized)
            seen.add(normalized)
    return result


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(
    path: Path,
    rows: Iterable[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for source in rows:
            writer.writerow(
                {field: str(source.get(field, "")) for field in fieldnames}
            )
    os.replace(temporary, path)


def _canonical_query_rows(rows: Iterable[dict[str, Any]]) -> bytes:
    normalized = [
        {
            "query_id": str(row["query_id"]),
            "query": normalize_query(str(row["query"])),
            "expected_answerability": str(
                row.get("expected_answerability", "unsure")
            ),
            "authored_at": str(row.get("authored_at", "")),
        }
        for row in rows
    ]
    return json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def query_rows_sha256(rows: Iterable[dict[str, Any]]) -> str:
    return hashlib.sha256(_canonical_query_rows(rows)).hexdigest()


def protocol_path(study_dir: Path) -> Path:
    return Path(study_dir) / "protocol.json"


def submissions_path(study_dir: Path) -> Path:
    return Path(study_dir) / "submissions.csv"


def frozen_queries_path(study_dir: Path) -> Path:
    return Path(study_dir) / "frozen_queries.csv"


def candidates_path(study_dir: Path) -> Path:
    return Path(study_dir) / "candidates.jsonl"


def reviews_path(study_dir: Path) -> Path:
    return Path(study_dir) / "reviews.csv"


def initialize_study(
    study_dir: Path,
    *,
    study_id: str,
    library_id: str,
    library_revision: str,
    target_count: int = DEFAULT_TARGET_COUNT,
    minimum_count: int = DEFAULT_MINIMUM_COUNT,
    maximum_count: int = DEFAULT_MAXIMUM_COUNT,
    minimum_no_answer_probes: int = DEFAULT_MINIMUM_NO_ANSWER_PROBES,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Create a collection protocol, or validate and return the existing one."""
    path = protocol_path(study_dir)
    if path.is_file():
        protocol = load_protocol(study_dir)
        if protocol["library_id"] != library_id:
            raise ValueError(
                "该盲测已绑定另一个资料库，不能混入当前资料库的查询。"
            )
        return protocol
    if not study_id.strip() or not library_id.strip() or not library_revision:
        raise ValueError("study_id、library_id 和 library_revision 不能为空。")
    if not (1 <= minimum_count <= target_count <= maximum_count):
        raise ValueError("数量必须满足 1 ≤ 最少条数 ≤ 目标条数 ≤ 最大条数。")
    if not (0 <= minimum_no_answer_probes <= target_count):
        raise ValueError("无答案探针数量必须在0和目标条数之间。")
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "study_id": study_id.strip(),
        "status": "collecting",
        "library_id": library_id.strip(),
        "library_revision_at_start": library_revision,
        "library_revision_at_freeze": None,
        "target_count": int(target_count),
        "minimum_count": int(minimum_count),
        "maximum_count": int(maximum_count),
        "minimum_no_answer_probes": int(minimum_no_answer_probes),
        "submission_count": 0,
        "candidate_count": 0,
        "review_count": 0,
        "created_at": utc_now(now),
        "frozen_at": None,
        "completed_at": None,
        "query_set_sha256": None,
        "candidate_policy": None,
        "query_origin": "user_authored",
        "study_type": "independent_blind",
        "collection_rule": (
            "Query authors receive no retrieval results before the snapshot "
            "is frozen."
        ),
    }
    _atomic_json(path, protocol)
    _write_csv(submissions_path(study_dir), [], SUBMISSION_FIELDS)
    return protocol


def load_protocol(study_dir: Path) -> dict[str, Any]:
    path = protocol_path(study_dir)
    if not path.is_file():
        raise FileNotFoundError(f"Blind-study protocol not found: {path}")
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("不支持的盲测协议版本。")
    if protocol.get("status") not in VALID_STATUSES:
        raise ValueError("盲测协议状态无效。")
    return protocol


def read_submissions(study_dir: Path, *, frozen: bool = False) -> list[dict[str, str]]:
    path = frozen_queries_path(study_dir) if frozen else submissions_path(study_dir)
    return _read_csv(path)


def _write_protocol(study_dir: Path, protocol: dict[str, Any]) -> None:
    _atomic_json(protocol_path(study_dir), protocol)


def assert_library_unchanged(
    protocol: dict[str, Any], current_library_revision: str
) -> None:
    expected = (
        protocol.get("library_revision_at_freeze")
        or protocol["library_revision_at_start"]
    )
    if current_library_revision != expected:
        raise ValueError(
            "资料库内容已变化。为保证盲测公平，当前轮次已暂停；"
            "请恢复原资料库版本或新建下一轮盲测。"
        )


def submit_query(
    study_dir: Path,
    *,
    query: str,
    expected_answerability: str = "unsure",
    current_library_revision: str,
    now: datetime | None = None,
) -> dict[str, str]:
    """Append one independently authored query without invoking retrieval."""
    protocol = load_protocol(study_dir)
    if protocol["status"] != "collecting":
        raise ValueError("本轮查询已冻结，不能继续添加或修改。")
    assert_library_unchanged(protocol, current_library_revision)
    normalized = normalize_query(query)
    if len(normalized) < 2:
        raise ValueError("查询至少需要2个字符。")
    if len(normalized) > 200:
        raise ValueError("查询不能超过200个字符。")
    if expected_answerability not in VALID_EXPECTATIONS:
        raise ValueError("无效的答案预期。")
    rows = read_submissions(study_dir)
    if len(rows) >= int(protocol["maximum_count"]):
        raise ValueError("本轮已达到最大查询数量。")
    duplicate_keys = {normalize_query(row["query"]).casefold() for row in rows}
    if normalized.casefold() in duplicate_keys:
        raise ValueError("这句话已经提交过，请换一个真实搜索需求。")
    prefix = re.sub(r"[^A-Za-z0-9]+", "_", protocol["study_id"]).strip("_")
    used_numbers = [
        int(match.group(1))
        for source in rows
        if (match := re.search(r"_(\d+)$", source["query_id"]))
    ]
    next_number = max(used_numbers, default=0) + 1
    row = {
        "query_id": f"{prefix}_{next_number:03d}",
        "query": normalized,
        "expected_answerability": expected_answerability,
        "authored_at": utc_now(now),
    }
    rows.append(row)
    _write_csv(submissions_path(study_dir), rows, SUBMISSION_FIELDS)
    protocol["submission_count"] = len(rows)
    _write_protocol(study_dir, protocol)
    return row


def delete_submission(
    study_dir: Path,
    query_id: str,
    *,
    current_library_revision: str,
) -> None:
    protocol = load_protocol(study_dir)
    if protocol["status"] != "collecting":
        raise ValueError("本轮查询已冻结，不能删除。")
    assert_library_unchanged(protocol, current_library_revision)
    rows = read_submissions(study_dir)
    kept = [row for row in rows if row["query_id"] != query_id]
    if len(kept) == len(rows):
        raise ValueError("找不到要删除的查询。")
    # IDs are audit identifiers and are deliberately not renumbered.
    _write_csv(submissions_path(study_dir), kept, SUBMISSION_FIELDS)
    protocol["submission_count"] = len(kept)
    _write_protocol(study_dir, protocol)


def update_submission_expectation(
    study_dir: Path,
    query_id: str,
    expected_answerability: str,
    *,
    current_library_revision: str,
) -> None:
    """Correct balance metadata while collection is still open."""
    protocol = load_protocol(study_dir)
    if protocol["status"] != "collecting":
        raise ValueError("本轮查询已冻结，不能修改答案预期。")
    assert_library_unchanged(protocol, current_library_revision)
    if expected_answerability not in VALID_EXPECTATIONS:
        raise ValueError("无效的答案预期。")
    rows = read_submissions(study_dir)
    matched = False
    for row in rows:
        if row["query_id"] == query_id:
            row["expected_answerability"] = expected_answerability
            matched = True
            break
    if not matched:
        raise ValueError("找不到要修改的查询。")
    _write_csv(submissions_path(study_dir), rows, SUBMISSION_FIELDS)


def replace_collecting_queries(
    study_dir: Path,
    entries: list[dict[str, str]],
    *,
    query_origin: str,
    current_library_revision: str,
    archive_path: Path | None = None,
    now: datetime | None = None,
) -> list[dict[str, str]]:
    """Atomically replace an open collection while archiving existing rows."""
    protocol = load_protocol(study_dir)
    if protocol["status"] != "collecting":
        raise ValueError("本轮查询已冻结，不能替换查询集合。")
    assert_library_unchanged(protocol, current_library_revision)
    if query_origin not in VALID_QUERY_ORIGINS:
        raise ValueError("无效的查询来源。")
    if not entries:
        raise ValueError("替换查询集合不能为空。")
    if len(entries) > int(protocol["maximum_count"]):
        raise ValueError("替换查询数量超过本轮上限。")
    normalized_entries: list[dict[str, str]] = []
    seen: set[str] = set()
    prefix = re.sub(r"[^A-Za-z0-9]+", "_", protocol["study_id"]).strip("_")
    authored_at = utc_now(now)
    for index, entry in enumerate(entries, start=1):
        query = normalize_query(entry.get("query", ""))
        expectation = entry.get("expected_answerability", "unsure")
        if len(query) < 2 or len(query) > 200:
            raise ValueError(f"第 {index} 条查询长度必须为2到200个字符。")
        if query.casefold() in seen:
            raise ValueError(f"替换集合中有重复查询：{query}")
        if expectation not in VALID_EXPECTATIONS:
            raise ValueError(f"第 {index} 条查询的答案预期无效。")
        seen.add(query.casefold())
        normalized_entries.append(
            {
                "query_id": f"{prefix}_{index:03d}",
                "query": query,
                "expected_answerability": expectation,
                "authored_at": authored_at,
            }
        )
    previous = read_submissions(study_dir)
    if previous and archive_path is None:
        raise ValueError("替换已有查询前必须提供归档路径。")
    if previous and archive_path is not None:
        if archive_path.resolve() == submissions_path(study_dir).resolve():
            raise ValueError("归档路径不能覆盖当前采集文件。")
        _write_csv(archive_path, previous, SUBMISSION_FIELDS)
    _write_csv(
        submissions_path(study_dir), normalized_entries, SUBMISSION_FIELDS
    )
    protocol.update(
        {
            "submission_count": len(normalized_entries),
            "query_origin": query_origin,
            "study_type": (
                "independent_blind"
                if query_origin == "user_authored"
                else "assistant_generated_diagnostic"
            ),
            "collection_replaced_at": authored_at,
            "replaced_submission_count": len(previous),
            "replaced_submission_archive": (
                archive_path.as_posix() if archive_path is not None else None
            ),
        }
    )
    _write_protocol(study_dir, protocol)
    return normalized_entries


def freeze_study(
    study_dir: Path,
    *,
    current_library_revision: str,
    require_target: bool = True,
    now: datetime | None = None,
) -> dict[str, Any]:
    protocol = load_protocol(study_dir)
    if protocol["status"] != "collecting":
        raise ValueError("只有采集中的盲测可以冻结。")
    assert_library_unchanged(protocol, current_library_revision)
    rows = read_submissions(study_dir)
    required = int(
        protocol["target_count"] if require_target else protocol["minimum_count"]
    )
    if len(rows) < required:
        raise ValueError(f"至少需要 {required} 条查询才能冻结。")
    if len(rows) > int(protocol["maximum_count"]):
        raise ValueError("查询数量超过本轮上限。")
    no_answer_probes = sum(
        row.get("expected_answerability") == "no_answer_probe"
        for row in rows
    )
    minimum_no_answer = int(protocol.get("minimum_no_answer_probes", 0))
    if no_answer_probes < minimum_no_answer:
        raise ValueError(
            f"至少需要 {minimum_no_answer} 条专测无答案的查询；"
            f"当前只有 {no_answer_probes} 条。"
        )
    _write_csv(frozen_queries_path(study_dir), rows, SUBMISSION_FIELDS)
    digest = query_rows_sha256(rows)
    protocol.update(
        {
            "status": "frozen",
            "submission_count": len(rows),
            "library_revision_at_freeze": current_library_revision,
            "query_set_sha256": digest,
            "frozen_at": utc_now(now),
        }
    )
    _write_protocol(study_dir, protocol)
    return protocol


def verify_frozen_snapshot(study_dir: Path) -> list[dict[str, str]]:
    protocol = load_protocol(study_dir)
    if protocol["status"] == "collecting":
        raise ValueError("查询尚未冻结。")
    rows = read_submissions(study_dir, frozen=True)
    digest = query_rows_sha256(rows)
    if not rows or digest != protocol.get("query_set_sha256"):
        raise ValueError("冻结查询的指纹不匹配，已停止后续评测。")
    return rows


def read_candidates(study_dir: Path) -> dict[str, dict[str, Any]]:
    path = candidates_path(study_dir)
    if not path.is_file():
        return {}
    result: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        result[str(row["query_id"])] = row
    return result


def _write_candidates(
    study_dir: Path, candidates: dict[str, dict[str, Any]]
) -> None:
    order = {
        row["query_id"]: index
        for index, row in enumerate(verify_frozen_snapshot(study_dir))
    }
    rows = sorted(candidates.values(), key=lambda row: order[row["query_id"]])
    content = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in rows
    )
    _atomic_text(candidates_path(study_dir), content)


def save_candidate(
    study_dir: Path,
    candidate: dict[str, Any],
    *,
    current_library_revision: str,
    now: datetime | None = None,
) -> None:
    protocol = load_protocol(study_dir)
    if protocol["status"] not in {"frozen", "reviewing"}:
        raise ValueError("候选只能写入已冻结且未完成的盲测。")
    assert_library_unchanged(protocol, current_library_revision)
    frozen = verify_frozen_snapshot(study_dir)
    frozen_ids = {row["query_id"] for row in frozen}
    query_id = str(candidate.get("query_id", "")).strip()
    if query_id not in frozen_ids:
        raise ValueError("候选对应的查询不在冻结集合中。")
    candidate_ids = split_item_ids(candidate.get("candidate_item_ids", []))
    if not candidate_ids:
        raise ValueError("候选池不能为空。")
    policy = {
        "search_policy_version": candidate.get("search_policy_version"),
        "retrieval_config_revision": candidate.get(
            "retrieval_config_revision"
        ),
        "library_revision": current_library_revision,
        "pool_methods": list(candidate.get("pool_methods", [])),
    }
    if not policy["search_policy_version"] or not policy["pool_methods"]:
        raise ValueError("候选缺少检索策略版本或候选池方法。")
    existing_policy = protocol.get("candidate_policy")
    if existing_policy is not None and existing_policy != policy:
        raise ValueError("候选策略与本轮已生成候选不一致。")
    normalized = dict(candidate)
    normalized["query_id"] = query_id
    normalized["candidate_item_ids"] = candidate_ids
    normalized["generated_at"] = utc_now(now)
    candidates = read_candidates(study_dir)
    candidates[query_id] = normalized
    _write_candidates(study_dir, candidates)
    protocol["candidate_policy"] = policy
    protocol["candidate_count"] = len(candidates)
    protocol["status"] = "reviewing"
    _write_protocol(study_dir, protocol)


def read_reviews(study_dir: Path) -> dict[str, dict[str, str]]:
    return {
        row["query_id"]: row
        for row in _read_csv(reviews_path(study_dir))
        if row.get("query_id")
    }


def save_relevance_review(
    study_dir: Path,
    review: dict[str, Any],
    *,
    known_item_ids: set[str],
    now: datetime | None = None,
) -> None:
    protocol = load_protocol(study_dir)
    if protocol["status"] != "reviewing":
        raise ValueError("只有候选已生成的盲测可以审核。")
    query_id = str(review.get("query_id", "")).strip()
    candidates = read_candidates(study_dir)
    if query_id not in candidates:
        raise ValueError("这条查询尚未生成盲审候选池。")
    decision = str(review.get("decision", "")).strip()
    if decision not in VALID_REVIEW_DECISIONS:
        raise ValueError("无效的相关性裁决。")
    relevant_ids = split_item_ids(review.get("relevant_item_ids", ""))
    if decision == "answerable" and not relevant_ids:
        raise ValueError("有答案查询至少需要一张相关图片。")
    if decision in {"no_answer", "excluded"} and relevant_ids:
        raise ValueError("无答案或排除查询不能填写相关图片ID。")
    unknown = sorted(set(relevant_ids) - known_item_ids)
    if unknown:
        raise ValueError("未知图片ID：" + "、".join(unknown))
    row = {
        "query_id": query_id,
        "decision": decision,
        "relevant_item_ids": ";".join(relevant_ids),
        "evidence_scope": str(review.get("evidence_scope", "candidate_pool")),
        "reviewer_type": str(review.get("reviewer_type", "human")),
        "reviewer_id": str(review.get("reviewer_id", "local_user")),
        "review_confidence": str(review.get("review_confidence", "")),
        "human_notes": normalize_query(str(review.get("human_notes", ""))),
        "reviewed_at": utc_now(now),
    }
    reviews = read_reviews(study_dir)
    reviews[query_id] = row
    frozen_order = {
        source["query_id"]: index
        for index, source in enumerate(verify_frozen_snapshot(study_dir))
    }
    ordered = sorted(reviews.values(), key=lambda value: frozen_order[value["query_id"]])
    _write_csv(reviews_path(study_dir), ordered, REVIEW_FIELDS)
    protocol["review_count"] = len(reviews)
    _write_protocol(study_dir, protocol)


def materialize_formal_rows(
    study_dir: Path,
    *,
    minimum_eligible: int = DEFAULT_MINIMUM_COUNT,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    frozen = verify_frozen_snapshot(study_dir)
    reviews = read_reviews(study_dir)
    missing = [row["query_id"] for row in frozen if row["query_id"] not in reviews]
    if missing:
        raise ValueError(f"仍有 {len(missing)} 条查询未完成人工相关性审核。")
    protocol = load_protocol(study_dir)
    independently_authored = (
        protocol.get("query_origin", "user_authored") == "user_authored"
    )
    formal: list[dict[str, Any]] = []
    for source in frozen:
        review = reviews[source["query_id"]]
        if review["decision"] == "excluded":
            continue
        formal.append(
            {
                "query_id": source["query_id"],
                "query": source["query"],
                "query_type": (
                    "independent_blind"
                    if independently_authored
                    else "assistant_diagnostic"
                ),
                "is_no_answer": str(review["decision"] == "no_answer").lower(),
                "relevant_item_ids": review["relevant_item_ids"],
                "split": "test",
                "review_status": "approved",
                "reviewer_type": review.get("reviewer_type", "human"),
                "provenance": (
                    "independently_authored_then_frozen_and_pooled"
                    if independently_authored
                    else "assistant_generated_then_frozen_and_pooled"
                ),
            }
        )
    if len(formal) < minimum_eligible:
        raise ValueError(
            f"排除后仅剩 {len(formal)} 条，少于正式评测要求的 "
            f"{minimum_eligible} 条。"
        )
    protocol["status"] = "completed"
    protocol["review_count"] = len(reviews)
    protocol["eligible_query_count"] = len(formal)
    protocol["completed_at"] = utc_now(now)
    _write_protocol(study_dir, protocol)
    return formal


def write_formal_query_set(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "query_id",
        "query",
        "query_type",
        "is_no_answer",
        "relevant_item_ids",
        "split",
        "review_status",
        "reviewer_type",
        "provenance",
    ]
    _write_csv(path, rows, fields)


def evaluator_query_set_sha256(rows: Iterable[dict[str, Any]]) -> str:
    """Match the canonical query fingerprint used by the live evaluator."""
    canonical = [
        {
            "query_id": str(row["query_id"]),
            "query": normalize_query(str(row["query"])),
            "query_type": str(row["query_type"]),
            "is_no_answer": str(row["is_no_answer"]).lower()
            in {"1", "true", "yes", "y"},
            "relevant_item_ids": sorted(
                split_item_ids(row.get("relevant_item_ids", ""))
            ),
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


def write_live_evaluation_protocol(
    path: Path,
    *,
    study_dir: Path,
    formal_rows: list[dict[str, Any]],
    query_file: str,
    library_dir: str,
    output: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    study = load_protocol(study_dir)
    policy = study.get("candidate_policy") or {}
    search_policy_version = policy.get("search_policy_version")
    if not search_policy_version:
        raise ValueError("本轮没有锁定候选检索策略版本。")
    timestamp = now or datetime.now(timezone.utc)
    independently_authored = (
        study.get("query_origin", "user_authored") == "user_authored"
    )
    protocol = {
        "name": f"{study['study_id']}_formal_evaluation",
        "created_date": timestamp.date().isoformat(),
        "seal_status": (
            "independently_authored_and_frozen_before_retrieval"
            if independently_authored
            else "assistant_generated_and_frozen_before_retrieval"
        ),
        "query_file": query_file,
        "query_set_sha256": evaluator_query_set_sha256(formal_rows),
        "split": "test",
        "required_review_status": "approved",
        "query_count": len(formal_rows),
        "library_dir": library_dir,
        "method": "quality_hybrid",
        "rerank_top_k": 0,
        "force_components": False,
        "expected_search_policy_version": int(search_policy_version),
        "query_timeout_seconds": 300,
        "output": output,
        "no_tuning_rule": (
            "Do not change queries, qrels, routing, weights, or thresholds "
            "between this export and the formal run."
        ),
        "metrics": [
            "ranker_recall_at_1/3/5/10",
            "ranker_mrr",
            "ranker_ndcg_at_10",
            "answerable_acceptance_rate",
            "no_answer_rejection_accuracy",
            "product_recall_at_1/3/5/10",
            "product_mrr",
            "product_ndcg_at_10",
            "end_to_end_top1_accuracy",
            "latency_p50/p95",
        ],
        "limitations": [
            "Relevance labels come from a pooled top-k of three retrieval routes; relevant documents outside the pool may remain unjudged.",
            "The workflow supports one local human assessor and does not measure inter-annotator agreement.",
            "Author answerability expectations are balance metadata only and are hidden from the assessor.",
            *(
                []
                if independently_authored
                else [
                    "Queries were generated by the assistant and this run is diagnostic, not an independently authored user benchmark."
                ]
            ),
        ],
    }
    _atomic_json(path, protocol)
    return protocol


def progress_summary(study_dir: Path) -> dict[str, Any]:
    protocol = load_protocol(study_dir)
    submissions = read_submissions(
        study_dir, frozen=protocol["status"] != "collecting"
    )
    expectations = {value: 0 for value in sorted(VALID_EXPECTATIONS)}
    for row in submissions:
        expectation = row.get("expected_answerability", "unsure")
        expectations[expectation] = expectations.get(expectation, 0) + 1
    return {
        "status": protocol["status"],
        "submitted": len(submissions),
        "target": int(protocol["target_count"]),
        "maximum": int(protocol["maximum_count"]),
        "candidates": len(read_candidates(study_dir)),
        "reviews": len(read_reviews(study_dir)),
        "expectations": expectations,
    }
