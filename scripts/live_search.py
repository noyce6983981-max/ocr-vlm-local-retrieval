"""Run an arbitrary text query through both retrieval branches."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.demo_backend import (  # noqa: E402
    build_method_scores,
    build_rrf_scores,
    normalize_rows,
)
from scripts.bm25_retrieval import bm25_query_weight  # noqa: E402
from scripts.color_retrieval import color_query_scores  # noqa: E402
from scripts.evaluate_library_retrieval import (  # noqa: E402
    apply_open_set_gate,
    gate_features,
)
from scripts.query_routing import (  # noqa: E402
    detect_color_intent,
    extract_quoted_terms,
    extract_strict_entity_term,
    extract_temporal_personal_terms,
    extract_topic_evidence_terms,
    expand_visual_query,
    infer_retrieval_route,
    is_pure_color_query,
)
from scripts.retrieval_rejection import (  # noqa: E402
    build_acceptance_decisions,
    infer_query_mode,
)


TEXT_PYTHON = PROJECT_ROOT / ".venv/Scripts/python.exe"
VISUAL_PYTHON = PROJECT_ROOT / ".venv-vl/Scripts/python.exe"
RERANKER_PYTHON = VISUAL_PYTHON
DEFAULT_LIBRARY_DIR = PROJECT_ROOT / "outputs/user_library"
CACHE_DIR = PROJECT_ROOT / "outputs/live_cache"
SEARCH_POLICY_VERSION = 16
MAX_STORED_RESULTS_PER_METHOD = 100
LOW_CONFIDENCE_RESULT_LIMIT = 20
FINAL_CACHE_MAX_FILES_PER_LIBRARY = 512
FINAL_CACHE_MAX_BYTES_PER_LIBRARY = 512 * 1024 * 1024
COMPONENT_CACHE_MAX_FILES = 4096
COMPONENT_CACHE_MAX_BYTES = 2 * 1024 * 1024 * 1024
SELECTED_RETRIEVAL_CONFIG = (
    PROJECT_ROOT / "config/selected_retrieval_config_v16.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--method",
        choices=(
            "text",
            "visual",
            "quality_hybrid",
            "reranker",
        ),
        default="quality_hybrid",
    )
    parser.add_argument(
        "--library-dir",
        type=Path,
        default=DEFAULT_LIBRARY_DIR,
        help="Search only this isolated document library.",
    )
    parser.add_argument(
        "--rerank-top-k",
        type=int,
        default=0,
        help="Rerank this many adaptive-fusion candidates; 0 disables it.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_json_object(path: Path) -> dict[str, Any] | None:
    """Return a JSON object or treat a partial/corrupt cache as a miss."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def write_json_atomic(path: Path, payload: Any) -> None:
    """Replace a cache only after a complete JSON file reaches disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def prune_json_cache(
    directory: Path,
    *,
    protected: set[Path] | None = None,
    max_files: int,
    max_bytes: int,
) -> dict[str, int]:
    """Bound a generated JSON cache by count and bytes, newest first."""
    if not directory.is_dir():
        return {"deleted_files": 0, "deleted_bytes": 0}
    protected_paths = {
        path.resolve() for path in (protected or set()) if path.exists()
    }
    candidates: list[tuple[Path, int, int]] = []
    kept_files = 0
    kept_bytes = 0
    for path in directory.glob("*.json"):
        try:
            stat = path.stat()
        except OSError:
            continue
        if path.resolve() in protected_paths:
            kept_files += 1
            kept_bytes += stat.st_size
            continue
        candidates.append((path, stat.st_size, stat.st_mtime_ns))
    candidates.sort(key=lambda row: row[2], reverse=True)
    deleted_files = 0
    deleted_bytes = 0
    for path, size, _ in candidates:
        if kept_files < max_files and kept_bytes + size <= max_bytes:
            kept_files += 1
            kept_bytes += size
            continue
        try:
            path.unlink()
        except OSError:
            continue
        deleted_files += 1
        deleted_bytes += size
    return {
        "deleted_files": deleted_files,
        "deleted_bytes": deleted_bytes,
    }


def prune_final_cache_if_needed(output_path: Path) -> None:
    try:
        output_path.resolve().relative_to(CACHE_DIR.resolve())
    except ValueError:
        return
    if output_path.parent.resolve() == (CACHE_DIR / "components").resolve():
        return
    prune_json_cache(
        output_path.parent,
        protected={output_path},
        max_files=FINAL_CACHE_MAX_FILES_PER_LIBRARY,
        max_bytes=FINAL_CACHE_MAX_BYTES_PER_LIBRARY,
    )


def reranker_cache_signature(
    item_ids: list[str], exploratory_query: bool
) -> str:
    """Version policy-dependent reranking separately from base embeddings."""
    mode = "exploratory" if exploratory_query else "exact"
    material = (
        f"policy={SEARCH_POLICY_VERSION}\nmode={mode}\n"
        + "\n".join(item_ids)
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:12]


def library_revision(
    library_dir: Path = DEFAULT_LIBRARY_DIR,
) -> str:
    """Hash searchable content while ignoring display-only category edits."""
    library_dir = library_dir.resolve()
    manifest_path = library_dir / "manifest.jsonl"
    if not manifest_path.is_file():
        return hashlib.sha256(
            str(library_dir).encode("utf-8")
        ).hexdigest()[:12]
    revision_paths = [
        library_dir / "text_index/index.faiss",
        library_dir / "bm25_index/index.json.gz",
        library_dir / "visual_index/embeddings.npy",
        library_dir / "metadata_index/index.faiss",
    ]
    overrides_dir = library_dir / "ocr/overrides"
    if overrides_dir.is_dir():
        revision_paths.extend(sorted(overrides_dir.glob("*.json")))
    digest = hashlib.sha256()
    digest.update(str(library_dir).encode("utf-8"))
    for row in combined_manifest(library_dir):
        digest.update(str(row["item_id"]).encode("utf-8"))
        digest.update(str(row.get("source_path", "")).encode("utf-8"))
    for path in revision_paths:
        if not path.is_file():
            continue
        try:
            revision_name = path.relative_to(library_dir).as_posix()
        except ValueError:
            revision_name = path.relative_to(PROJECT_ROOT).as_posix()
        digest.update(revision_name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def retrieval_config_revision() -> str:
    """Hash ranking policy separately so model component caches stay reusable."""
    if not SELECTED_RETRIEVAL_CONFIG.is_file():
        return "fallback"
    return hashlib.sha256(
        SELECTED_RETRIEVAL_CONFIG.read_bytes()
    ).hexdigest()[:12]


def selected_runtime_config(
    library_dir: Path = DEFAULT_LIBRARY_DIR,
) -> dict[str, Any]:
    """Load the reviewed ranker only for the library it was trained on."""
    fallback = {
        "ranker": {
            "family": "adaptive",
            "params": {
                "text_weight_cap": 0.6,
                "sparse_scale": 1.0,
            },
        },
        "open_set_gate": None,
        "reranker_gate": None,
        "source": "fallback",
    }
    if (
        library_dir.resolve() != DEFAULT_LIBRARY_DIR.resolve()
        or not SELECTED_RETRIEVAL_CONFIG.is_file()
    ):
        return fallback
    payload = json.loads(
        SELECTED_RETRIEVAL_CONFIG.read_text(encoding="utf-8")
    )
    ranker = payload.get("ranker", {})
    if ranker.get("family") not in {"adaptive", "query_aware"}:
        return fallback
    return {
        "ranker": ranker,
        "open_set_gate": payload.get("open_set_gate"),
        "reranker_gate": payload.get("reranker_gate"),
        "source": SELECTED_RETRIEVAL_CONFIG.relative_to(
            PROJECT_ROOT
        ).as_posix(),
    }


def query_key(query: str, revision: str | None = None) -> str:
    revision = revision if revision is not None else library_revision()
    material = f"{revision}\n{query}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()[:16]


def required_search_branches(
    method: str,
    retrieval_route: str,
    exploratory_query: bool,
) -> dict[str, bool]:
    """Choose evidence branches without bypassing a calibrated gate."""
    if method == "text":
        return {"text": True, "bm25": False, "visual": False}
    if method == "visual":
        return {"text": False, "bm25": False, "visual": True}
    if retrieval_route == "entity_exact":
        return {"text": True, "bm25": True, "visual": False}
    if retrieval_route == "visual_discovery":
        return {"text": True, "bm25": True, "visual": True}
    if retrieval_route == "text_evidence":
        # Explicit OCR/document lookups already have dense text and sparse
        # keyword evidence. Loading the visual model adds large cold-start
        # latency without independently calibrated evidence for this route.
        return {"text": True, "bm25": True, "visual": False}
    if method == "quality_hybrid" and not exploratory_query:
        # The reviewed open-set gate was trained on all three signals.
        return {"text": True, "bm25": True, "visual": True}
    # Precision mode has its own reranker gate. Discovery queries select
    # branches by topic type and are ranked without factual-answer rejection.
    return {
        "text": retrieval_route != "visual_discovery",
        "bm25": retrieval_route
        in {"text_evidence", "topic_discovery", "mixed"},
        "visual": retrieval_route
        in {"visual_metadata", "visual_discovery", "mixed"},
    }


COMPOSITE_VISUAL_MARKERS = (
    "和",
    "与",
    "以及",
    "同时",
    "并且",
    "、",
    "旁",
    "旁边",
    "戴着",
    "戴",
    "搭着",
    "放着",
    "摆放",
    "上写着",
    "下的",
    "飞越",
    "游过",
    "停在",
    "站在",
    "穿过",
    "驶过",
    "跃过",
    "降落",
    "围着",
    "顶着",
    "堆成",
    "坐在",
    "举着",
    "趴在",
    "装进",
    "挂在",
)

STRONG_RELATIONAL_VISUAL_MARKERS = (
    "戴着",
    "搭着",
    "放着",
    "摆放",
    "飞越",
    "游过",
    "停在",
    "站在",
    "穿过",
    "驶过",
    "跃过",
    "降落",
    "围着",
    "顶着",
    "堆成",
    "坐在",
    "举着",
    "趴在",
    "装进",
    "挂在",
)

VISUAL_LITERAL_EVIDENCE_TERMS = (
    "餐厅菜单",
    "价目表",
    "通知",
    "英文",
    "代码",
    "写着",
    "写有",
)


def is_composite_visual_query(query: str, retrieval_route: str) -> bool:
    """Detect visual requests that require more than one pictured concept."""
    if retrieval_route not in {
        "mixed",
        "visual_metadata",
        "visual_discovery",
    }:
        return False
    normalized = "".join(query.split())
    return len(normalized) >= 8 and any(
        marker in normalized for marker in COMPOSITE_VISUAL_MARKERS
    )


def apply_composite_visual_guard(
    query: str,
    retrieval_route: str,
    ranking: list[dict[str, Any]],
    decision: dict[str, Any],
    ranker_params: dict[str, Any],
    gate_source: str,
    complete_literal_evidence_ids: set[str] | None = None,
) -> bool:
    """Reject composite visual answers supported only by weak text overlap."""
    if not is_composite_visual_query(query, retrieval_route):
        return False
    complete_literal_evidence_ids = complete_literal_evidence_ids or set()
    if any(
        row.get("item_id") in complete_literal_evidence_ids
        for row in ranking[:10]
    ):
        return False
    normalized_query = "".join(query.split())
    strong_relation = any(
        marker in normalized_query
        for marker in STRONG_RELATIONAL_VISUAL_MARKERS
    )
    threshold = float(
        ranker_params.get(
            "strong_relation_visual_min_score",
            ranker_params.get("composite_visual_min_score", 0.35),
        )
        if strong_relation
        else ranker_params.get("composite_visual_min_score", 0.35)
    )
    top_visual = max(
        (
            float(row.get("raw_visual_score", 0.0))
            for row in ranking[:10]
        ),
        default=0.0,
    )
    decision["composite_visual_guard_applied"] = True
    decision["composite_visual_signal"] = round(top_visual, 6)
    decision["composite_visual_threshold"] = round(threshold, 6)
    if not decision.get("accepted", False) or top_visual >= threshold:
        return False
    decision.update(
        {
            "accepted": False,
            "reason": (
                "查询要求多个画面概念同时成立，但候选的原始视觉"
                f"相似度最高仅 {top_visual:.3f}，低于复合视觉门槛 "
                f"{threshold:.3f}；不会只凭其中一个 OCR 词接收结果。"
            ),
            "signal_name": "composite_visual_raw_cosine",
            "signal": round(top_visual, 6),
            "threshold": round(threshold, 6),
            "gate_source": gate_source,
        }
    )
    return True


def apply_route_acceptance_guard(
    query: str,
    retrieval_route: str,
    ranking: list[dict[str, Any]],
    decision: dict[str, Any],
    ranker_params: dict[str, Any],
    gate_source: str,
    quoted_evidence_ids: set[str] | None = None,
) -> bool:
    """Require route-aligned evidence from the same candidate.

    The historical mixed gate took the maximum from each branch independently,
    so unrelated pages could jointly make a query look answerable.  V16 keeps
    the fast branches but evaluates evidence aligned on individual candidates.
    """
    top_rows = ranking[:10]
    quoted_evidence_ids = quoted_evidence_ids or set()
    quoted_matches = [
        row
        for row in top_rows
        if row.get("item_id") in quoted_evidence_ids
    ]
    quoted_terms = extract_quoted_terms(query)
    if (
        retrieval_route in {"text_evidence", "mixed"}
        and quoted_terms
        and not quoted_matches
    ):
        decision.update(
            {
                "accepted": False,
                "reason": (
                    "查询明确给出了引号文字，但前10项没有任何资料在"
                    "同一候选中完整包含全部引文，因此不返回近似答案。"
                ),
                "signal_name": "complete_quoted_evidence_count",
                "signal": 0,
                "threshold": 1,
                "gate_source": gate_source,
            }
        )
        return True
    if not top_rows:
        decision.update(
            {
                "accepted": False,
                "reason": "没有可用候选，系统不会返回推测结果。",
                "signal_name": "route_aligned_evidence_count",
                "signal": 0,
                "threshold": 1,
                "gate_source": gate_source,
            }
        )
        return True

    if retrieval_route == "text_evidence":
        dense_threshold = float(
            ranker_params.get("text_evidence_min_dense_score", 0.50)
        )
        bm25_threshold = float(
            ranker_params.get("text_evidence_min_bm25_score", 4.0)
        )
        aligned = [
            row
            for row in top_rows
            if float(row.get("raw_text_score", 0.0)) >= dense_threshold
            and float(row.get("bm25_raw_score", 0.0)) >= bm25_threshold
        ]
        accepted = bool(aligned or quoted_matches)
        decision.update(
            {
                "accepted": accepted,
                "reason": (
                    "文字证据必须在同一候选上同时满足语义和关键词门槛；"
                    f"前10项有 {len(aligned)} 项满足联合门槛、"
                    f"{len(quoted_matches)} 项精确命中引号证据，"
                    "两者至少满足一种。"
                ),
                "signal_name": "aligned_text_bm25_candidate_count",
                "signal": len(aligned) + len(quoted_matches),
                "threshold": 1,
                "dense_threshold": round(dense_threshold, 6),
                "bm25_threshold": round(bm25_threshold, 6),
                "gate_source": gate_source,
            }
        )
        return True

    if retrieval_route == "visual_metadata":
        visual_threshold = float(
            ranker_params.get("visual_metadata_min_visual_score", 0.34)
        )
        visual_rows = [
            row
            for row in top_rows
            if float(row.get("raw_visual_score", 0.0)) >= visual_threshold
        ]
        requires_literal = (
            "手写" not in query
            and any(term in query for term in VISUAL_LITERAL_EVIDENCE_TERMS)
        )
        literal_rows = [
            row
            for row in visual_rows
            if float(row.get("raw_text_score", 0.0))
            >= float(ranker_params.get("visual_text_min_dense_score", 0.50))
            and float(row.get("bm25_raw_score", 0.0))
            >= float(ranker_params.get("visual_text_min_bm25_score", 4.0))
        ]
        accepted = bool(
            quoted_matches
            or (literal_rows if requires_literal else visual_rows)
        )
        signal = len(literal_rows if requires_literal else visual_rows)
        evidence_name = (
            "视觉与文字联合证据" if requires_literal else "视觉与元数据联合证据"
        )
        decision.update(
            {
                "accepted": accepted,
                "reason": (
                    f"{evidence_name}必须落在同一候选；前10项有 {signal} "
                    f"项通过，另有 {len(quoted_matches)} 项精确命中"
                    "引号证据；两者至少满足一种。"
                ),
                "signal_name": (
                    "aligned_visual_text_candidate_count"
                    if requires_literal
                    else "visual_metadata_candidate_count"
                ),
                "signal": signal + len(quoted_matches),
                "threshold": 1,
                "visual_threshold": round(visual_threshold, 6),
                "gate_source": gate_source,
            }
        )
        return True
    return False


def apply_contextual_evidence_guard(
    query: str,
    ranking: list[dict[str, Any]],
    decision: dict[str, Any],
    contextual_evidence_ids: set[str],
    gate_source: str,
) -> bool:
    """Reject user-relative time constraints absent from indexed evidence."""
    contextual_terms = extract_temporal_personal_terms(query)
    if not contextual_terms:
        return False
    matched = [
        row["item_id"]
        for row in ranking[:10]
        if row.get("item_id") in contextual_evidence_ids
    ]
    if matched:
        return False
    decision.update(
        {
            "accepted": False,
            "reason": (
                "查询包含相对时间或个人上下文（"
                + "、".join(contextual_terms)
                + "），但前10项的OCR和元数据没有对应证据；系统不会"
                "只凭相似画面猜测。"
            ),
            "signal_name": "indexed_context_evidence_count",
            "signal": 0,
            "threshold": 1,
            "gate_source": gate_source,
        }
    )
    return True


def resolve_search_intent(
    method: str,
    query: str,
) -> tuple[str, bool, str, str | None]:
    """Route exact lookups strictly and browse queries recall-first."""
    route = infer_retrieval_route(query)
    strict_entity_term = (
        extract_strict_entity_term(query)
        if route == "entity_exact"
        else None
    )
    if strict_entity_term is not None:
        return route, False, query, strict_entity_term
    if route == "visual_discovery":
        return route, True, expand_visual_query(query), None
    if route == "topic_discovery":
        return route, True, query, None
    return route, False, query, None


def combined_manifest(
    library_dir: Path = DEFAULT_LIBRARY_DIR,
) -> list[dict[str, Any]]:
    manifest_path = library_dir / "manifest.jsonl"
    rows = [
        row
        for row in read_jsonl(manifest_path)
        if bool(row.get("search_enabled", True))
    ]
    item_ids = [row["item_id"] for row in rows]
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("Combined manifest contains duplicate item IDs.")
    return rows


def compact_exact_text(value: Any) -> str:
    """Normalize OCR and metadata for conservative exact-term matching."""
    normalized = unicodedata.normalize("NFKC", str(value)).casefold()
    return "".join(character for character in normalized if character.isalnum())


def exact_evidence_item_ids(
    term: str,
    library_dir: Path,
    manifest: list[dict[str, Any]],
) -> set[str]:
    """Find pages where a short entity literally occurs in indexed evidence."""
    needle = compact_exact_text(term)
    matched: set[str] = set()
    for row in manifest:
        searchable = " ".join(
            str(value)
            for key, value in row.items()
            if key != "item_id" and isinstance(value, (str, int, float))
        )
        if needle in compact_exact_text(searchable):
            matched.add(str(row["item_id"]))

    for relative_path in (
        "text_index/metadata.jsonl",
        "metadata_index/metadata.jsonl",
    ):
        path = library_dir / relative_path
        if not path.is_file():
            continue
        for row in read_jsonl(path):
            searchable = " ".join(
                str(value)
                for key, value in row.items()
                if key not in {"item_id", "chunk_id"}
                and isinstance(value, (str, int, float))
            )
            if needle in compact_exact_text(searchable):
                matched.add(str(row["item_id"]))
    return matched


def apply_strict_entity_policy(
    term: str,
    evidence_item_ids: set[str],
    rankings: dict[str, list[dict[str, Any]]],
    acceptance: dict[str, dict[str, Any]],
    gate_source: str,
) -> None:
    """Never substitute visually or semantically similar pages for a name."""
    if evidence_item_ids:
        for method, rows in rankings.items():
            rankings[method] = [
                row
                for row in rows
                if row["item_id"] in evidence_item_ids
            ]
            acceptance[method] = {
                "accepted": True,
                "query_mode": "entity_exact",
                "reason": (
                    f"人名/短实体“{term}”在 OCR 文字或元数据中精确命中 "
                    f"{len(evidence_item_ids)} 页；仅展示这些页面。"
                ),
                "signal_name": "exact_entity_evidence_count",
                "signal": len(evidence_item_ids),
                "threshold": 1,
                "gate_source": gate_source,
            }
        return

    for method in rankings:
        acceptance[method] = {
            "accepted": False,
            "query_mode": "entity_exact",
            "reason": (
                f"人名/短实体“{term}”未在 OCR 文字或元数据中精确出现；"
                "系统不会用相似人物或自然图像代替。"
            ),
            "signal_name": "exact_entity_evidence_count",
            "signal": 0,
            "threshold": 1,
            "gate_source": gate_source,
        }


def discovery_acceptance_decision(
    retrieval_route: str,
    ranking: list[dict[str, Any]],
    exact_topic_evidence_count: int,
    ranker_params: dict[str, Any],
    gate_source: str,
    color_intent: str | None = None,
) -> dict[str, Any]:
    """Apply a low but non-zero relevance floor to browse-style queries."""
    top_rows = ranking[:10]
    if retrieval_route == "visual_discovery":
        if color_intent:
            signal = max(
                (float(row.get("color_score", 0.0)) for row in top_rows),
                default=0.0,
            )
            threshold = float(
                ranker_params.get("color_discovery_min_coverage", 0.18)
            )
            signal_name = "color_coverage_relevance"
            reason = (
                "颜色查询同时使用视觉语义和真实像素覆盖率；"
                f"当前目标颜色覆盖率 {signal:.3f}，门槛 {threshold:.3f}。"
            )
        else:
            signal = max(
                (
                    float(row.get("raw_visual_score", 0.0))
                    for row in top_rows
                ),
                default=0.0,
            )
            threshold = float(
                ranker_params.get("visual_discovery_min_score", 0.35)
            )
            signal_name = "visual_discovery_relevance"
            reason = (
                "视觉主题采用召回优先的三路融合，同时保留最低相关性门槛；"
                f"当前视觉相关分 {signal:.3f}，门槛 {threshold:.3f}。"
            )
        accepted = signal >= threshold
    else:
        signal = max(
            (float(row.get("raw_text_score", 0.0)) for row in top_rows),
            default=0.0,
        )
        threshold = float(
            ranker_params.get("topic_discovery_min_dense_score", 0.44)
        )
        accepted = exact_topic_evidence_count > 0 or signal >= threshold
        reason = (
            f"主题查询精确命中 {exact_topic_evidence_count} 页，"
            f"语义相关分 {signal:.3f}，最低门槛 {threshold:.3f}；"
            "未达门槛时默认不把弱相似内容当作结果。"
        )
        signal_name = "topic_discovery_relevance"
    return {
        "accepted": accepted,
        "query_mode": retrieval_route,
        "reason": reason,
        "signal_name": signal_name,
        "signal": round(signal, 6),
        "threshold": round(threshold, 6),
        "exact_topic_evidence_count": exact_topic_evidence_count,
        "gate_source": gate_source,
    }


def filter_discovery_rankings(
    retrieval_route: str,
    rankings: dict[str, list[dict[str, Any]]],
    exact_topic_ids: set[str],
    ranker_params: dict[str, Any],
    color_intent: str | None,
    pure_color_query: bool,
) -> dict[str, Any]:
    """Remove weak individual results after recall and before display."""
    summary: dict[str, Any] = {
        "applied": False,
        "route": retrieval_route,
        "methods": {},
    }
    if retrieval_route not in {"visual_discovery", "topic_discovery"}:
        return summary
    for method in ("quality_hybrid", "reranker"):
        rows = rankings.get(method)
        if rows is None:
            continue
        before_count = len(rows)
        cutoff = 0.0
        if retrieval_route == "topic_discovery":
            minimum = float(
                ranker_params.get("topic_result_min_dense_score", 0.44)
            )
            best = max(
                (float(row.get("raw_text_score", 0.0)) for row in rows),
                default=0.0,
            )
            cutoff = max(
                minimum,
                best
                - float(
                    ranker_params.get("topic_result_relative_margin", 0.12)
                ),
            )
            rankings[method] = [
                row
                for row in rows
                if row["item_id"] in exact_topic_ids
                or float(row.get("raw_text_score", 0.0)) >= cutoff
            ]
        elif color_intent and pure_color_query:
            minimum = float(
                ranker_params.get("color_result_min_coverage", 0.18)
            )
            best = max(
                (float(row.get("color_score", 0.0)) for row in rows),
                default=0.0,
            )
            cutoff = max(
                minimum,
                best
                - float(
                    ranker_params.get("color_result_relative_margin", 0.35)
                ),
            )
            rankings[method] = [
                row
                for row in rows
                if float(row.get("color_score", 0.0)) >= cutoff
            ]
        elif color_intent:
            color_threshold = float(
                ranker_params.get("color_object_min_coverage", 0.12)
            )
            minimum_visual = float(
                ranker_params.get("color_object_min_visual_score", 0.30)
            )
            best_visual = max(
                (
                    float(row.get("raw_visual_score", 0.0))
                    for row in rows
                ),
                default=0.0,
            )
            visual_threshold = max(
                minimum_visual,
                best_visual
                - float(
                    ranker_params.get(
                        "color_object_visual_relative_margin",
                        0.10,
                    )
                ),
            )
            rankings[method] = [
                row
                for row in rows
                if float(row.get("color_score", 0.0)) >= color_threshold
                and float(row.get("raw_visual_score", 0.0))
                >= visual_threshold
            ]
            cutoff = visual_threshold
        else:
            minimum = float(
                ranker_params.get("visual_result_min_score", 0.35)
            )
            best = max(
                (
                    float(row.get("raw_visual_score", 0.0))
                    for row in rows
                ),
                default=0.0,
            )
            cutoff = max(
                minimum,
                best
                - float(
                    ranker_params.get("visual_result_relative_margin", 0.14)
                ),
            )
            rankings[method] = [
                row
                for row in rows
                if float(row.get("raw_visual_score", 0.0)) >= cutoff
            ]
        summary["methods"][method] = {
            "before": before_count,
            "after": len(rankings[method]),
            "cutoff": round(cutoff, 6),
        }
    summary["applied"] = bool(summary["methods"])
    return summary


def snapshot_low_confidence_candidates(
    rankings: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    """Keep a small pre-filter window for an explicit user reveal action."""
    return {
        method: [dict(row) for row in rows[:LOW_CONFIDENCE_RESULT_LIMIT]]
        for method, rows in rankings.items()
        if method in {"quality_hybrid", "reranker"}
    }


def truncate_rankings_for_output(
    rankings: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    """Persist enough UI/evaluation hits without duplicating the whole library."""
    return {
        method: rows[:MAX_STORED_RESULTS_PER_METHOD]
        for method, rows in rankings.items()
    }


def apply_route_ranking_policy(
    rankings: dict[str, list[dict[str, Any]]],
    retrieval_route: str,
    color_intent: str | None,
    has_exact_topic_evidence: bool,
    has_quoted_evidence: bool,
    ranker_params: dict[str, Any],
    query: str = "",
) -> str:
    """Select a precomputed ranking only when cross-dataset evidence agrees."""
    if retrieval_route == "visual_discovery" and color_intent:
        return "quality_hybrid"
    if retrieval_route == "visual_metadata" and has_quoted_evidence:
        return "quality_hybrid"
    source_by_route = ranker_params.get("route_ranking_sources", {})
    plain_visual_object_query = (
        retrieval_route == "visual_metadata"
        and bool(re.match(r"^(?:找|查找|搜索)一张", query.strip()))
        and query.strip("。！？? ").endswith(("照片", "图片", "图像"))
        and not extract_quoted_terms(query)
        and not any(marker in query for marker in COMPOSITE_VISUAL_MARKERS)
    )
    if plain_visual_object_query:
        source_key = "visual_metadata_plain_object"
    elif (
        retrieval_route == "topic_discovery"
        and not has_exact_topic_evidence
    ):
        source_key = "topic_discovery_without_exact"
    else:
        source_key = retrieval_route
    source = str(source_by_route.get(source_key, "quality_hybrid"))
    if source == "quality_hybrid" or source not in rankings:
        return "quality_hybrid"
    rankings["quality_hybrid"] = [dict(row) for row in rankings[source]]
    return source


def promote_complete_quoted_evidence(
    rankings: dict[str, list[dict[str, Any]]],
    quoted_evidence_ids: set[str],
) -> None:
    """Place complete literal matches before approximate candidates."""
    if not quoted_evidence_ids:
        return
    for method in ("quality_hybrid", "reranker"):
        rows = rankings.get(method)
        if not rows:
            continue
        exact = [
            row for row in rows if row.get("item_id") in quoted_evidence_ids
        ]
        approximate = [
            row for row in rows if row.get("item_id") not in quoted_evidence_ids
        ]
        rankings[method] = exact + approximate


def read_ocr_rows(
    library_dir: Path = DEFAULT_LIBRARY_DIR,
) -> list[dict[str, str]]:
    path = library_dir / "ocr/summary.csv"
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def ocr_text(
    item_id: str,
    library_dir: Path = DEFAULT_LIBRARY_DIR,
) -> str:
    override_path = library_dir / "ocr/overrides" / f"{item_id}.json"
    if override_path.is_file():
        payload = json.loads(override_path.read_text(encoding="utf-8"))
        return "\n".join(
            str(value) for value in payload.get("rec_texts", [])
        )
    for directory in (library_dir / "ocr/json",):
        path = directory / f"{item_id}.json"
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            return "\n".join(
                str(value) for value in payload.get("rec_texts", [])
            )
    return ""


def add_reranker_ranking(
    rankings: dict[str, list[dict[str, Any]]],
    reranker_payload: dict[str, Any],
) -> None:
    """Add reranked Top-K followed by the remaining hybrid candidates."""
    hybrid = rankings.get("quality_hybrid") or rankings["adaptive"]
    hybrid_by_id = {row["item_id"]: row for row in hybrid}
    score_by_id = dict(
        zip(reranker_payload["item_ids"], reranker_payload["scores"])
    )
    reranked_ids = sorted(
        score_by_id,
        key=lambda item_id: float(score_by_id[item_id]),
        reverse=True,
    )
    reranked_set = set(reranked_ids)
    reranked_rows = [
        {
            **hybrid_by_id[item_id],
            "score": round(float(score_by_id[item_id]), 6),
            "reranker_score": round(float(score_by_id[item_id]), 6),
        }
        for item_id in reranked_ids
    ]
    reranked_rows.extend(
        {
            **row,
            "reranker_score": None,
        }
        for row in hybrid
        if row["item_id"] not in reranked_set
    )
    rankings["reranker"] = reranked_rows


def run_branch(command: list[str], label: str) -> None:
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout)[-1600:]
        raise RuntimeError(f"{label} branch failed:\n{detail}")


def align_scores(
    payload: dict[str, Any], item_ids: list[str]
) -> np.ndarray:
    source_ids = payload["item_ids"]
    source_scores = payload["scores"]
    score_by_id = dict(zip(source_ids, source_scores))
    return np.array(
        [float(score_by_id[item_id]) for item_id in item_ids],
        dtype=np.float32,
    )


def align_score_field(
    payload: dict[str, Any],
    item_ids: list[str],
    field: str,
) -> np.ndarray:
    source_ids = payload["item_ids"]
    source_scores = payload[field]
    score_by_id = dict(zip(source_ids, source_scores))
    return np.array(
        [float(score_by_id[item_id]) for item_id in item_ids],
        dtype=np.float32,
    )


def cached_query_matches(
    path: Path, query: str, revision: str | None = None
) -> bool:
    if not path.is_file():
        return False
    payload = read_json_object(path)
    if payload is None:
        return False
    if payload.get("query") != query:
        return False
    return revision is None or payload.get("library_revision") == revision


def cached_visual_query_matches(
    path: Path,
    query: str,
    encoded_query: str,
    revision: str | None = None,
) -> bool:
    """Validate both the visible query and the text encoded by the VLM."""
    if not cached_query_matches(path, query, revision):
        return False
    payload = read_json_object(path)
    if payload is None:
        return False
    return payload.get("encoded_query", query) == encoded_query


def main() -> None:
    args = parse_args()
    query = " ".join(args.query.split())
    if not query:
        raise ValueError("Query cannot be empty.")
    if args.rerank_top_k < 0:
        raise ValueError("--rerank-top-k cannot be negative.")

    library_dir = (
        args.library_dir
        if args.library_dir.is_absolute()
        else PROJECT_ROOT / args.library_dir
    ).resolve()
    manifest_path = library_dir / "manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "当前资料库还是空的，请先上传并完成入库。"
        )
    revision = library_revision(library_dir)
    config_revision = retrieval_config_revision()
    (
        retrieval_route,
        exploratory_query,
        visual_query,
        strict_entity_term,
    ) = resolve_search_intent(args.method, query)
    color_intent = detect_color_intent(query)
    pure_color_query = is_pure_color_query(query)
    key = query_key(query, revision)
    output_path = (
        args.output
        if args.output is not None
        else CACHE_DIR / f"{key}.json"
    )
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    if output_path.is_file() and not args.force:
        cached = read_json_object(output_path)
        if cached is not None and (
            cached.get("query") == query
            and cached.get("library_revision") == revision
            and cached.get("rerank_top_k", 0) == args.rerank_top_k
            and cached.get("requested_method", "quality_hybrid")
            == args.method
            and cached.get("search_policy_version")
            == SEARCH_POLICY_VERSION
            and cached.get("retrieval_config_revision")
            == config_revision
            and cached.get("retrieval_route") == retrieval_route
            and cached.get("visual_query", query) == visual_query
        ):
            print(output_path)
            return

    started = time.perf_counter()
    manifest = combined_manifest(library_dir)
    item_ids = [row["item_id"] for row in manifest]
    runtime_config = selected_runtime_config(library_dir)
    exact_entity_ids = (
        exact_evidence_item_ids(
            strict_entity_term,
            library_dir,
            manifest,
        )
        if strict_entity_term
        else set()
    )
    topic_evidence_terms = (
        extract_topic_evidence_terms(query)
        if retrieval_route == "topic_discovery"
        else []
    )
    topic_evidence_counts: dict[str, int] = {}
    for term in topic_evidence_terms:
        for item_id in exact_evidence_item_ids(term, library_dir, manifest):
            topic_evidence_counts[item_id] = (
                topic_evidence_counts.get(item_id, 0) + 1
            )
    literal_evidence_ids = set(topic_evidence_counts)
    exact_topic_ids = (
        literal_evidence_ids
        if retrieval_route == "topic_discovery"
        else set()
    )
    contextual_terms = extract_temporal_personal_terms(query)
    contextual_evidence_ids: set[str] = set()
    for term in contextual_terms:
        contextual_evidence_ids.update(
            exact_evidence_item_ids(term, library_dir, manifest)
        )
    quoted_terms = extract_quoted_terms(query)
    quoted_term_matches = [
        exact_evidence_item_ids(term, library_dir, manifest)
        for term in quoted_terms
    ]
    quoted_evidence_ids = (
        set.intersection(*quoted_term_matches)
        if quoted_term_matches
        else set()
    )
    if strict_entity_term and not exact_entity_ids:
        rankings: dict[str, list[dict[str, Any]]] = {
            args.method: []
        }
        acceptance: dict[str, dict[str, Any]] = {}
        apply_strict_entity_policy(
            strict_entity_term,
            exact_entity_ids,
            rankings,
            acceptance,
            runtime_config["source"],
        )
        payload = {
            "query": query,
            "search_policy_version": SEARCH_POLICY_VERSION,
            "retrieval_config_revision": config_revision,
            "requested_method": args.method,
            "query_mode": "entity_exact",
            "retrieval_route": retrieval_route,
            "visual_query": visual_query,
            "strict_entity_term": strict_entity_term,
            "color_intent": color_intent,
            "pure_color_query": pure_color_query,
            "exact_entity_evidence_item_ids": [],
            "exact_topic_evidence_item_ids": [],
            "topic_evidence_terms": topic_evidence_terms,
            "contextual_evidence_terms": contextual_terms,
            "quoted_evidence_terms": quoted_terms,
            "library_dir": library_dir.relative_to(PROJECT_ROOT).as_posix(),
            "cache_key": key,
            "library_revision": revision,
            "rerank_top_k": args.rerank_top_k,
            "executed_branches": {
                "text": False,
                "bm25": False,
                "visual": False,
                "reranker": False,
                "color": False,
            },
            "bm25_query_weight": 0.0,
            "retrieval_config": runtime_config,
            "timings": {
                "text_seconds": 0.0,
                "bm25_seconds": 0.0,
                "visual_seconds": 0.0,
                "reranker_seconds": 0.0,
                "total_seconds": round(time.perf_counter() - started, 3),
            },
            "rankings": rankings,
            "acceptance": acceptance,
        }
        write_json_atomic(output_path, payload)
        prune_final_cache_if_needed(output_path)
        print(output_path)
        return

    component_dir = CACHE_DIR / "components"
    text_path = component_dir / f"{key}_text.json"
    bm25_path = component_dir / f"{key}_bm25.json"
    visual_path = component_dir / f"{key}_visual.json"
    branches = required_search_branches(
        args.method,
        retrieval_route,
        exploratory_query,
    )
    needs_text = branches["text"]
    needs_bm25 = branches["bm25"]
    needs_visual = branches["visual"]
    text_cache_hit = (
        needs_text
        and not args.force
        and cached_query_matches(text_path, query, revision)
    )
    bm25_cache_hit = (
        needs_bm25
        and not args.force
        and cached_query_matches(bm25_path, query, revision)
    )
    visual_cache_hit = (
        needs_visual
        and not args.force
        and cached_visual_query_matches(
            visual_path, query, visual_query, revision
        )
    )
    component_cache_hits: dict[str, bool | None] = {
        "text": text_cache_hit if needs_text else None,
        "bm25": bm25_cache_hit if needs_bm25 else None,
        "visual": visual_cache_hit if needs_visual else None,
        "reranker": None,
    }
    active_component_paths = {text_path, bm25_path, visual_path}

    if needs_text and not text_cache_hit:
        run_branch(
            [
                str(TEXT_PYTHON),
                str(PROJECT_ROOT / "scripts/score_text_query.py"),
                query,
                "--manifest",
                str(manifest_path),
                "--index-dir",
                str(library_dir / "text_index"),
                "--metadata-index-dir",
                str(library_dir / "metadata_index"),
                "--output",
                str(text_path),
                "--library-revision",
                revision,
            ],
            "text",
        )
    if needs_bm25 and not bm25_cache_hit:
        run_branch(
            [
                str(TEXT_PYTHON),
                str(PROJECT_ROOT / "scripts/score_bm25_query.py"),
                query,
                "--manifest",
                str(manifest_path),
                "--index-dir",
                str(library_dir / "bm25_index"),
                "--output",
                str(bm25_path),
                "--library-revision",
                revision,
            ],
            "bm25",
        )
    if needs_visual and not visual_cache_hit:
        run_branch(
            [
                str(VISUAL_PYTHON),
                str(PROJECT_ROOT / "scripts/score_visual_query.py"),
                query,
                "--encoded-query",
                visual_query,
                "--index-dir",
                str(library_dir / "visual_index"),
                "--output",
                str(visual_path),
                "--library-revision",
                revision,
            ],
            "visual",
        )

    empty_scores = np.zeros(len(item_ids), dtype=np.float32)
    text_payload = (
        json.loads(text_path.read_text(encoding="utf-8"))
        if needs_text
        else {"elapsed_seconds": 0.0}
    )
    bm25_payload = (
        json.loads(bm25_path.read_text(encoding="utf-8"))
        if needs_bm25
        else {"elapsed_seconds": 0.0}
    )
    visual_payload = (
        json.loads(visual_path.read_text(encoding="utf-8"))
        if needs_visual
        else {"elapsed_seconds": 0.0}
    )
    text_scores = (
        align_scores(text_payload, item_ids)
        if needs_text
        else empty_scores.copy()
    )
    bm25_scores = (
        align_scores(bm25_payload, item_ids)
        if needs_bm25
        else empty_scores.copy()
    )
    visual_scores = (
        align_scores(visual_payload, item_ids)
        if needs_visual
        else empty_scores.copy()
    )
    metadata_scores = (
        align_score_field(
            text_payload, item_ids, "metadata_scores"
        )
        if needs_text and "metadata_scores" in text_payload
        else text_scores.copy()
    )
    color_scores = color_query_scores(
        color_intent,
        library_dir,
        manifest,
    )

    ocr_rows = read_ocr_rows(library_dir)
    confidence_by_id = {
        row["item_id"]: float(row["mean_confidence"]) for row in ocr_rows
    }
    confidences = np.array(
        [confidence_by_id.get(item_id, 0.0) for item_id in item_ids],
        dtype=np.float32,
    )
    ranker_params = runtime_config["ranker"].get("params", {})
    adaptive_text_weight_cap = float(
        ranker_params.get("text_weight_cap", 0.6)
    )
    sparse_scale = float(ranker_params.get("sparse_scale", 1.0))
    method_scores, text_weights = build_method_scores(
        text_scores[None, :],
        visual_scores[None, :],
        confidences,
        adaptive_text_weight_cap=adaptive_text_weight_cap,
    )
    rrf_scores, rrf_branches = build_rrf_scores(
        text_scores,
        bm25_scores,
        visual_scores,
        confidences,
    )
    normalized_bm25_scores = normalize_rows(
        bm25_scores[None, :]
    )[0]
    normalized_metadata_scores = normalize_rows(
        metadata_scores[None, :]
    )[0]
    query_mode = infer_query_mode(query)
    ranker_family = runtime_config["ranker"].get(
        "family", "adaptive"
    )
    sparse_weight = sparse_scale * bm25_query_weight(query)
    if ranker_family == "query_aware":
        if retrieval_route in {"text_evidence", "entity_exact"}:
            sparse_weight = float(
                ranker_params.get("text_bm25_weight", 0.25)
            )
            quality_hybrid_scores = (
                method_scores["text"][0]
                + sparse_weight * normalized_bm25_scores
            )
        elif retrieval_route == "visual_metadata":
            visual_weight = float(
                ranker_params.get("visual_weight", 0.25)
            )
            sparse_weight = 0.0
            quality_hybrid_scores = (
                visual_weight * method_scores["visual"][0]
                + (1.0 - visual_weight)
                * normalized_metadata_scores
            )
        elif retrieval_route == "visual_discovery":
            weight_key = (
                "pure_color_discovery_weights"
                if color_intent and pure_color_query
                else "color_object_discovery_weights"
                if color_intent
                else "visual_discovery_weights"
            )
            discovery_weights = ranker_params.get(weight_key, {})
            visual_discovery_weight = float(
                discovery_weights.get(
                    "visual",
                    0.42
                    if color_intent and pure_color_query
                    else 0.60
                    if color_intent
                    else 0.78,
                )
            )
            text_discovery_weight = float(
                discovery_weights.get(
                    "text", 0.07 if color_intent else 0.15
                )
            )
            bm25_discovery_weight = float(
                discovery_weights.get(
                    "bm25", 0.03 if color_intent else 0.07
                )
            )
            color_discovery_weight = float(
                discovery_weights.get(
                    "color",
                    0.48
                    if color_intent and pure_color_query
                    else 0.30
                    if color_intent
                    else 0.0,
                )
            )
            sparse_weight = bm25_discovery_weight
            quality_hybrid_scores = (
                visual_discovery_weight * method_scores["visual"][0]
                + text_discovery_weight * method_scores["text"][0]
                + bm25_discovery_weight * normalized_bm25_scores
                + color_discovery_weight * color_scores
            )
        elif retrieval_route == "topic_discovery":
            quality_hybrid_scores = (
                method_scores["text"][0]
                + sparse_weight * normalized_bm25_scores
            )
        else:
            quality_hybrid_scores = (
                method_scores["adaptive"][0]
                + sparse_weight * normalized_bm25_scores
            )
    else:
        quality_hybrid_scores = (
            method_scores["adaptive"][0]
            + sparse_weight * normalized_bm25_scores
        )

    if retrieval_route == "topic_discovery" and topic_evidence_counts:
        exact_topic_boost = float(
            ranker_params.get("exact_topic_evidence_boost", 0.20)
        )
        exact_topic_counts = np.asarray(
            [topic_evidence_counts.get(item_id, 0) for item_id in item_ids],
            dtype=np.float32,
        )
        quality_hybrid_scores = (
            quality_hybrid_scores
            + exact_topic_boost * np.minimum(exact_topic_counts, 2.0)
        )

    manifest_by_id = {row["item_id"]: row for row in manifest}
    rankings: dict[str, list[dict[str, Any]]] = {}
    visual_metadata_weights = ranker_params.get(
        "visual_metadata_weights", {"metadata": 0.5, "visual": 0.5}
    )
    visual_metadata_blend_scores = (
        float(visual_metadata_weights.get("metadata", 0.5))
        * normalized_metadata_scores
        + float(visual_metadata_weights.get("visual", 0.5))
        * method_scores["visual"][0]
    )
    score_vectors = {
        **{
            method: matrix[0]
            for method, matrix in method_scores.items()
        },
        "bm25": normalized_bm25_scores,
        "metadata": normalized_metadata_scores,
        "visual_metadata_blend": visual_metadata_blend_scores,
        **rrf_scores,
        "quality_hybrid": quality_hybrid_scores,
    }
    for method, scores in score_vectors.items():
        order = np.argsort(-scores)
        rankings[method] = [
            {
                **manifest_by_id[item_ids[index]],
                "score": round(float(scores[index]), 6),
                "text_score": round(
                    float(method_scores["text"][0, index]), 6
                ),
                "visual_score": round(
                    float(method_scores["visual"][0, index]), 6
                ),
                "bm25_score": round(
                    float(normalized_bm25_scores[index]), 6
                ),
                "metadata_score": round(
                    float(normalized_metadata_scores[index]), 6
                ),
                "bm25_raw_score": round(
                    float(bm25_scores[index]), 6
                ),
                "raw_text_score": round(
                    float(text_scores[index]), 6
                ),
                "raw_visual_score": round(
                    float(visual_scores[index]), 6
                ),
                "raw_metadata_score": round(
                    float(metadata_scores[index]), 6
                ),
                "color_score": round(
                    float(color_scores[index]), 6
                ),
                "color_intent": color_intent,
                "pure_color_query": pure_color_query,
                "retrieval_route": retrieval_route,
                "exact_topic_match_count": topic_evidence_counts.get(
                    item_ids[index], 0
                ),
                "literal_evidence_match_count": topic_evidence_counts.get(
                    item_ids[index], 0
                ),
                "dense_rrf": round(
                    float(rrf_branches["dense_rrf"][index]), 8
                ),
                "bm25_rrf": round(
                    float(rrf_branches["bm25_rrf"][index]), 8
                ),
                "visual_rrf": round(
                    float(rrf_branches["visual_rrf"][index]), 8
                ),
                "ocr_confidence": round(float(confidences[index]), 6),
                "text_weight": round(float(text_weights[index]), 6),
                "bm25_query_weight": sparse_weight,
            }
            for index in order
        ]

    route_ranking_source = apply_route_ranking_policy(
        rankings,
        retrieval_route,
        color_intent,
        bool(exact_topic_ids),
        bool(quoted_evidence_ids),
        ranker_params,
        query=query,
    )

    reranker_payload: dict[str, Any] | None = None
    if args.rerank_top_k:
        candidate_count = min(args.rerank_top_k, len(item_ids))
        metadata_rows_path = (
            library_dir / "metadata_index/metadata.jsonl"
        )
        metadata_text_by_id = (
            {
                row["item_id"]: row.get("metadata_text", "")
                for row in read_jsonl(metadata_rows_path)
            }
            if metadata_rows_path.is_file()
            else {}
        )
        candidates = [
            {
                "item_id": row["item_id"],
                "source_path": row["source_path"],
                "ocr_text": ocr_text(row["item_id"], library_dir),
                "metadata_text": metadata_text_by_id.get(
                    row["item_id"], ""
                ),
            }
            for row in rankings["quality_hybrid"][:candidate_count]
        ]
        candidate_signature = reranker_cache_signature(
            [candidate["item_id"] for candidate in candidates],
            exploratory_query,
        )
        candidate_path = (
            component_dir
            / (
                f"{key}_rerank_candidates_{candidate_count}_"
                f"{candidate_signature}.json"
            )
        )
        reranker_path = (
            component_dir
            / (
                f"{key}_reranker_{candidate_count}_"
                f"{candidate_signature}.json"
            )
        )
        write_json_atomic(candidate_path, candidates)
        active_component_paths.update({candidate_path, reranker_path})
        reranker_cache_hit = (
            not args.force
            and cached_query_matches(reranker_path, query, revision)
        )
        component_cache_hits["reranker"] = reranker_cache_hit
        if not reranker_cache_hit:
            reranker_command = [
                str(RERANKER_PYTHON),
                str(PROJECT_ROOT / "scripts/rerank_candidates.py"),
                query,
                "--candidates",
                str(candidate_path),
                "--output",
                str(reranker_path),
                "--library-revision",
                revision,
            ]
            if exploratory_query:
                reranker_command.append("--exploratory")
            run_branch(reranker_command, "reranker")
        reranker_payload = json.loads(
            reranker_path.read_text(encoding="utf-8")
        )
        add_reranker_ranking(rankings, reranker_payload)

    promote_complete_quoted_evidence(rankings, quoted_evidence_ids)

    low_confidence_rankings = snapshot_low_confidence_candidates(rankings)
    result_filter = filter_discovery_rankings(
        retrieval_route,
        rankings,
        exact_topic_ids,
        ranker_params,
        color_intent,
        pure_color_query,
    )

    acceptance = build_acceptance_decisions(query, rankings)
    open_set_gate = runtime_config.get("open_set_gate")
    if open_set_gate and needs_text and needs_bm25 and needs_visual:
        accepted_values, acceptance_probabilities = apply_open_set_gate(
            open_set_gate,
            gate_features(
                text_scores[None, :],
                bm25_scores[None, :],
                visual_scores[None, :],
            ),
        )
        probability = float(acceptance_probabilities[0])
        threshold = float(open_set_gate["threshold"])
        decision = {
            "accepted": bool(accepted_values[0]),
            "query_mode": query_mode,
            "reason": (
                f"人工真值训练的开放集门控概率 {probability:.3f}，"
                f"判定阈值 {threshold:.3f}。"
            ),
            "signal_name": "reviewed_open_set_probability",
            "signal": round(probability, 6),
            "threshold": round(threshold, 6),
            "gate_source": runtime_config["source"],
        }
        if exploratory_query:
            decision = {
                "accepted": True,
                "query_mode": query_mode,
                "reason": (
                    "探索型查询用于发现相似内容，不执行事实型无答案"
                    f"硬拒答；当前开放集参考概率 {probability:.3f}。"
                ),
                "signal_name": "exploratory_open_set_reference",
                "signal": round(probability, 6),
                "threshold": None,
                "gate_source": runtime_config["source"],
            }
        acceptance["quality_hybrid"] = decision
        if "reranker" in rankings:
            acceptance["reranker"] = dict(decision)
    reranker_gate = runtime_config.get("reranker_gate")
    if reranker_payload is not None and reranker_gate:
        reranker_top_score = max(
            float(value) for value in reranker_payload["scores"]
        )
        reranker_threshold = float(reranker_gate["threshold"])
        acceptance["reranker"] = {
            "accepted": (
                True
                if exploratory_query
                else reranker_top_score >= reranker_threshold
            ),
            "query_mode": (
                retrieval_route if exploratory_query else query_mode
            ),
            "reason": (
                (
                    "浏览型查询采用相关性排序；"
                    f"当前精排相关分 {reranker_top_score:.3f}。"
                )
                if exploratory_query
                else (
                    f"多模态精排完整匹配分 {reranker_top_score:.3f}，"
                    f"人工真值校准阈值 {reranker_threshold:.3f}。"
                )
            ),
            "signal_name": (
                "qwen3_vl_exploratory_relevance"
                if exploratory_query
                else "qwen3_vl_reranker_score"
            ),
            "signal": round(reranker_top_score, 6),
            "threshold": (
                None
                if exploratory_query
                else round(reranker_threshold, 6)
            ),
            "gate_source": runtime_config["source"],
        }
    if exploratory_query:
        for discovery_method in ("quality_hybrid", "reranker"):
            if discovery_method not in rankings:
                continue
            acceptance[discovery_method] = discovery_acceptance_decision(
                retrieval_route,
                rankings[discovery_method],
                len(exact_topic_ids),
                ranker_params,
                runtime_config["source"],
                color_intent,
            )
    for guarded_method in ("quality_hybrid", "reranker"):
        if guarded_method not in rankings:
            continue
        guarded_decision = acceptance.setdefault(
            guarded_method,
            {
                "accepted": False,
                "query_mode": query_mode,
                "reason": "没有可用候选。",
                "signal_name": "none",
                "signal": 0.0,
                "threshold": 0.0,
            },
        )
        guarded_ranking = rankings[guarded_method]
        apply_route_acceptance_guard(
            query,
            retrieval_route,
            guarded_ranking,
            guarded_decision,
            ranker_params,
            runtime_config["source"],
            quoted_evidence_ids,
        )
        apply_contextual_evidence_guard(
            query,
            guarded_ranking,
            guarded_decision,
            contextual_evidence_ids,
            runtime_config["source"],
        )
        apply_composite_visual_guard(
            query,
            retrieval_route,
            guarded_ranking,
            guarded_decision,
            ranker_params,
            runtime_config["source"],
            quoted_evidence_ids,
        )
    if strict_entity_term:
        apply_strict_entity_policy(
            strict_entity_term,
            exact_entity_ids,
            rankings,
            acceptance,
            runtime_config["source"],
        )
    rankings = truncate_rankings_for_output(rankings)
    payload = {
        "query": query,
        "search_policy_version": SEARCH_POLICY_VERSION,
        "retrieval_config_revision": config_revision,
        "requested_method": args.method,
        "query_mode": query_mode,
        "retrieval_route": retrieval_route,
        "route_ranking_source": route_ranking_source,
        "visual_query": visual_query,
        "strict_entity_term": strict_entity_term,
        "color_intent": color_intent,
        "pure_color_query": pure_color_query,
        "composite_visual_query": is_composite_visual_query(
            query, retrieval_route
        ),
        "exact_entity_evidence_item_ids": sorted(exact_entity_ids),
        "exact_topic_evidence_item_ids": sorted(exact_topic_ids),
        "literal_evidence_item_ids": sorted(literal_evidence_ids),
        "topic_evidence_terms": topic_evidence_terms,
        "contextual_evidence_terms": contextual_terms,
        "contextual_evidence_item_ids": sorted(contextual_evidence_ids),
        "quoted_evidence_terms": quoted_terms,
        "quoted_evidence_item_ids": sorted(quoted_evidence_ids),
        "library_dir": library_dir.relative_to(PROJECT_ROOT).as_posix(),
        "cache_key": key,
        "library_revision": revision,
        "rerank_top_k": args.rerank_top_k,
        "executed_branches": {
            "text": needs_text,
            "bm25": needs_bm25,
            "visual": needs_visual,
            "reranker": reranker_payload is not None,
            "color": color_intent is not None,
        },
        "bm25_query_weight": sparse_weight,
        "retrieval_config": runtime_config,
        "result_filter": result_filter,
        "low_confidence_rankings": low_confidence_rankings,
        "stored_result_limit_per_method": MAX_STORED_RESULTS_PER_METHOD,
        "component_cache_hits": component_cache_hits,
        "timings": {
            "text_seconds": text_payload["elapsed_seconds"],
            "bm25_seconds": bm25_payload["elapsed_seconds"],
            "visual_seconds": visual_payload["elapsed_seconds"],
            "reranker_seconds": (
                reranker_payload["elapsed_seconds"]
                if reranker_payload is not None
                else 0.0
            ),
            "total_seconds": round(time.perf_counter() - started, 3),
        },
        "rankings": rankings,
        "acceptance": acceptance,
    }
    write_json_atomic(output_path, payload)
    prune_final_cache_if_needed(output_path)
    prune_json_cache(
        component_dir,
        protected=active_component_paths,
        max_files=COMPONENT_CACHE_MAX_FILES,
        max_bytes=COMPONENT_CACHE_MAX_BYTES,
    )
    print(output_path)


if __name__ == "__main__":
    main()
