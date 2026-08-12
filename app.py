"""Streamlit browser for the reviewed multimodal retrieval experiment."""

from __future__ import annotations

import csv
import html
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
import pandas as pd
import streamlit as st

from scripts.apply_quality_reviews import publish_quality_reviews
from scripts.assistant_category_proposals import (
    read_assistant_proposals,
    resolve_assistant_proposal,
)
from scripts.blind_query_study import (
    assert_library_unchanged as assert_blind_library_unchanged,
)
from scripts.blind_query_study import (
    delete_submission as delete_blind_submission,
)
from scripts.blind_query_study import (
    freeze_study as freeze_blind_study,
)
from scripts.blind_query_study import (
    initialize_study as initialize_blind_study,
)
from scripts.blind_query_study import (
    load_protocol as load_blind_protocol,
)
from scripts.blind_query_study import (
    materialize_formal_rows as materialize_blind_formal_rows,
)
from scripts.blind_query_study import (
    progress_summary as blind_progress_summary,
)
from scripts.blind_query_study import (
    read_candidates as read_blind_candidates,
)
from scripts.blind_query_study import (
    read_reviews as read_blind_reviews,
)
from scripts.blind_query_study import (
    read_submissions as read_blind_submissions,
)
from scripts.blind_query_study import (
    save_relevance_review as save_blind_relevance_review,
)
from scripts.blind_query_study import (
    submit_query as submit_blind_query,
)
from scripts.blind_query_study import (
    verify_frozen_snapshot as verify_blind_snapshot,
)
from scripts.blind_query_study import (
    write_formal_query_set as write_blind_formal_query_set,
)
from scripts.blind_query_study import (
    write_live_evaluation_protocol as write_blind_evaluation_protocol,
)
from scripts.build_formal_query_set import build_formal_rows
from scripts.demo_backend import (
    align_score_matrix,
    build_method_scores,
)
from scripts.library_manager import (
    create_library,
    library_by_id,
    load_registry,
    manifest_count,
    resolve_library_dir,
    set_active_library,
)
from scripts.live_search import (
    SEARCH_POLICY_VERSION,
    cached_query_matches,
    library_revision,
    query_key,
    required_search_branches,
    resolve_search_intent,
    retrieval_config_revision,
    write_json_atomic,
)
from scripts.quality_review import (
    read_quality_reviews,
    save_quality_review,
    validate_quality_review,
)
from scripts.query_review import (
    read_reviews,
    save_review,
    validate_review,
)
from scripts.retrieval_explain import build_retrieval_explanation
from scripts.taxonomy import (
    CATEGORY_DESCRIPTIONS,
    CATEGORY_LABELS,
    QUALITY_LABELS,
    normalize_category,
    parse_quality_tags,
)
from scripts.v18_1_live_search import (
    INTENT_ROUTING_MODES,
)

PROJECT_ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = PROJECT_ROOT / "data/manifest/dataset_v1_manifest.jsonl"
QUERY_PATH = PROJECT_ROOT / "data/evaluation/dataset_v1_queries.csv"
OCR_SUMMARY_PATH = PROJECT_ROOT / "outputs/ocr_dataset_v1/summary.csv"
OCR_JSON_DIR = PROJECT_ROOT / "outputs/ocr_dataset_v1/json"
USER_LIBRARY_DIR = PROJECT_ROOT / "outputs/user_library"
USER_MANIFEST_PATH = USER_LIBRARY_DIR / "manifest.jsonl"
USER_OCR_SUMMARY_PATH = USER_LIBRARY_DIR / "ocr/summary.csv"
USER_OCR_JSON_DIR = USER_LIBRARY_DIR / "ocr/json"
UPLOAD_STAGING_DIR = PROJECT_ROOT / "outputs/upload_staging"
BATCH_JOBS_DIR = PROJECT_ROOT / "outputs/batch_jobs"
TEXT_SCORES_PATH = (
    PROJECT_ROOT / "outputs/evaluation_dataset_v1/text_score_matrix.npz"
)
VISUAL_SCORES_PATH = (
    PROJECT_ROOT / "outputs/evaluation_dataset_v1/visual_score_matrix.npz"
)
TEXT_REPORT_PATH = (
    PROJECT_ROOT / "outputs/evaluation_dataset_v1/text_retrieval.json"
)
VISUAL_REPORT_PATH = (
    PROJECT_ROOT / "outputs/evaluation_dataset_v1/visual_retrieval.json"
)
FUSION_REPORT_PATH = (
    PROJECT_ROOT / "outputs/evaluation_dataset_v1/fusion_retrieval.json"
)
LIVE_CACHE_DIR = PROJECT_ROOT / "outputs/live_cache"
QUERY_REVIEW_QUEUE_PATH = (
    PROJECT_ROOT
    / "data/evaluation/"
    "public_dataset_1500_retrieval_query_queue_100.csv"
)
QUERY_HUMAN_REVIEWS_PATH = (
    PROJECT_ROOT
    / "data/evaluation/"
    "public_dataset_1500_retrieval_query_human_reviews.csv"
)
BLIND_STUDY_DIR = PROJECT_ROOT / "data/evaluation/blind_study_v1"
BLIND_FORMAL_QUERY_PATH = BLIND_STUDY_DIR / "formal_queries.csv"
BLIND_EVALUATION_PROTOCOL_PATH = BLIND_STUDY_DIR / "evaluation_protocol.json"
BLIND_EVALUATION_OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs/evaluation/library_retrieval/blind_study_v1_formal.json"
)
BLIND_MODEL_HARD_QUEUE_PATH = (
    BLIND_STUDY_DIR / "model_review_final_unresolved.json"
)


def v19_intent_runtime_settings() -> tuple[str, str, float]:
    """Read the explicit, default-off routing mode for the UI."""

    mode = os.getenv("OCR_VLM_INTENT_ROUTING_MODE", "").strip().lower()
    if not mode:
        legacy_enabled = os.getenv(
            "OCR_VLM_ENABLE_V19_INTENT_ROUTING", ""
        ).strip().lower()
        mode = (
            "guarded"
            if legacy_enabled in {"1", "true", "yes", "on"}
            else "off"
        )
    if mode not in INTENT_ROUTING_MODES:
        raise ValueError(
            "OCR_VLM_INTENT_ROUTING_MODE must be off, shadow, guarded or active"
        )
    service_url = os.getenv(
        "OCR_VLM_V19_INTENT_ROUTING_URL",
        "http://127.0.0.1:8765",
    ).strip()
    if not service_url:
        raise ValueError("OCR_VLM_V19_INTENT_ROUTING_URL cannot be empty")
    raw_timeout = os.getenv("OCR_VLM_V19_INTENT_TIMEOUT_SECONDS", "2.0")
    timeout_seconds = float(raw_timeout)
    if timeout_seconds <= 0:
        raise ValueError("OCR_VLM_V19_INTENT_TIMEOUT_SECONDS must be positive")
    return mode, service_url, timeout_seconds


QUALITY_GATE_PATH = (
    PROJECT_ROOT
    / "data/evaluation/public_dataset_1500_quality_gate.csv"
)
QUALITY_HUMAN_REVIEWS_PATH = (
    PROJECT_ROOT
    / "data/evaluation/public_dataset_1500_quality_human_reviews.csv"
)
CATEGORY_REVIEW_SPLITS_PATH = (
    PROJECT_ROOT
    / "data/evaluation/public_dataset_1500_category_review_splits.csv"
)
ASSISTANT_CATEGORY_PROPOSALS_PATH = (
    PROJECT_ROOT
    / "data/evaluation/public_dataset_1500_category_assistant_proposals.csv"
)
CATEGORY_FEEDBACK_POLICY_PATH = (
    PROJECT_ROOT / "config/category_feedback_policy.json"
)
FEEDBACK_MODEL_DIR = USER_LIBRARY_DIR / "feedback_model"
FEEDBACK_REPORT_PATH = (
    FEEDBACK_MODEL_DIR / "feedback_training_report.json"
)
FEEDBACK_QUEUE_PATH = FEEDBACK_MODEL_DIR / "active_learning_queue.csv"

METHOD_LABELS = {
    "text": "OCR文本检索",
    "visual": "Qwen3-VL视觉检索",
    "fixed": "固定50/50融合",
    "adaptive": "OCR质量自适应融合",
}
LIVE_METHOD_LABELS = {
    "text": "OCR文字搜索",
    "visual": "图片语义搜索",
    "quality_hybrid": "智能混合搜索（快速）",
    "reranker": "多模态完整核验（较慢）",
}
BLIND_EXPECTATION_LABELS = {
    "不确定，让后续审核决定": "unsure",
    "我印象中资料库里有答案": "answerable",
    "我故意测试库中无答案": "no_answer_probe",
}


def correction_url(item_id: str) -> str:
    return (
        "http://127.0.0.1:8502/?item_id="
        + quote(item_id, safe="")
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_feedback_artifacts() -> tuple[dict[str, Any], list[dict[str, str]]]:
    report = (
        json.loads(FEEDBACK_REPORT_PATH.read_text(encoding="utf-8"))
        if FEEDBACK_REPORT_PATH.is_file()
        else {}
    )
    queue = (
        read_csv_rows(FEEDBACK_QUEUE_PATH)
        if FEEDBACK_QUEUE_PATH.is_file()
        else []
    )
    return report, queue


def read_category_feedback_policy() -> dict[str, Any]:
    if not CATEGORY_FEEDBACK_POLICY_PATH.is_file():
        return {"override_threshold": 0.80}
    return json.loads(
        CATEGORY_FEEDBACK_POLICY_PATH.read_text(encoding="utf-8")
    )


def recent_batch_jobs(
    library_dir: Path,
    limit: int = 5,
) -> list[dict[str, Any]]:
    relative_library = library_dir.relative_to(PROJECT_ROOT).as_posix()
    rows: list[dict[str, Any]] = []
    if not BATCH_JOBS_DIR.is_dir():
        return rows
    for path in BATCH_JOBS_DIR.glob("*/status.json"):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if row.get("library_dir") != relative_library:
            continue
        progress_path = path.parent / "ocr_progress.json"
        if progress_path.is_file():
            try:
                row["ocr_progress"] = json.loads(
                    progress_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                row["ocr_progress"] = {}
        rows.append(row)
    rows.sort(key=lambda row: str(row.get("updated_at", "")), reverse=True)
    return rows[:limit]


def load_runtime_library(library_dir: Path) -> dict[str, Any]:
    manifest_path = library_dir / "manifest.jsonl"
    ocr_summary_path = library_dir / "ocr/summary.csv"
    all_manifest = (
        read_jsonl(manifest_path) if manifest_path.is_file() else []
    )
    manifest = [
        row
        for row in all_manifest
        if bool(row.get("search_enabled", True))
    ]
    ocr_rows = (
        read_csv_rows(ocr_summary_path)
        if ocr_summary_path.is_file()
        else []
    )
    return {
        "manifest": manifest,
        "all_manifest": all_manifest,
        "ocr_by_id": {row["item_id"]: row for row in ocr_rows},
    }


@st.cache_data(show_spinner=False)
def load_experiment() -> dict[str, Any]:
    required = (
        MANIFEST_PATH,
        QUERY_PATH,
        OCR_SUMMARY_PATH,
        TEXT_SCORES_PATH,
        VISUAL_SCORES_PATH,
        TEXT_REPORT_PATH,
        VISUAL_REPORT_PATH,
        FUSION_REPORT_PATH,
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "缺少实验产物：" + "、".join(str(path) for path in missing)
        )

    manifest = read_jsonl(MANIFEST_PATH)
    queries = read_csv_rows(QUERY_PATH)
    ocr_rows = read_csv_rows(OCR_SUMMARY_PATH)
    item_ids = [row["item_id"] for row in manifest]
    query_ids = [row["query_id"] for row in queries]

    ocr_by_id = {row["item_id"]: row for row in ocr_rows}
    confidences = np.array(
        [float(ocr_by_id[item_id]["mean_confidence"]) for item_id in item_ids],
        dtype=np.float32,
    )

    with np.load(TEXT_SCORES_PATH) as archive:
        text_scores = align_score_matrix(archive, query_ids, item_ids)
    with np.load(VISUAL_SCORES_PATH) as archive:
        visual_scores = align_score_matrix(archive, query_ids, item_ids)

    method_scores, text_weights = build_method_scores(
        text_scores, visual_scores, confidences
    )
    text_report = json.loads(TEXT_REPORT_PATH.read_text(encoding="utf-8"))
    visual_report = json.loads(
        VISUAL_REPORT_PATH.read_text(encoding="utf-8")
    )
    fusion_report = json.loads(
        FUSION_REPORT_PATH.read_text(encoding="utf-8")
    )
    metrics = {
        "text": text_report["overall"],
        "visual": visual_report["overall"],
        "fixed": fusion_report["fixed_fusion"]["metrics"],
        "adaptive": fusion_report["quality_adaptive_fusion"]["metrics"],
    }
    return {
        "manifest": manifest,
        "queries": queries,
        "item_ids": item_ids,
        "ocr_by_id": ocr_by_id,
        "method_scores": method_scores,
        "text_weights": text_weights,
        "metrics": metrics,
    }


def ocr_preview(
    item_id: str,
    max_chars: int = 220,
    library_dir: Path | None = None,
) -> str:
    text = full_ocr_text(item_id, library_dir=library_dir)
    return text[:max_chars] + ("…" if len(text) > max_chars else "")


def full_ocr_text(
    item_id: str,
    library_dir: Path | None = None,
) -> str:
    path = (
        library_dir / "ocr/json" / f"{item_id}.json"
        if library_dir is not None
        else OCR_JSON_DIR / f"{item_id}.json"
    )
    if not path.is_file():
        return ""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return " ".join(
        str(value) for value in payload.get("rec_texts", [])
    )


def highlight_evidence(text: str, terms: list[str]) -> str:
    highlighted = html.escape(text)
    for term in sorted(set(terms), key=len, reverse=True):
        escaped_term = html.escape(term)
        highlighted = re.sub(
            re.escape(escaped_term),
            lambda match: f"<mark>{match.group(0)}</mark>",
            highlighted,
            flags=re.IGNORECASE,
        )
    return highlighted


def render_retrieval_explanation(
    query: str,
    item_id: str,
    result_row: dict[str, Any],
    library_dir: Path,
) -> None:
    explanation = build_retrieval_explanation(
        query,
        full_ocr_text(item_id, library_dir=library_dir),
        result_row,
    )
    st.markdown(
        (
            '<div class="evidence-summary">'
            '<span class="evidence-icon">✓</span>'
            f'主要依据 · {html.escape(explanation["dominant_branch"])}'
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    with st.expander("查看检索依据"):
        st.caption(
            "分支归一化分数（仅用于本次查询）："
            f"OCR语义 {explanation['text_score']:.3f}　·　"
            f"关键词 {explanation['bm25_score']:.3f}　·　"
            f"视觉内容 {explanation['visual_score']:.3f}　·　"
            f"颜色覆盖 {explanation['color_score']:.3f}"
        )
        if explanation["matched_terms"]:
            st.write(
                "**OCR字面命中：** "
                + "、".join(explanation["matched_terms"])
            )
        if explanation["ocr_excerpt"]:
            st.markdown(
                '<div class="evidence-quote">'
                + highlight_evidence(
                    explanation["ocr_excerpt"],
                    explanation["matched_terms"],
                )
                + "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.caption(
                "OCR中没有直接字面命中，本结果主要依赖语义或视觉相似度。"
            )


@st.cache_resource(show_spinner=False)
def persistent_text_model_runtime() -> tuple[
    ThreadPoolExecutor,
    Future[Any],
    threading.Lock,
]:
    """Start loading BGE-M3 once and retain it across Streamlit reruns."""
    executor = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="bge-m3-loader",
    )

    def load_model() -> Any:
        from scripts.score_text_query import load_text_model

        return load_text_model(
            PROJECT_ROOT / "models/bge-m3",
            "cuda:0",
        )

    future = executor.submit(load_model)
    return executor, future, threading.Lock()


@st.cache_resource(show_spinner=False)
def persistent_text_resources(
    library_dir_value: str,
    revision: str,
) -> dict[str, Any]:
    """Keep FAISS text indexes resident for one library content revision."""
    del revision  # The value deliberately participates in the cache key.
    from scripts.score_text_query import load_text_resources

    library_dir = Path(library_dir_value)
    return load_text_resources(
        library_dir / "manifest.jsonl",
        library_dir / "text_index",
        metadata_index_dir=library_dir / "metadata_index",
    )


def ensure_persistent_text_component(
    query: str,
    library_dir: Path,
    revision: str,
    method_key: str,
) -> str:
    """Materialize text evidence with a resident model before CLI fusion."""
    route, exploratory, _, _ = resolve_search_intent(method_key, query)
    branches = required_search_branches(method_key, route, exploratory)
    if not branches["text"]:
        return "not_needed"
    component_path = (
        LIVE_CACHE_DIR
        / "components"
        / f"{query_key(query, revision)}_text.json"
    )
    if cached_query_matches(component_path, query, revision):
        return "component_cache_hit"

    _, model_future, model_lock = persistent_text_model_runtime()
    model = model_future.result(timeout=180)
    resources = persistent_text_resources(str(library_dir), revision)
    with model_lock:
        if cached_query_matches(component_path, query, revision):
            return "component_cache_hit"
        from scripts.score_text_query import score_text_query_payload

        payload = score_text_query_payload(
            query,
            model,
            resources,
            library_revision=revision,
        )
        write_json_atomic(component_path, payload)
    return "resident_model"


def execute_live_search(
    query: str,
    library_id: str,
    library_dir: Path,
    method_key: str = "quality_hybrid",
    rerank_top_k: int = 0,
) -> dict[str, Any]:
    normalized_query = " ".join(query.split())
    intent_routing_mode, v19_service_url, v19_timeout_seconds = (
        v19_intent_runtime_settings()
    )
    revision = library_revision(library_dir)
    cache_key = query_key(normalized_query, revision)
    suffix = f"_{method_key}"
    if rerank_top_k:
        suffix += f"_rerank{rerank_top_k}"
    if intent_routing_mode != "off":
        suffix += f"_intent_{intent_routing_mode}"
    output_path = (
        LIVE_CACHE_DIR
        / library_id
        / f"{cache_key}{suffix}.json"
    )
    final_cache_hit = False
    if output_path.is_file():
        try:
            expected_route, _, expected_visual_query, _ = resolve_search_intent(
                method_key,
                normalized_query,
            )
            cached_payload = json.loads(
                output_path.read_text(encoding="utf-8")
            )
            routing_audit = cached_payload.get("v18_1_intent_routing", {})
            common_cache_match = (
                cached_payload.get("search_policy_version")
                == SEARCH_POLICY_VERSION
                and cached_payload.get("retrieval_config_revision")
                == retrieval_config_revision()
                and cached_payload.get("requested_method") == method_key
            )
            if intent_routing_mode == "off":
                final_cache_hit = (
                    common_cache_match
                    and cached_payload.get("retrieval_route") == expected_route
                    and cached_payload.get("visual_query", normalized_query)
                    == expected_visual_query
                )
            else:
                final_cache_hit = (
                    common_cache_match
                    and isinstance(routing_audit, dict)
                    and routing_audit.get("routing_mode")
                    == intent_routing_mode
                )
        except (json.JSONDecodeError, OSError):
            final_cache_hit = False
    started = time.perf_counter()
    text_runtime_status = "not_needed"
    if not final_cache_hit:
        try:
            text_runtime_status = ensure_persistent_text_component(
                normalized_query,
                library_dir,
                revision,
                method_key,
            )
        except Exception:
            # The CLI path below remains the authoritative safe fallback.
            text_runtime_status = "fallback"
    command = [
        sys.executable,
        str(
            PROJECT_ROOT
            / (
                "scripts/v18_1_live_search.py"
                if intent_routing_mode != "off"
                else "scripts/live_search.py"
            )
        ),
        normalized_query,
        "--library-dir",
        str(library_dir),
        "--output",
        str(output_path),
        "--rerank-top-k",
        str(rerank_top_k),
        "--method",
        method_key,
    ]
    if intent_routing_mode != "off":
        command.extend(
            [
                "--intent-routing-mode",
                intent_routing_mode,
                "--v19-intent-routing-url",
                v19_service_url,
                "--v19-intent-timeout-seconds",
                str(v19_timeout_seconds),
            ]
        )
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
        detail = (completed.stderr or completed.stdout)[-1800:]
        raise RuntimeError(detail)
    result = json.loads(output_path.read_text(encoding="utf-8"))
    result["_ui_cache_hit"] = final_cache_hit
    result["_ui_response_seconds"] = round(
        time.perf_counter() - started, 3
    )
    result["_ui_text_runtime_status"] = text_runtime_status
    result["_ui_intent_routing_mode"] = intent_routing_mode
    result["_ui_v19_intent_routing_enabled"] = intent_routing_mode != "off"
    return result


def execute_batch_ingest(
    paths: list[Path],
    library_dir: Path,
    declared_ai_percent: float,
) -> dict[str, Any]:
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/batch_ingest.py"),
            *[str(path) for path in paths],
            "--library-dir",
            str(library_dir),
            "--declared-ai-percent",
            str(declared_ai_percent),
            "--max-pages",
            "1000",
            "--max-files",
            "1000",
            "--max-uncompressed-mb",
            "2048",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=10800,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout)[-3000:]
        raise RuntimeError(detail)
    output_lines = [
        line for line in completed.stdout.splitlines() if line.strip()
    ]
    if not output_lines:
        raise RuntimeError("批量入库脚本没有返回结果。")
    return json.loads(output_lines[-1])


def render_result_card(
    rank: int,
    item: dict[str, Any],
    score: float,
    expected_item_id: str,
    ocr_row: dict[str, str],
    text_weight: float,
    query: str,
    result_row: dict[str, Any],
    library_dir: Path | None = None,
) -> None:
    is_expected = item["item_id"] == expected_item_id
    badge = " · 真值目标" if is_expected else ""
    st.markdown(f"#### Top {rank}{badge}")
    image_path = PROJECT_ROOT / item["source_path"]
    if image_path.is_file():
        st.image(str(image_path), width="stretch")
    else:
        st.warning("本地原图未找到")
    st.markdown(f"**{item['display_name_zh']}**")
    st.caption(item["item_id"])
    st.caption(
        "当前类别："
        + CATEGORY_LABELS.get(
            normalize_category(str(item.get("category", ""))),
            str(item.get("category", "未分类")),
        )
    )
    st.metric("融合排序分", f"{score:.4f}")
    st.caption("该分数只用于本次候选排序，不代表相关概率。")
    left, right = st.columns(2)
    if result_row.get("color_intent"):
        left.metric(
            "目标颜色覆盖",
            f"{float(result_row.get('color_score', 0.0)):.1%}",
        )
        right.metric(
            "视觉语义原始分",
            f"{float(result_row.get('raw_visual_score', 0.0)):.3f}",
        )
    elif result_row.get("retrieval_route") == "visual_discovery":
        left.metric(
            "视觉语义原始分",
            f"{float(result_row.get('raw_visual_score', 0.0)):.3f}",
        )
        if result_row.get("reranker_score") is not None:
            right.metric(
                "精排相关分",
                f"{float(result_row['reranker_score']):.3f}",
            )
        else:
            right.metric(
                "视觉分支归一化",
                f"{float(result_row.get('visual_score', 0.0)):.3f}",
            )
    else:
        left.metric(
            "OCR置信度",
            f"{float(ocr_row.get('mean_confidence', 0.0)):.3f}",
        )
        right.metric("文本权重", f"{text_weight:.3f}")
    source_name = item.get("public_source_name") or item.get(
        "source", "本地上传"
    )
    source_filename = item.get("source_file_name", "")
    page_number = item.get("page_number", 1)
    page_count = item.get("page_count", 1)
    st.caption(
        f"证据来源：{source_name} · {source_filename}"
        + (
            f" · 第 {page_number}/{page_count} 页"
            if int(page_count or 1) > 1
            else ""
        )
    )
    with st.expander("查看来源与许可"):
        st.write(f"原文件：{source_filename or '未记录'}")
        st.write(f"资料来源：{source_name}")
        if item.get("public_source_url"):
            st.markdown(
                f"[打开公开来源]({item['public_source_url']})"
            )
        if item.get("license"):
            st.write(f"许可：{item['license']}")
        if item.get("license_url"):
            st.markdown(f"[查看许可]({item['license_url']})")
    with st.expander("查看OCR文本"):
        st.write(
            ocr_preview(
                item["item_id"], library_dir=library_dir
            )
            or "未识别到文字"
        )
    if library_dir is not None:
        render_retrieval_explanation(
            query,
            item["item_id"],
            result_row,
            library_dir,
        )
        st.link_button(
            "纠正此页分类",
            correction_url(item["item_id"]),
            width="stretch",
        )


def render_live_search(
    method_key: str,
    top_k: int,
    library: dict[str, Any],
    library_dir: Path,
) -> None:
    data = load_runtime_library(library_dir)
    st.subheader("开始搜索")
    st.markdown(
        (
            '<div class="search-summary">'
            f'<span class="search-count">{len(data["manifest"])} 页可检索</span>'
            f'<span>仅搜索“{html.escape(library["name"])}”</span>'
            '<span>本机运行</span>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    if not data["manifest"]:
        st.warning("当前资料库还是空的，请先切换到“上传并入库”。")
        return
    st.caption("直接描述你记得的文字、画面或版式，不必知道原文件名。")
    with st.expander("查看检索流程"):
        st.write(
            "系统先判断查询需要精确证据、主题浏览还是视觉发现，"
            "再按需组合语义、关键词和视觉召回；选择完整核验时，"
            "会对候选页面做第二次多模态重排。"
        )
    query = st.text_input(
        "输入检索问题",
        value="",
        placeholder="描述你要找的文字、画面或版式",
        max_chars=200,
    )
    search_clicked = st.button(
        "开始实时检索",
        type="primary",
        width="stretch",
    )
    if search_clicked:
        if not query.strip():
            st.warning("请输入检索内容。")
        else:
            try:
                with st.spinner(
                    "正在理解查询并运行所需的检索分支；"
                    "首次加载模型可能需要一些时间…"
                ):
                    if method_key == "reranker":
                        try:
                            result = execute_live_search(
                                query,
                                library["id"],
                                library_dir,
                                method_key="reranker",
                                rerank_top_k=10,
                            )
                        except (
                            RuntimeError,
                            subprocess.TimeoutExpired,
                        ) as precision_error:
                            result = execute_live_search(
                                query,
                                library["id"],
                                library_dir,
                                method_key="quality_hybrid",
                                rerank_top_k=0,
                            )
                            result["_precision_fallback"] = True
                            result["_precision_fallback_reason"] = str(
                                precision_error
                            )[-500:]
                        st.session_state["live_result"] = result
                    else:
                        st.session_state["live_result"] = (
                            execute_live_search(
                                query,
                                library["id"],
                                library_dir,
                                method_key=method_key,
                                rerank_top_k=0,
                            )
                        )
            except (RuntimeError, subprocess.TimeoutExpired) as error:
                st.error("实时检索失败。")
                st.code(str(error))

    result = st.session_state.get("live_result")
    expected_library_path = library_dir.relative_to(
        PROJECT_ROOT
    ).as_posix()
    if result and (
        result.get("library_dir") != expected_library_path
        or result.get("library_revision") != library_revision(library_dir)
    ):
        st.session_state.pop("live_result", None)
        result = None
    if not result:
        st.info(
            "这里不是预设问题下拉框。点击按钮后会针对你输入的新问题"
            "重新计算向量；相同问题再次查询会直接读取缓存。"
        )
        return
    if result.get("_precision_fallback"):
        st.warning(
            "多模态完整核验本次超时或运行失败，已自动回退到快速搜索，"
            "候选结果仍可正常查看。"
        )
        if method_key == "reranker":
            method_key = "quality_hybrid"
    if method_key not in result["rankings"]:
        st.info("已切换检索方法，请重新点击“开始实时检索”。")
        return

    if result.get("_ui_cache_hit"):
        st.success(
            f"查询完成：{result['query']}　｜　缓存命中　｜　"
            f"本次响应 {result['_ui_response_seconds']:.2f} 秒"
        )
        st.caption(
            f"页面中保存的原始模型计算耗时为 "
            f"{result['timings']['total_seconds']:.1f} 秒；"
            "缓存命中后不会重新运行模型。"
        )
    else:
        st.success(
            f"查询完成：{result['query']}　｜　模型新计算　｜　"
            f"本次耗时 {result['_ui_response_seconds']:.1f} 秒"
        )
    ranking = result["rankings"][method_key]
    decision = result.get("acceptance", {}).get(method_key)
    if decision:
        route_label = (
            "人名/短实体精确检索"
            if decision.get("query_mode") == "entity_exact"
            else "视觉主题发现"
            if decision.get("query_mode") == "visual_discovery"
            else "主题浏览检索"
            if decision.get("query_mode") == "topic_discovery"
            else "视觉描述查询"
            if decision.get("query_mode") == "visual"
            else "文本/混合查询"
        )
        branch_labels = {
            "text": "OCR语义",
            "bm25": "关键词",
            "visual": "视觉语义",
            "color": "颜色特征",
            "reranker": "多模态重排",
        }
        active_branches = [
            label
            for branch, label in branch_labels.items()
            if result.get("executed_branches", {}).get(branch)
        ]
        branch_text = " + ".join(active_branches) or "无模型分支"
        st.caption(
            f"查询理解：{route_label}　·　参与排序：{branch_text}　·　"
            f"{decision['reason']}"
        )
        filter_info = (
            result.get("result_filter", {})
            .get("methods", {})
            .get(method_key)
        )
        if filter_info and filter_info.get("after", 0) < filter_info.get(
            "before", 0
        ):
            st.caption(
                "结果级相关性过滤："
                f"从 {filter_info['before']} 个召回候选中保留 "
                f"{filter_info['after']} 个达到当前查询门槛的结果。"
            )
        if not decision.get("accepted", True):
            st.warning(
                "当前资料库中没有找到达到可靠阈值的精确答案。"
            )
            if decision.get("query_mode") == "entity_exact":
                st.caption(
                    "为避免把同姓人物、相似表格或自然图像误当答案，"
                    "人名检索没有精确文字证据时不展示相似候选。"
                )
                return
            show_low_confidence = st.checkbox(
                "手动查看低置信度相似候选（不代表确定命中）",
                key=(
                    f"show_low_confidence_{method_key}_"
                    f"{result.get('cache_key', '')}"
                ),
            )
            if not show_low_confidence:
                return
            ranking = result.get("low_confidence_rankings", {}).get(
                method_key,
                ranking,
            )
    manifest_by_id = {
        row["item_id"]: row for row in data["manifest"]
    }
    display_count = min(top_k, len(ranking))
    for start in range(0, display_count, 3):
        result_columns = st.columns(min(3, display_count - start))
        for offset, column in enumerate(result_columns):
            rank = start + offset + 1
            row = ranking[rank - 1]
            item_id = row["item_id"]
            with column:
                render_result_card(
                    rank=rank,
                    item=manifest_by_id[item_id],
                    score=float(row["score"]),
                    expected_item_id="",
                    ocr_row=data["ocr_by_id"][item_id],
                    text_weight=float(row["text_weight"]),
                    query=result["query"],
                    result_row=row,
                    library_dir=library_dir,
                )
                if method_key == "rrf":
                    st.caption(
                        f"RRF贡献：Dense {row.get('dense_rrf', 0.0):.4f} · "
                        f"BM25 {row.get('bm25_rrf', 0.0):.4f} · "
                        f"Image {row.get('visual_rrf', 0.0):.4f}"
                    )
                if method_key in {"quality_hybrid", "reranker"}:
                    st.caption(
                        f"本查询BM25校准权重 "
                        f"{row.get('bm25_query_weight', 0.0):.2f}"
                    )
                if row.get("reranker_score") is not None:
                    st.caption(
                        f"多模态重排 {row['reranker_score']:.3f}"
                    )


def render_upload(
    library: dict[str, Any], library_dir: Path
) -> None:
    st.subheader("批量上传并自动入库")
    st.info(
        f"本批文件只会写入“{library['name']}”，不会出现在其他资料库。"
    )
    st.write(
        "支持多张图片、ZIP、PDF、DOCX和PPTX。系统先安全解包并自动页面化，"
        "再让OCR、BGE-M3和Qwen3-VL各加载一次完成整批处理。"
    )
    jobs = recent_batch_jobs(library_dir)
    if jobs:
        phase_labels = {
            "preparing_documents": "文档转换",
            "deduplicating": "去重",
            "ocr": "OCR识别",
            "building_text_corpus": "文本整理",
            "building_text_index": "文本索引",
            "building_sparse_index": "BM25稀疏索引",
            "building_visual_index": "视觉索引",
            "committing_indexes": "提交索引",
            "completed": "已完成",
        }
        with st.expander("最近导入任务", expanded=False):
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "任务": row.get("job_id", ""),
                            "状态": row.get("status", ""),
                            "阶段": phase_labels.get(
                                str(row.get("phase", "")),
                                row.get("phase", ""),
                            ),
                            "新增页": row.get("new_pages", ""),
                            "总页数": row.get("total_pages", ""),
                            "OCR进度": (
                                f"{row.get('ocr_progress', {}).get('processed_pages', 0)}"
                                f"/{row.get('ocr_progress', {}).get('total_pages', 0)}"
                                if row.get("ocr_progress")
                                else ""
                            ),
                            "更新时间": row.get("updated_at", ""),
                        }
                        for row in jobs
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
    uploads = st.file_uploader(
        "选择文件，可多选；200张图片建议压成一个ZIP",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
            "bmp",
            "tif",
            "tiff",
            "zip",
            "pdf",
            "docx",
            "pptx",
        ],
        accept_multiple_files=True,
    )
    declared_ai_percent = st.number_input(
        "本批资料中AI生成内容比例（人工声明）",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=1.0,
        help=(
            "自动识别AI图片不可靠，因此由资料整理者确认来源。"
            "本项目从下一批开始强制不超过5%。"
        ),
    )
    total_bytes = sum(upload.size for upload in uploads)
    if uploads:
        st.info(
            f"已选择 {len(uploads)} 个上传文件，共 "
            f"{total_bytes / 1024**2:.1f} MB。文档转换后的总页数上限为1000。"
        )
    clicked = st.button(
        "开始批量转换、OCR并建立索引",
        type="primary",
        width="stretch",
    )
    if clicked:
        if not uploads:
            st.warning("请先选择文件。")
            return
        if declared_ai_percent > 5.0:
            st.error(
                "本项目规定新数据中AI生成内容不得超过5%，本批暂不入库。"
            )
            return
        if total_bytes > 1024 * 1024 * 1024:
            st.warning("当前一次上传请控制在1GB以内。")
            return
        staging_dir = (
            UPLOAD_STAGING_DIR / f"batch_{uuid.uuid4().hex[:12]}"
        )
        staging_dir.mkdir(parents=True, exist_ok=False)
        staging_paths: list[Path] = []
        for index, upload in enumerate(uploads):
            safe_name = Path(upload.name).name
            target = staging_dir / f"{index:04d}_{safe_name}"
            target.write_bytes(upload.getvalue())
            staging_paths.append(target)
        try:
            with st.spinner(
                "正在批量页面化并顺序运行三个模型。200页可能需要十几分钟，"
                "请保持此页面打开…"
            ):
                result = execute_batch_ingest(
                    staging_paths,
                    library_dir,
                    declared_ai_percent,
                )
        except (RuntimeError, subprocess.TimeoutExpired) as error:
            st.error("批量入库失败。")
            st.code(str(error))
            return
        finally:
            shutil.rmtree(staging_dir, ignore_errors=True)

        st.session_state.pop("live_result", None)
        if result["status"] == "duplicate":
            st.info("本批文件中的页面都已经入库，没有重复建立索引。")
        else:
            st.success(
                "批量入库完成，可以切换到“实时任意查询”搜索新资料。"
            )
            columns = st.columns(4)
            columns[0].metric("新增页", result["new_pages"])
            columns[1].metric("重复页", result["duplicate_pages"])
            columns[2].metric("图库总页数", result["total_pages"])
            columns[3].metric(
                "总耗时", f"{result['timings']['total_seconds']:.1f}s"
            )
            st.warning(
                f"还有 {result['pending_review']} 页需要人工审核名称、类别和OCR质量。"
            )


def render_library_manager(
    registry: dict[str, Any],
) -> None:
    st.subheader("资料库管理")
    st.write(
        "每个资料库都有独立的原图、OCR结果、文本索引和视觉索引。"
        "搜索和上传默认只作用于当前选中的库。"
    )
    privacy_labels = {
        "public": "公开研究库",
        "study": "学习/科研库",
        "private": "私人敏感库",
    }
    rows = []
    for library in registry["libraries"]:
        library_dir = resolve_library_dir(library)
        rows.append(
            {
                "资料库": library["name"],
                "类型": privacy_labels[library["privacy"]],
                "页数": manifest_count(library_dir),
                "本地目录": library["path"],
            }
        )
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        width="stretch",
    )
    st.warning(
        "所有库目前都只保存在本机并被Git忽略。私人敏感库也不会调用商业API。"
    )
    with st.form("create_library"):
        name = st.text_input(
            "新资料库名称",
            placeholder="例如：课程资料、科研论文、私人证件",
        )
        privacy_label = st.selectbox(
            "资料库类型",
            options=list(privacy_labels.values()),
            index=1,
        )
        submitted = st.form_submit_button(
            "创建独立资料库",
            type="primary",
            width="stretch",
        )
    if submitted:
        privacy = next(
            key
            for key, label in privacy_labels.items()
            if label == privacy_label
        )
        try:
            library = create_library(name, privacy)
        except ValueError as error:
            st.error(str(error))
        else:
            st.session_state["selected_library_id"] = library["id"]
            st.success(f"已创建“{library['name']}”。")
            st.rerun()


def render_quality_review(
    library: dict[str, Any],
    library_dir: Path,
    requested_item_id: str = "",
) -> None:
    st.subheader("图片分类与OCR纠错")
    with st.expander("查看主类别与质量标签规则"):
        st.write(
            "先只按页面内容选择一个主类别，再单独勾选画面质量。"
            "模糊成绩单仍是“证件/表格/表单/票据 + 模糊”，"
            "不能再归为“退化文档”。"
        )
        for category, label in CATEGORY_LABELS.items():
            st.markdown(
                f"**{label}：** {CATEGORY_DESCRIPTIONS[category]}"
            )
        st.caption(
            "“无明显质量问题”与其他质量问题互斥；"
            "不确定时可以暂不选择质量标签并在备注中说明。"
        )
    if library["id"] != "public_research_200":
        st.info(
            "当前资料库还没有生成质量路由报告。导入完成后可运行同一质量闸门，"
            "再在这里逐页审核。"
        )
        return
    if not QUALITY_GATE_PATH.is_file():
        st.warning("尚未生成质量路由报告。")
        return

    quality_rows = read_csv_rows(QUALITY_GATE_PATH)
    manifest = read_jsonl(library_dir / "manifest.jsonl")
    manifest_by_id = {row["item_id"]: row for row in manifest}
    ocr_summary_path = library_dir / "ocr/summary.csv"
    ocr_by_id = {
        row["item_id"]: row
        for row in (
            read_csv_rows(ocr_summary_path)
            if ocr_summary_path.is_file()
            else []
        )
    }
    reviews = read_quality_reviews(QUALITY_HUMAN_REVIEWS_PATH)
    assistant_proposals = read_assistant_proposals(
        ASSISTANT_CATEGORY_PROPOSALS_PATH
    )
    review_split_rows = (
        read_csv_rows(CATEGORY_REVIEW_SPLITS_PATH)
        if CATEGORY_REVIEW_SPLITS_PATH.is_file()
        else []
    )
    review_split_by_id = {
        row["item_id"]: row for row in review_split_rows
    }
    split_labels = {
        "train": "训练集",
        "validation": "验证集",
        "test": "测试集",
    }
    feedback_report, feedback_queue = read_feedback_artifacts()
    feedback_policy = read_category_feedback_policy()
    feedback_override_threshold = float(
        feedback_policy.get("override_threshold", 0.80)
    )
    feedback_by_id = {
        row["item_id"]: row for row in feedback_queue
    }
    model_conflict_ids = {
        row["item_id"]
        for row in feedback_queue
        if row.get("priority_reason") == "模型与当前类别冲突"
    }
    candidates_by_id: dict[str, dict[str, Any]] = {}
    for source_row in quality_rows:
        item_id = source_row["item_id"]
        if (
            source_row["quality_route"] == "pass"
            and item_id not in model_conflict_ids
        ):
            continue
        item = manifest_by_id.get(item_id, {})
        candidate = {
            **source_row,
            "category": normalize_category(
                str(
                    item.get(
                        "category",
                        source_row.get("category", ""),
                    )
                )
            ),
            "quality_tags": item.get("quality_tags", []),
        }
        if (
            source_row["quality_route"] == "pass"
            and item_id in model_conflict_ids
        ):
            candidate.update(
                {
                    "quality_route": "model_category_review",
                    "recommended_action": (
                        "反馈模型与当前类别不一致，建议人工看图确认"
                    ),
                }
            )
        candidates_by_id[item_id] = candidate

    for item in manifest:
        item_id = item["item_id"]
        needs_taxonomy_review = (
            item.get("taxonomy_review_status") == "pending"
        )
        direct_request = item_id == requested_item_id
        if not needs_taxonomy_review and not direct_request:
            continue
        ocr_row = ocr_by_id.get(item_id, {})
        synthetic = {
            "item_id": item_id,
            "filename": item.get("source_file_name", ""),
            "category": normalize_category(
                str(item.get("category", "general_text_document"))
            ),
            "suggested_category": "",
            "source_name": item.get("public_source_name")
            or item.get("source", "本地上传"),
            "has_text_expected": "",
            "text_box_count": ocr_row.get("text_box_count", 0),
            "character_count": ocr_row.get("character_count", 0),
            "mean_confidence": ocr_row.get("mean_confidence", 0.0),
            "sensitive_text_detected": "",
            "quality_route": (
                "direct_review"
                if direct_request and not needs_taxonomy_review
                else "taxonomy_review"
            ),
            "recommended_action": (
                "从检索结果直接发起，请确认主类别和质量标签"
                if direct_request and not needs_taxonomy_review
                else "旧类别同时混合内容与质量，请人工确认新的主类别"
            ),
            "quality_tags": item.get("quality_tags", []),
        }
        candidates_by_id.setdefault(item_id, synthetic)

    for item_id, proposal in assistant_proposals.items():
        if proposal.get("status", "pending") != "pending":
            continue
        item = manifest_by_id.get(item_id)
        if item is None:
            continue
        ocr_row = ocr_by_id.get(item_id, {})
        candidate = candidates_by_id.setdefault(
            item_id,
            {
                "item_id": item_id,
                "filename": item.get("source_file_name", ""),
                "category": normalize_category(
                    str(
                        item.get(
                            "category",
                            "general_text_document",
                        )
                    )
                ),
                "suggested_category": "",
                "source_name": (
                    item.get("public_source_name")
                    or item.get("source", "本地上传")
                ),
                "has_text_expected": "",
                "text_box_count": ocr_row.get("text_box_count", 0),
                "character_count": ocr_row.get("character_count", 0),
                "mean_confidence": ocr_row.get(
                    "mean_confidence", 0.0
                ),
                "sensitive_text_detected": "",
                "quality_route": "assistant_proposal_review",
                "recommended_action": (
                    "AI已逐页看图提出低权重类别建议，"
                    "请人工确认或修改后再进入正式训练。"
                ),
                "quality_tags": item.get("quality_tags", []),
            },
        )
        candidate["quality_route"] = "assistant_proposal_review"
        candidate["recommended_action"] = (
            "AI已逐页看图提出低权重类别建议，"
            "请人工确认或修改后再进入正式训练。"
        )
        candidate["assistant_proposal"] = proposal
    candidates = list(candidates_by_id.values())
    for candidate in candidates:
        assignment = review_split_by_id.get(
            candidate["item_id"], {}
        )
        candidate["review_split"] = assignment.get("split", "train")
        candidate["review_sequence"] = assignment.get(
            "review_sequence", ""
        )
    reviewed_ids = set(reviews)

    def run_feedback_training() -> None:
        completed = subprocess.run(
            [
                sys.executable,
                str(
                    PROJECT_ROOT
                    / "scripts/train_feedback_category_model.py"
                ),
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
            check=False,
        )
        if completed.returncode != 0:
            st.error("反馈模型训练失败。")
            st.code((completed.stderr or completed.stdout)[-1800:])
        else:
            st.success(
                "训练完成：仅训练集参与拟合，验证集和测试集保持隔离。"
            )
            st.rerun()

    if feedback_report:
        feedback_metrics = feedback_report.get(
            "human_feedback_metrics", {}
        )
        weak_metrics = feedback_report.get("weak_label_metrics", {})
        with st.expander("反馈学习状态", expanded=True):
            feedback_columns = st.columns(4)
            feedback_columns[0].metric(
                "有效人工反馈",
                feedback_metrics.get("reviewed_count", 0),
            )
            feedback_columns[1].metric(
                "弱标签 Macro-F1",
                f"{float(weak_metrics.get('macro_f1') or 0.0):.3f}",
            )
            feedback_columns[2].metric(
                "复核样本OOF一致率",
                f"{float(feedback_metrics.get('out_of_fold_model_accuracy') or 0.0):.1%}",
            )
            feedback_columns[3].metric(
                "模型分类冲突",
                len(model_conflict_ids),
            )
            st.caption(
                "模型融合Qwen3-VL图像向量、BGE-M3文本向量和OCR质量特征。"
                "Macro-F1主要基于来源弱标签，不等同于独立人工真值准确率。"
            )
            if feedback_policy.get("selected_on") == "validation":
                st.caption(
                    "验证集选择的保守门控：仅当模型置信度达到 "
                    f"{feedback_override_threshold:.0%} 时覆盖当前规则类别。"
                )
            retrain_clicked = st.button(
                "使用最新纠错重新学习",
                width="stretch",
            )
            if retrain_clicked:
                run_feedback_training()
    else:
        st.info(
            "尚未训练反馈分类器。先审核训练集；训练时验证集和测试集"
            "会被程序自动排除。"
        )
        if st.button(
            "训练首版反馈分类器",
            disabled=not CATEGORY_REVIEW_SPLITS_PATH.is_file(),
            width="stretch",
        ):
            run_feedback_training()

    columns = st.columns(4)
    columns[0].metric("资料库总页数", len(manifest))
    columns[1].metric("自动放行", len(manifest) - len(candidates))
    columns[2].metric("需要人工审核", len(candidates))
    columns[3].metric(
        "已人工裁决",
        sum(row["item_id"] in reviewed_ids for row in candidates),
    )
    st.caption(
        "自动规则只负责把风险页送入队列，不代替你的最终判断；"
        "每次人工裁决都会保存时间、决定和备注，形成可审计记录。"
    )
    if review_split_rows:
        split_progress = []
        for split in ("train", "validation", "test"):
            assigned_ids = {
                row["item_id"]
                for row in candidates
                if row["review_split"] == split
            }
            completed_count = len(assigned_ids & reviewed_ids)
            split_progress.append(
                f"{split_labels[split]} {completed_count}/"
                f"{len(assigned_ids)}"
            )
        st.caption(
            "固定实验划分：" + "　｜　".join(split_progress)
            + "。同源或近重复页面不会跨集合。"
        )
    else:
        st.warning(
            "尚未找到固定审核划分；请先运行 "
            "`python scripts/build_category_review_splits.py`。"
        )
    with st.expander("发布已审核决定到当前资料库"):
        st.write(
            "发布后，“重做OCR”和“隔离”页面会暂时退出检索，"
            "重分类会更新页面类别；发布前自动备份旧manifest，原图不会删除。"
        )
        publish_confirmed = st.checkbox(
            f"确认发布当前 {len(reviews)} 条人工决定",
            disabled=not reviews,
        )
        publish_clicked = st.button(
            "发布质量决定",
            disabled=not reviews or not publish_confirmed,
            width="stretch",
        )
        if publish_clicked:
            try:
                report = publish_quality_reviews(
                    library_dir / "manifest.jsonl",
                    reviews,
                )
            except (OSError, ValueError) as error:
                st.error(f"发布失败：{error}")
            else:
                st.session_state.pop("live_result", None)
                st.success(
                    f"已发布 {report['review_count']} 条决定；"
                    f"当前启用 {report['active_pages']} 页，"
                    f"暂停 {report['inactive_pages']} 页。旧manifest已备份。"
                )
                st.rerun()

    category_labels = dict(CATEGORY_LABELS)
    category_filter_labels = {
        "全部类别": "all",
        **{label: value for value, label in category_labels.items()},
    }
    category_filter_label = st.pills(
        "按图片类别筛选",
        options=list(category_filter_labels),
        default="全部类别",
    )
    selected_category = category_filter_labels[
        category_filter_label or "全部类别"
    ]

    split_filter_labels = {
        "训练集（先审核）": "train",
        "验证集": "validation",
        "测试集（最后审核）": "test",
        "全部集合": "all",
    }
    requested_split = review_split_by_id.get(
        requested_item_id, {}
    ).get("split", "train")
    default_split_label = next(
        (
            label
            for label, value in split_filter_labels.items()
            if value == requested_split
        ),
        "训练集（先审核）",
    )
    split_filter_label = st.selectbox(
        "按实验用途筛选",
        options=list(split_filter_labels),
        index=list(split_filter_labels).index(default_split_label),
        help=(
            "训练集用于学习；验证集用于调参；测试集只做最终一次评分。"
        ),
    )
    selected_split = split_filter_labels[split_filter_label]

    route_labels = {
        "privacy_review": "隐私复核",
        "category_review": "类别复核",
        "model_category_review": "模型分类冲突",
        "assistant_proposal_review": "AI辅助提议待确认",
        "taxonomy_review": "新类别待确认",
        "direct_review": "检索结果直接纠错",
        "ocr_low_confidence": "OCR低置信度",
        "category_text_leak": "无文字类出现文字",
        "ocr_retry": "建议重新OCR",
    }
    filter_labels = {"全部待审核": "all", **{
        label: value for value, label in route_labels.items()
    }}
    filter_label = st.selectbox(
        "按风险类型筛选",
        options=list(filter_labels),
    )
    selected_route = filter_labels[filter_label]
    filtered = [
        row
        for row in candidates
        if selected_route == "all"
        or row["quality_route"] == selected_route
    ]
    filtered = [
        row
        for row in filtered
        if selected_category == "all"
        or (
            reviews.get(row["item_id"], {}).get("revised_category")
            or row["category"]
        )
        == selected_category
    ]
    filtered = [
        row
        for row in filtered
        if selected_split == "all"
        or row["review_split"] == selected_split
    ]
    if not filtered:
        st.success("这个类别和风险条件下没有待审核图片。")
        return
    pending_indices = [
        index
        for index, row in enumerate(filtered)
        if row["item_id"] not in reviewed_ids
    ]
    requested_indices = [
        index
        for index, candidate in enumerate(filtered)
        if candidate["item_id"] == requested_item_id
    ]
    default_index = (
        requested_indices[0]
        if requested_indices
        else pending_indices[0]
        if pending_indices
        else 0
    )
    selected_index = st.selectbox(
        "选择图片",
        options=list(range(len(filtered))),
        index=default_index,
        format_func=lambda index: (
            f"{index + 1:03d} · "
            f"{split_labels.get(filtered[index]['review_split'], '训练集')}"
            f" · "
            f"{route_labels.get(filtered[index]['quality_route'], filtered[index]['quality_route'])}"
            f" · {'已裁决' if filtered[index]['item_id'] in reviewed_ids else '待审核'}"
            f" · {category_labels.get(reviews.get(filtered[index]['item_id'], {}).get('revised_category') or filtered[index]['category'], filtered[index]['category'])}"
            f" · {filtered[index]['item_id'][-6:]}"
        ),
    )
    row = filtered[selected_index]
    item = manifest_by_id.get(row["item_id"])
    if item is None:
        st.error(f"资料库中找不到页面：{row['item_id']}")
        return

    effective_category = normalize_category(
        reviews.get(row["item_id"], {}).get("revised_category")
        or str(row["category"])
    )
    left, right = st.columns([3, 2])
    with left:
        image_path = PROJECT_ROOT / item["source_path"]
        st.image(
            str(image_path),
            caption=(
                f"{category_labels.get(effective_category, effective_category)}"
                f" · {row['item_id']}"
            ),
            width="stretch",
        )
    with right:
        current_split = row.get("review_split", "train")
        st.markdown(
            f"**实验用途：** "
            f"{split_labels.get(current_split, current_split)}"
        )
        st.markdown(
            f"**自动路由：** "
            f"{route_labels.get(row['quality_route'], row['quality_route'])}"
        )
        st.write(f"**建议动作：** {row['recommended_action']}")
        st.write(
            f"**当前类别：** "
            f"{category_labels.get(effective_category, effective_category)}"
        )
        current_item_quality_tags = parse_quality_tags(
            item.get("quality_tags")
        )
        st.write(
            "**当前质量：** "
            + (
                "、".join(
                    QUALITY_LABELS[tag]
                    for tag in current_item_quality_tags
                )
                if current_item_quality_tags
                else "尚未人工标记"
            )
        )
        raw_model_feedback = feedback_by_id.get(row["item_id"])
        model_feedback = (
            raw_model_feedback
            if current_split == "train"
            else None
        )
        if model_feedback:
            predicted_category = normalize_category(
                model_feedback.get("predicted_category", "")
            )
            st.write(
                f"**反馈模型建议：** "
                f"{category_labels.get(predicted_category, predicted_category)}"
                f"（置信度 {float(model_feedback.get('confidence', 0.0)):.1%}）"
            )
        elif raw_model_feedback and current_split != "train":
            st.caption(
                "该页属于留出集，已隐藏模型建议，避免影响人工真值。"
            )
        assistant_proposal = row.get("assistant_proposal")
        if assistant_proposal:
            proposed_category = normalize_category(
                assistant_proposal.get("proposed_category", "")
            )
            st.info(
                "AI辅助提议（尚未成为人工真值）\n\n"
                f"- 建议类别："
                f"{category_labels.get(proposed_category, proposed_category)}\n"
                f"- 提议轮次：第 "
                f"{assistant_proposal.get('proposal_round', '')} 轮\n"
                f"- 看图依据："
                f"{assistant_proposal.get('assistant_notes', '')}"
            )
        st.write(
            f"**OCR：** {row['text_box_count']} 个文本框，"
            f"{row['character_count']} 字符，"
            f"平均置信度 {float(row['mean_confidence']):.3f}"
        )
        st.write(f"**公开来源：** {row['source_name']}")
        if item.get("public_source_url"):
            st.markdown(f"[打开原始公开来源]({item['public_source_url']})")
        with st.expander("查看OCR文本"):
            st.write(
                ocr_preview(row["item_id"], library_dir=library_dir)
                or "没有识别到文字"
            )

    decision_labels = {
        "保存类别与质量（可用）": "save",
        "重新运行OCR": "ocr_retry",
        "隔离，不进入正式评测": "quarantined",
    }
    old_review = reviews.get(row["item_id"], {})
    old_decision = old_review.get("decision", "accepted")
    old_decision_label = (
        "重新运行OCR"
        if old_decision == "ocr_retry"
        else "隔离，不进入正式评测"
        if old_decision == "quarantined"
        else "保存类别与质量（可用）"
    )
    current_category = normalize_category(
        old_review.get("revised_category") or str(row["category"])
    )
    assistant_proposal = row.get("assistant_proposal")
    assistant_category = (
        normalize_category(
            assistant_proposal.get("proposed_category", "")
        )
        if assistant_proposal
        else ""
    )
    suggested_category = normalize_category(
        row.get("suggested_category", "")
    ) if row.get("suggested_category") else ""
    if (
        not old_review
        and assistant_category in category_labels
    ):
        current_category = assistant_category
        st.info(
            "已预选AI看图建议；它目前只是低权重提议，"
            "必须由你保存后才会成为人工真值。"
        )
    elif (
        not old_review
        and suggested_category in category_labels
        and suggested_category != row["category"]
    ):
        current_category = suggested_category
        st.info(
            f"内容规则建议改为“{category_labels[suggested_category]}”，"
            "请结合图片确认。"
        )
    elif (
        not old_review
        and model_feedback
        and float(model_feedback.get("confidence", 0.0))
        >= feedback_override_threshold
        and normalize_category(
            model_feedback.get("predicted_category", "")
        )
        in category_labels
        and normalize_category(
            model_feedback.get("predicted_category", "")
        )
        != row["category"]
    ):
        current_category = normalize_category(
            model_feedback["predicted_category"]
        )
        st.info(
            f"反馈模型高置信建议改为"
            f"“{category_labels[current_category]}”，请结合图片确认。"
        )
    with st.form(f"quality_review_{row['item_id']}"):
        decision_label = st.radio(
            "你的最终裁决",
            options=list(decision_labels),
            index=list(decision_labels).index(old_decision_label),
        )
        revised_category = st.selectbox(
            "主类别",
            options=list(category_labels),
            index=list(category_labels).index(current_category),
            format_func=category_labels.get,
        )
        old_quality_tags = parse_quality_tags(
            old_review.get("quality_tags")
            or item.get("quality_tags")
        )
        revised_quality_tags = st.multiselect(
            "质量标签（可以多选）",
            options=list(QUALITY_LABELS),
            default=old_quality_tags,
            format_func=QUALITY_LABELS.get,
        )
        st.caption(
            "“无明显质量问题”不能与模糊、反光等问题同时选择。"
        )
        notes = st.text_input(
            "判断依据（建议写清看到了什么）",
            value=old_review.get("human_notes", ""),
        )
        submitted = st.form_submit_button(
            "保存人工质量裁决",
            type="primary",
            width="stretch",
        )
    if submitted:
        selected_action = decision_labels[decision_label]
        decision = (
            "reclassified"
            if selected_action == "save"
            and revised_category != normalize_category(
                str(item.get("category", ""))
            )
            else "accepted"
            if selected_action == "save"
            else selected_action
        )
        category_value = (
            revised_category if decision == "reclassified" else ""
        )
        try:
            category_value, normalized_quality_tags = (
                validate_quality_review(
                item_id=row["item_id"],
                decision=decision,
                revised_category=category_value,
                quality_tags=revised_quality_tags,
                known_item_ids=set(manifest_by_id),
            )
            )
            save_quality_review(
                QUALITY_HUMAN_REVIEWS_PATH,
                {
                    "item_id": row["item_id"],
                    "decision": decision,
                    "revised_category": category_value,
                    "quality_tags": normalized_quality_tags,
                    "human_notes": notes.strip(),
                },
            )
            resolve_assistant_proposal(
                ASSISTANT_CATEGORY_PROPOSALS_PATH,
                item_id=row["item_id"],
                human_category=(
                    revised_category
                    if decision in {"accepted", "reclassified"}
                    else None
                ),
            )
        except ValueError as error:
            st.error(str(error))
        else:
            st.success("已保存。该决定已进入人工审核记录，不会静默删除原始证据。")
            st.rerun()


def execute_next_blind_candidate(
    study_dir: Path,
    library_dir: Path,
) -> dict[str, Any]:
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/build_blind_study_candidates.py"),
            "--study-dir",
            str(study_dir),
            "--library-dir",
            str(library_dir),
            "--limit",
            "1",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=900,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout)[-2200:]
        raise RuntimeError(detail)
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("候选生成脚本没有返回进度。")
    return json.loads(lines[-1])


def render_blind_query_collection(
    selected_library: dict[str, Any],
    library_dir: Path,
) -> None:
    """Collect natural queries without executing or displaying retrieval."""
    st.subheader("独立盲测查询采集")
    st.info(
        "请只输入你现实中会搜索的一句话。采集阶段不会运行检索、"
        "不会展示候选图片，也不会告诉你系统能否回答。"
    )
    revision = library_revision(library_dir)
    try:
        protocol = initialize_blind_study(
            BLIND_STUDY_DIR,
            study_id="blind_v1",
            library_id=selected_library["id"],
            library_revision=revision,
        )
        assert_blind_library_unchanged(protocol, revision)
    except (ValueError, OSError) as error:
        st.error(str(error))
        return
    summary = blind_progress_summary(BLIND_STUDY_DIR)
    if protocol.get("query_origin") == "assistant_generated":
        st.warning(
            "本轮查询由助手生成，只能作为诊断集；它不会被表述为"
            "独立用户盲测成绩。"
        )
    left, middle, right = st.columns(3)
    left.metric("已提交", summary["submitted"])
    middle.metric("目标", summary["target"])
    right.metric("当前阶段", "盲采集中" if summary["status"] == "collecting" else "已封存")
    st.progress(min(summary["submitted"] / summary["target"], 1.0))
    st.caption(
        f"目标 {summary['target']} 条，上限 {summary['maximum']} 条。"
        "题目封存前，系统结果始终不可见。"
    )
    minimum_no_answer = int(protocol.get("minimum_no_answer_probes", 0))
    current_no_answer = summary["expectations"].get("no_answer_probe", 0)
    st.caption(
        f"其中至少保留 {minimum_no_answer} 条专测无答案查询；"
        f"当前 {current_no_answer} 条。"
    )

    if summary["status"] != "collecting":
        st.success(
            "本轮查询已经封存，采集入口已锁定。后续候选生成和相关性"
            "审核只在内部纠错站进行。"
        )
        return

    with st.form("blind_query_submission", clear_on_submit=True):
        query = st.text_area(
            "你现在想从这个资料库里找什么？",
            height=100,
            max_chars=200,
            placeholder="直接输入真实需求，不需要使用固定格式",
        )
        expectation_label = st.selectbox(
            "你预期资料库里有答案吗？（不确定就保持默认）",
            options=list(BLIND_EXPECTATION_LABELS),
        )
        submitted = st.form_submit_button(
            "提交这条盲测查询",
            type="primary",
            width="stretch",
        )
    if submitted:
        try:
            submit_blind_query(
                BLIND_STUDY_DIR,
                query=query,
                expected_answerability=BLIND_EXPECTATION_LABELS[
                    expectation_label
                ],
                current_library_revision=revision,
            )
        except ValueError as error:
            st.error(str(error))
        else:
            st.success("已密封保存；本次没有运行检索。")
            st.rerun()

    submissions = read_blind_submissions(BLIND_STUDY_DIR)
    if submissions:
        latest = submissions[-1]
        with st.expander("刚才输错了？只可撤回最后一条"):
            st.write(latest["query"])
            if st.button("撤回最后一条", width="stretch"):
                try:
                    delete_blind_submission(
                        BLIND_STUDY_DIR,
                        latest["query_id"],
                        current_library_revision=revision,
                    )
                except ValueError as error:
                    st.error(str(error))
                else:
                    st.rerun()
    if summary["submitted"] >= summary["target"]:
        st.success(
            "采集目标已达到。请到内部纠错站确认封存；封存后才能开始"
            "生成多路候选和人工相关性审核。"
        )


def render_blind_study_review(
    selected_library: dict[str, Any],
    library_dir: Path,
) -> None:
    st.subheader("独立盲测管理与相关性审核")
    if not (BLIND_STUDY_DIR / "protocol.json").is_file():
        st.info("本轮尚未开始采集。请先在主站的“独立盲测”中提交查询。")
        return
    try:
        protocol = load_blind_protocol(BLIND_STUDY_DIR)
        revision = library_revision(library_dir)
        if protocol["library_id"] != selected_library["id"]:
            raise ValueError("请切换到本轮盲测绑定的资料库。")
        assert_blind_library_unchanged(protocol, revision)
        summary = blind_progress_summary(BLIND_STUDY_DIR)
    except (ValueError, OSError) as error:
        st.error(str(error))
        return

    if protocol.get("query_origin") == "assistant_generated":
        st.warning(
            "本轮是助手生成诊断集，不是独立用户查询集；正式报告必须"
            "保留这一来源说明。"
        )

    metrics = st.columns(4)
    metrics[0].metric("冻结查询", summary["submitted"])
    metrics[1].metric("候选已生成", summary["candidates"])
    metrics[2].metric("已审核", summary["reviews"])
    metrics[3].metric(
        "阶段",
        {
            "collecting": "采集中",
            "frozen": "已冻结",
            "reviewing": "审核中",
            "completed": "已完成",
        }[summary["status"]],
    )

    if protocol["status"] == "collecting":
        st.warning(
            "采集尚未结束。为避免看见系统输出后改题，这里只显示数量，"
            "不展示题目或候选。"
        )
        expectations = summary["expectations"]
        st.caption(
            "作者预期分布（不是相关性真值）："
            f"有答案 {expectations.get('answerable', 0)} · "
            f"不确定 {expectations.get('unsure', 0)} · "
            f"专测无答案 {expectations.get('no_answer_probe', 0)}"
        )
        enough_no_answer = (
            expectations.get("no_answer_probe", 0)
            >= int(protocol.get("minimum_no_answer_probes", 0))
        )
        ready = summary["submitted"] >= summary["target"] and enough_no_answer
        confirmed = st.checkbox(
            "我确认查询作者已停止出题，冻结后不再添加或修改",
            disabled=not ready,
        )
        if st.button(
            f"冻结本轮 {summary['target']} 条查询",
            type="primary",
            width="stretch",
            disabled=not (ready and confirmed),
        ):
            try:
                freeze_blind_study(
                    BLIND_STUDY_DIR,
                    current_library_revision=revision,
                )
            except ValueError as error:
                st.error(str(error))
            else:
                st.success("冻结完成，查询指纹已写入协议。")
                st.rerun()
        if not ready:
            remaining = max(0, summary["target"] - summary["submitted"])
            no_answer_remaining = max(
                0,
                int(protocol.get("minimum_no_answer_probes", 0))
                - expectations.get("no_answer_probe", 0),
            )
            st.caption(
                f"距离冻结要求：总查询还差 {remaining} 条；"
                f"专测无答案还差 {no_answer_remaining} 条。"
            )
        return

    try:
        frozen = verify_blind_snapshot(BLIND_STUDY_DIR)
    except ValueError as error:
        st.error(str(error))
        return
    candidates = read_blind_candidates(BLIND_STUDY_DIR)
    reviews = read_blind_reviews(BLIND_STUDY_DIR)
    if protocol["status"] == "completed":
        st.success(
            f"本轮盲测真值已完成并导出到 {BLIND_FORMAL_QUERY_PATH.name}。"
        )
        return

    if len(candidates) < len(frozen):
        st.info(
            "候选池由智能混合、OCR 文字和视觉语义三路 Top-12 合并，"
            "每次只生成一题并立即落盘，可随时中断后继续。"
        )
        if st.button(
            "生成下一条盲审候选",
            type="primary",
            width="stretch",
        ):
            try:
                with st.spinner("正在运行三路检索并密封候选池……"):
                    result = execute_next_blind_candidate(
                        BLIND_STUDY_DIR, library_dir
                    )
            except (RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
                st.error(f"候选生成失败：{error}")
            else:
                st.success(
                    f"已生成 {result['processed_count']} 条；"
                    f"剩余 {result['remaining_count']} 条。"
                )
                st.rerun()

    reviewable = [row for row in frozen if row["query_id"] in candidates]
    if not reviewable:
        st.caption("先生成第一条候选，之后才会出现人工审核卡。")
        return
    pending = [row for row in reviewable if row["query_id"] not in reviews]
    default = reviewable.index(pending[0]) if pending else 0
    selected_index = st.selectbox(
        "选择盲审任务",
        options=list(range(len(reviewable))),
        index=default,
        format_func=lambda index: (
            f"{reviewable[index]['query_id']} · "
            f"{'已审核' if reviewable[index]['query_id'] in reviews else '待审核'}"
        ),
    )
    row = reviewable[selected_index]
    candidate = candidates[row["query_id"]]
    st.markdown(f"### 查询：{row['query']}")
    st.caption("候选顺序已经盲化；页面不显示系统排名、分数或作者预期。")
    if BLIND_MODEL_HARD_QUEUE_PATH.is_file():
        hard_payload = json.loads(
            BLIND_MODEL_HARD_QUEUE_PATH.read_text(encoding="utf-8")
        )
        hard_by_id = {
            item["query_id"]: item
            for item in hard_payload.get("unresolved", [])
        }
        hard_item = hard_by_id.get(row["query_id"])
        if hard_item is not None:
            st.warning(
                "这题经过三位无记忆AI复核仍未形成精确一致，"
                "需要你做最终判断。"
            )
            decision_display = {
                "answerable": "有答案",
                "no_answer": "无答案",
                "excluded": "排除",
                "ambiguous": "仍然含糊",
            }
            with st.expander("查看AI给出的中文判读说明"):
                for vote in hard_item.get("votes", []):
                    st.markdown(
                        f"**{vote['reviewer']} · "
                        f"{decision_display.get(vote['decision'], vote['decision'])} · "
                        f"置信度 {float(vote['confidence']):.0%}**"
                    )
                    st.write(vote.get("notes_zh", "未提供说明"))
                    if vote.get("uncertain_item_ids"):
                        st.caption(
                            "仍有疑问的图片ID："
                            + "、".join(vote["uncertain_item_ids"])
                        )
    manifest = read_jsonl(library_dir / "manifest.jsonl")
    manifest_by_id = {item["item_id"]: item for item in manifest}
    candidate_ids = list(candidate["candidate_item_ids"])
    columns = st.columns(3)
    for index, item_id in enumerate(candidate_ids):
        item = manifest_by_id.get(item_id)
        with columns[index % 3]:
            if item is None:
                st.error(f"找不到候选 {item_id}")
                continue
            image_path = PROJECT_ROOT / item["source_path"]
            st.image(str(image_path), width="stretch")
            st.caption(f"{item.get('source_file_name', item_id)} · {item_id}")

    old = reviews.get(row["query_id"], {})
    old_relevant = [
        item_id
        for item_id in old.get("relevant_item_ids", "").split(";")
        if item_id in candidate_ids
    ]
    old_additional = [
        item_id
        for item_id in old.get("relevant_item_ids", "").split(";")
        if item_id and item_id not in candidate_ids
    ]
    decision_labels = {
        "有答案，并标出所有相关图片": "answerable",
        "确认资料库中无答案": "no_answer",
        "排除：问题含糊、隐私或无法公平判断": "excluded",
    }
    decision_values = list(decision_labels.values())
    old_decision = old.get("decision", "answerable")
    default_decision_index = (
        decision_values.index(old_decision)
        if old_decision in decision_values
        else 0
    )
    evidence_labels = {
        "只核对了多路候选池": "candidate_pool",
        "还浏览了完整资料库": "library_browse",
        "查询作者明确知道目标资料": "author_known",
    }
    old_scope = old.get("evidence_scope", "candidate_pool")
    scope_values = list(evidence_labels.values())
    scope_index = scope_values.index(old_scope) if old_scope in scope_values else 0
    with st.form(f"blind_review_{row['query_id']}"):
        selected_ids = st.multiselect(
            "哪些候选都算正确答案？",
            options=candidate_ids,
            default=old_relevant,
            format_func=lambda item_id: (
                f"{manifest_by_id.get(item_id, {}).get('source_file_name', item_id)}"
                f" · {item_id}"
            ),
        )
        with st.expander("候选池漏掉了正确图片？补充图片 ID"):
            additional_ids = st.text_input(
                "额外相关图片ID（多张用分号分隔）",
                value=";".join(old_additional),
            )
        decision_label = st.radio(
            "最终裁决",
            options=list(decision_labels),
            index=default_decision_index,
        )
        scope_label = st.radio(
            "你核对到了什么范围？",
            options=list(evidence_labels),
            index=scope_index,
        )
        notes = st.text_input("判断依据或备注（可选）", value=old.get("human_notes", ""))
        save_clicked = st.form_submit_button(
            "保存这条盲审真值",
            type="primary",
            width="stretch",
        )
    if save_clicked:
        combined_ids = ";".join([*selected_ids, additional_ids])
        try:
            save_blind_relevance_review(
                BLIND_STUDY_DIR,
                {
                    "query_id": row["query_id"],
                    "decision": decision_labels[decision_label],
                    "relevant_item_ids": combined_ids,
                    "evidence_scope": evidence_labels[scope_label],
                    "reviewer_type": "human",
                    "reviewer_id": "local_user",
                    "human_notes": notes,
                },
                known_item_ids=set(manifest_by_id),
            )
        except ValueError as error:
            st.error(str(error))
        else:
            st.success("人工相关性真值已保存。")
            st.rerun()

    candidates_complete = len(candidates) == len(frozen)
    reviews_complete = len(reviews) == len(frozen)
    if candidates_complete and reviews_complete:
        st.success("候选和人工审核均已完成，可以生成正式盲测文件。")
        if st.button("生成正式盲测集", type="primary", width="stretch"):
            try:
                formal_rows = materialize_blind_formal_rows(
                    BLIND_STUDY_DIR,
                    minimum_eligible=int(protocol["minimum_count"]),
                )
                write_blind_formal_query_set(
                    BLIND_FORMAL_QUERY_PATH, formal_rows
                )
                write_blind_evaluation_protocol(
                    BLIND_EVALUATION_PROTOCOL_PATH,
                    study_dir=BLIND_STUDY_DIR,
                    formal_rows=formal_rows,
                    query_file=BLIND_FORMAL_QUERY_PATH.relative_to(
                        PROJECT_ROOT
                    ).as_posix(),
                    library_dir=library_dir.relative_to(PROJECT_ROOT).as_posix(),
                    output=BLIND_EVALUATION_OUTPUT_PATH.relative_to(
                        PROJECT_ROOT
                    ).as_posix(),
                )
            except ValueError as error:
                st.error(str(error))
            else:
                st.success(f"已生成 {len(formal_rows)} 条正式盲测查询。")
                st.rerun()


def render_query_review() -> None:
    st.subheader("检索答案纠错")
    st.info(
        "你只做一件事：看图后判断“这道搜索题是否合理”。"
        "合理就写一句普通人会搜索的话；涉及隐私、说不清或一题多解就排除。"
    )
    queue = read_csv_rows(QUERY_REVIEW_QUEUE_PATH)
    reviews = read_reviews(QUERY_HUMAN_REVIEWS_PATH)
    reviewed_count = sum(
        query["query_id"] in reviews for query in queue
    )
    st.progress(reviewed_count / len(queue) if queue else 0.0)
    st.caption(f"已完成 {reviewed_count} / {len(queue)} 条检索真值")
    if not USER_MANIFEST_PATH.is_file():
        st.info(
            "公开发行版不包含本地资料清单。请先创建或导入资料库，"
            "再进行检索真值审核。"
        )
        return
    manifest = read_jsonl(USER_MANIFEST_PATH)
    manifest_by_id = {row["item_id"]: row for row in manifest}
    if not queue:
        st.info("当前没有待审核的检索真值任务。")
        return

    pending_indices = [
        index
        for index, query in enumerate(queue)
        if query["query_id"] not in reviews
    ]
    default_index = pending_indices[0] if pending_indices else 0
    selected_index = st.selectbox(
        "选择复核任务",
        options=list(range(len(queue))),
        index=default_index,
        format_func=lambda index: (
            f"{index + 1:02d} · {queue[index]['query_id']} · "
            f"{'已完成' if queue[index]['query_id'] in reviews else '待复核'}"
        ),
    )
    row = queue[selected_index]
    is_no_answer_task = row.get("query_type") == "no_answer"
    relevant_ids = [
        item_id
        for item_id in row["relevant_item_ids"].split(";")
        if item_id
    ]
    display_ids = (
        [
            item_id
            for item_id in row.get("candidate_item_ids", "").split(";")
            if item_id
        ]
        if is_no_answer_task
        else relevant_ids
    )
    category_labels = dict(CATEGORY_LABELS)
    category_labels["open_set_no_answer"] = "库中无答案"
    split_labels = {"validation": "验证集", "test": "测试集"}
    risk_labels = {
        "blank_query": "系统没有生成安全查询",
        "privacy": "可能涉及隐私",
        "category": "类别需要确认",
        "text_failure": "文字检索失败",
        "visual_failure": "视觉检索失败",
        "adaptive_failure": "融合检索失败",
        "ambiguous": "可能一题多解",
        "no_answer": "需要确认库中确实没有答案",
    }
    risk_codes = [
        value
        for value in row["diagnostic_review_reasons"].split(",")
        if value
    ]
    risks = "、".join(
        risk_labels.get(value, value) for value in risk_codes
    )
    st.markdown(
        f"**图片类型：** {category_labels.get(row['category'], row['category'])}"
        f"　　**用途：** {split_labels.get(row['split'], row['split'])}"
        f"　　**为什么要你看：** {risks}"
    )
    privacy_risk = "privacy" in risk_codes
    if privacy_risk:
        st.error(
            "系统建议：排除这题。图片含个人身份信息，不应为了增加题目数量"
            "而设计可检索隐私的查询。"
        )
    elif is_no_answer_task:
        st.warning(
            "这是拒答测试：确认资料库中没有符合问题的图片或文档。"
            "下面若显示候选，只是系统目前最接近的低置信结果，不代表正确答案。"
        )
        if row.get("system_acceptance_reason"):
            st.caption(
                "当前拒答判断："
                + row["system_acceptance_reason"]
            )
    elif "blank_query" in risk_codes:
        st.warning(
            "系统建议：如果能用不含姓名、号码、地址的一句话描述图片，就"
            "“改写后通过”；否则排除。"
        )
    else:
        st.write(
            "系统建议：检查候选查询是否自然、是否只对应下面这组图；"
            "不自然就改一句。"
        )
    columns = st.columns(min(3, max(1, len(display_ids))))
    for index, item_id in enumerate(display_ids):
        item = manifest_by_id.get(item_id)
        with columns[index % len(columns)]:
            if item is None:
                st.error(f"找不到 {item_id}")
                continue
            image_path = PROJECT_ROOT / item["source_path"]
            st.image(
                str(image_path),
                caption=(
                    ("系统低置信候选 · " if is_no_answer_task else "")
                    + f"{item['source_file_name']} · {item_id}"
                ),
                width="stretch",
            )
    if is_no_answer_task and not display_ids:
        st.caption("无答案任务尚未附带候选图；请先按问题语义判断。")

    old_review = reviews.get(row["query_id"], {})
    proposed_query = (
        old_review.get("reviewed_query")
        or row.get("human_query_revision")
        or row["query"]
    )
    proposed_ids = (
        old_review.get("relevant_item_ids") or row["relevant_item_ids"]
    )
    label_to_value = {
        "接受当前查询": "accepted",
        "修改后接受": "revised",
        "确认库中无答案": "no_answer",
        "排除（不公平/隐私/无法描述）": "excluded",
    }
    value_to_label = {value: label for label, value in label_to_value.items()}
    default_decision = (
        "excluded"
        if privacy_risk
        else "no_answer"
        if is_no_answer_task
        else "accepted"
    )
    old_decision = old_review.get("decision", default_decision)
    with st.form(f"query_review_{row['query_id']}"):
        reviewed_query = st.text_area(
            "如果保留这题：写一句普通人会输入的搜索话",
            value=proposed_query,
            height=100,
            placeholder=(
                "例如：查找蓝天下的校园建筑。"
                "若选择排除，这里可以留空。"
            ),
        )
        with st.expander("高级设置：哪些图片都算正确答案（通常不用改）"):
            relevant_item_ids = st.text_input(
                "正确图片ID（多张用英文分号分隔）",
                value=proposed_ids,
            )
        decision_label = st.radio(
            "最后选一个",
            options=[
                "接受当前查询",
                "修改后接受",
                "确认库中无答案",
                "排除（不公平/隐私/无法描述）",
            ],
            index=list(label_to_value).index(
                value_to_label.get(old_decision, "接受当前查询")
            ),
            horizontal=True,
        )
        notes = st.text_input(
            "判断依据或备注（可选）",
            value=old_review.get("human_notes", ""),
        )
        submitted = st.form_submit_button(
            "保存这条人工真值",
            type="primary",
            width="stretch",
        )
    if submitted:
        decision = label_to_value[decision_label]
        try:
            normalized_query, normalized_ids = validate_review(
                decision,
                reviewed_query,
                relevant_item_ids,
                set(manifest_by_id),
            )
            save_review(
                QUERY_HUMAN_REVIEWS_PATH,
                {
                    "query_id": row["query_id"],
                    "decision": decision,
                    "reviewed_query": normalized_query,
                    "relevant_item_ids": normalized_ids,
                    "human_notes": notes.strip(),
                },
            )
        except ValueError as error:
            st.error(str(error))
        else:
            st.success("已保存。你的判断会作为正式评测集的人工真值。")
            st.rerun()

    if all(query["query_id"] in reviews for query in queue):
        try:
            formal_rows = build_formal_rows(queue, reviews)
        except ValueError as error:
            st.error(
                f"{len(queue)}条虽已操作完，但质量闸门未通过：{error}"
            )
        else:
            st.success(
                f"质量闸门通过：保留{len(formal_rows)}条正式查询，"
                "可以生成正式数据集并运行消融实验。"
            )


def render_product_header(library_name: str) -> None:
    safe_library_name = html.escape(library_name)
    st.markdown(
        f"""
        <section class="product-hero">
          <div class="hero-copy">
            <div class="hero-eyebrow">
              <span class="status-dot"></span>
              本地多模态资料工作台
            </div>
            <h1>一句话，找回所有资料</h1>
            <p>
              图片和文档统一入库，文字、版面与视觉内容协同检索。
            </p>
            <div class="hero-tags">
              <span>图片与文档</span>
              <span>自然语言检索</span>
              <span>本机运行</span>
              <span class="library-tag">当前 · {safe_library_name}</span>
            </div>
          </div>
          <div class="hero-visual" aria-hidden="true">
            <div class="visual-toolbar">
              <span class="visual-dot"></span>
              <span class="visual-dot"></span>
              <span class="visual-dot"></span>
              <b>语义检索</b>
            </div>
            <div class="visual-query">
              <span class="query-icon">⌕</span>
              找到包含目标内容的资料
            </div>
            <div class="visual-result">
              <div class="result-rank">01</div>
              <div class="result-thumb">DOC</div>
              <div class="result-lines">
                <i></i><i></i><i></i>
              </div>
              <div class="result-score">96%</div>
            </div>
            <div class="visual-result secondary">
              <div class="result-rank">02</div>
              <div class="result-thumb image">IMG</div>
              <div class="result-lines">
                <i></i><i></i><i></i>
              </div>
              <div class="result-score">89%</div>
            </div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="OCR-VLM 多模态检索",
        page_icon="🔎",
        layout="wide",
    )
    # Begin model initialization while the rest of the page is rendered.
    # Search remains safe if this fails because execute_live_search falls
    # back to the existing isolated CLI scorer.
    persistent_text_model_runtime()
    st.markdown(
        """
        <style>
        :root {
            --navy-950: #071A2B;
            --navy-800: #12304A;
            --teal-500: #18B6A4;
            --teal-300: #68DDCF;
            --blue-500: #3E78D6;
            --ink-900: #132238;
            --ink-600: #5D6B7D;
            --line: #DCE5EC;
            --surface: rgba(255, 255, 255, 0.92);
        }
        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at 78% 4%, rgba(24, 182, 164, 0.10), transparent 26rem),
                radial-gradient(circle at 7% 34%, rgba(62, 120, 214, 0.08), transparent 30rem),
                #F4F7FA;
        }
        [data-testid="stHeader"] {
            background: rgba(244, 247, 250, 0.82);
            backdrop-filter: blur(14px);
        }
        .block-container {
            max-width: 1480px;
            padding-top: 1.4rem;
            padding-bottom: 4rem;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #F9FBFC 0%, #EFF5F6 100%);
            border-right: 1px solid #DCE6EA;
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            color: var(--ink-600);
        }
        [data-testid="stSidebar"] [role="radiogroup"] label {
            border-radius: 11px;
            padding: 0.48rem 0.62rem;
            margin: 0.12rem 0;
            transition: background 150ms ease, transform 150ms ease;
        }
        [data-testid="stSidebar"] [role="radiogroup"] label:hover {
            background: rgba(24, 182, 164, 0.08);
            transform: translateX(2px);
        }
        [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
            background: rgba(24, 182, 164, 0.13);
            color: #087C71;
        }
        .sidebar-brand {
            display: flex;
            align-items: center;
            gap: 0.7rem;
            margin: 0.2rem 0 1.2rem;
            padding: 0.35rem 0.15rem;
            color: var(--navy-950);
            font-size: 1.02rem;
            font-weight: 760;
            letter-spacing: -0.01em;
        }
        .sidebar-brand-icon {
            display: grid;
            place-items: center;
            width: 2.15rem;
            height: 2.15rem;
            border-radius: 11px;
            color: white;
            background: linear-gradient(145deg, var(--teal-500), var(--blue-500));
            box-shadow: 0 8px 20px rgba(24, 182, 164, 0.24);
        }
        .product-hero {
            position: relative;
            display: grid;
            grid-template-columns: minmax(0, 1.25fr) minmax(300px, 0.75fr);
            gap: 2rem;
            min-height: 310px;
            margin: 0.35rem 0 2rem;
            padding: clamp(2rem, 4vw, 3.4rem);
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.16);
            border-radius: 28px;
            color: white;
            background:
                radial-gradient(circle at 88% 18%, rgba(104, 221, 207, 0.25), transparent 16rem),
                linear-gradient(125deg, #071A2B 0%, #103752 54%, #0D5B62 100%);
            box-shadow: 0 24px 60px rgba(7, 26, 43, 0.18);
        }
        .product-hero::after {
            content: "";
            position: absolute;
            right: -5rem;
            bottom: -8rem;
            width: 24rem;
            height: 24rem;
            border: 1px solid rgba(104, 221, 207, 0.17);
            border-radius: 50%;
            box-shadow:
                0 0 0 3rem rgba(104, 221, 207, 0.035),
                0 0 0 6rem rgba(104, 221, 207, 0.025);
        }
        .hero-copy {
            position: relative;
            z-index: 2;
            align-self: center;
        }
        .hero-eyebrow {
            display: inline-flex;
            align-items: center;
            gap: 0.55rem;
            margin-bottom: 1.2rem;
            color: #BDEFE9;
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.11em;
        }
        .status-dot {
            width: 0.48rem;
            height: 0.48rem;
            border-radius: 50%;
            background: #62E6C8;
            box-shadow: 0 0 0 5px rgba(98, 230, 200, 0.13);
        }
        .product-hero h1 {
            max-width: 760px;
            margin: 0;
            color: white;
            font-size: clamp(2.15rem, 3.7vw, 3.6rem);
            line-height: 1.08;
            letter-spacing: -0.045em;
        }
        .product-hero p {
            max-width: 680px;
            margin: 1.15rem 0 1.5rem;
            color: rgba(232, 245, 247, 0.78);
            font-size: 1rem;
            line-height: 1.75;
        }
        .hero-tags {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
        }
        .hero-tags span {
            padding: 0.44rem 0.7rem;
            border: 1px solid rgba(255, 255, 255, 0.14);
            border-radius: 999px;
            color: #DDEDF0;
            background: rgba(255, 255, 255, 0.07);
            font-size: 0.78rem;
        }
        .hero-tags .library-tag {
            color: #071A2B;
            border-color: transparent;
            background: #68DDCF;
            font-weight: 700;
        }
        .hero-visual {
            position: relative;
            z-index: 2;
            align-self: center;
            padding: 1rem;
            border: 1px solid rgba(255, 255, 255, 0.14);
            border-radius: 20px;
            background: rgba(5, 22, 34, 0.42);
            box-shadow: 0 18px 50px rgba(2, 13, 23, 0.28);
            backdrop-filter: blur(14px);
            transform: rotate(1.2deg);
        }
        .visual-toolbar {
            display: flex;
            align-items: center;
            gap: 0.34rem;
            margin-bottom: 0.85rem;
            color: #CFE4E8;
            font-size: 0.72rem;
        }
        .visual-toolbar b {
            margin-left: 0.35rem;
            font-weight: 650;
        }
        .visual-dot {
            width: 0.42rem;
            height: 0.42rem;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.38);
        }
        .visual-query {
            display: flex;
            align-items: center;
            gap: 0.55rem;
            padding: 0.74rem 0.8rem;
            border: 1px solid rgba(104, 221, 207, 0.22);
            border-radius: 11px;
            color: #E8F6F7;
            background: rgba(255, 255, 255, 0.08);
            font-size: 0.76rem;
        }
        .query-icon {
            color: #68DDCF;
            font-size: 1.2rem;
        }
        .visual-result {
            display: grid;
            grid-template-columns: auto auto 1fr auto;
            align-items: center;
            gap: 0.7rem;
            margin-top: 0.7rem;
            padding: 0.65rem;
            border-radius: 12px;
            background: rgba(255, 255, 255, 0.96);
            color: var(--ink-900);
        }
        .visual-result.secondary {
            opacity: 0.72;
            transform: scale(0.97);
        }
        .result-rank {
            color: #76909F;
            font-size: 0.68rem;
            font-weight: 750;
        }
        .result-thumb {
            display: grid;
            place-items: center;
            width: 2.35rem;
            height: 2.35rem;
            border-radius: 9px;
            color: #087C71;
            background: #D9F5F1;
            font-size: 0.58rem;
            font-weight: 800;
        }
        .result-thumb.image {
            color: #285FAF;
            background: #E3EDFC;
        }
        .result-lines {
            display: grid;
            gap: 0.3rem;
        }
        .result-lines i {
            display: block;
            height: 0.3rem;
            border-radius: 99px;
            background: #D9E2E8;
        }
        .result-lines i:nth-child(2) { width: 82%; }
        .result-lines i:nth-child(3) { width: 58%; }
        .result-score {
            color: #087C71;
            font-size: 0.7rem;
            font-weight: 800;
        }
        .search-summary {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin: -0.2rem 0 0.7rem;
        }
        .search-summary span {
            padding: 0.36rem 0.62rem;
            border: 1px solid var(--line);
            border-radius: 999px;
            color: var(--ink-600);
            background: rgba(255, 255, 255, 0.72);
            font-size: 0.76rem;
        }
        .search-summary .search-count {
            color: #087C71;
            border-color: rgba(24, 182, 164, 0.22);
            background: rgba(24, 182, 164, 0.09);
            font-weight: 750;
        }
        .evidence-summary {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            margin: 0.65rem 0 0.15rem;
            padding: 0.38rem 0.58rem;
            border: 1px solid rgba(24, 182, 164, 0.20);
            border-radius: 9px;
            color: #087C71;
            background: rgba(24, 182, 164, 0.08);
            font-size: 0.76rem;
            font-weight: 720;
        }
        .evidence-icon {
            display: grid;
            place-items: center;
            width: 1rem;
            height: 1rem;
            border-radius: 50%;
            color: white;
            background: var(--teal-500);
            font-size: 0.62rem;
        }
        .evidence-quote {
            margin-top: 0.65rem;
            padding: 0.78rem 0.85rem;
            border-left: 3px solid var(--teal-500);
            border-radius: 0 10px 10px 0;
            color: #435266;
            background: #F6FAFA;
            font-size: 0.86rem;
            line-height: 1.7;
        }
        .evidence-quote mark {
            padding: 0.05rem 0.18rem;
            border-radius: 4px;
            color: #075E56;
            background: #CFF5EC;
        }
        h1, h2, h3, h4 {
            color: var(--ink-900);
            letter-spacing: -0.025em;
        }
        [data-testid="stMetric"] {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 0.95rem 1.05rem;
            box-shadow: 0 9px 25px rgba(18, 48, 74, 0.05);
        }
        [data-testid="stMetricLabel"] {
            color: var(--ink-600);
        }
        div.stButton > button {
            min-height: 2.75rem;
            border-radius: 11px;
            border-color: #C9D7DF;
            font-weight: 680;
            transition: transform 140ms ease, box-shadow 140ms ease;
        }
        div.stButton > button:hover {
            transform: translateY(-1px);
            border-color: var(--teal-500);
            color: #087C71;
        }
        div.stButton > button[kind="primary"] {
            border: 0;
            color: white;
            background: linear-gradient(110deg, #159B91, #3474CC);
            box-shadow: 0 10px 22px rgba(31, 126, 158, 0.20);
        }
        div[data-baseweb="input"] > div,
        div[data-baseweb="select"] > div,
        [data-testid="stFileUploaderDropzone"],
        [data-testid="stExpander"] {
            border-color: var(--line);
            border-radius: 13px;
            background: rgba(255, 255, 255, 0.88);
        }
        [data-testid="stDataFrame"],
        [data-testid="stImage"] img {
            border-radius: 15px;
        }
        [data-testid="stAlert"] {
            border-radius: 13px;
        }
        @media (max-width: 920px) {
            .product-hero {
                grid-template-columns: 1fr;
                min-height: 0;
            }
            .hero-visual {
                display: none;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    registry = load_registry()
    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
          <span class="sidebar-brand-icon">⌕</span>
          <span>多模态资料工作台</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    library_ids = [row["id"] for row in registry["libraries"]]
    selected_library_id = st.session_state.get(
        "selected_library_id", registry["active_library_id"]
    )
    if selected_library_id not in library_ids:
        selected_library_id = registry["active_library_id"]
    selected_library_id = st.sidebar.selectbox(
        "当前资料库（严格隔离）",
        options=library_ids,
        index=library_ids.index(selected_library_id),
        format_func=lambda library_id: library_by_id(
            registry, library_id
        )["name"],
    )
    if selected_library_id != registry["active_library_id"]:
        set_active_library(selected_library_id)
        registry["active_library_id"] = selected_library_id
        st.session_state.pop("live_result", None)
    st.session_state["selected_library_id"] = selected_library_id
    selected_library = library_by_id(registry, selected_library_id)
    selected_library_dir = resolve_library_dir(selected_library)

    mode = st.sidebar.radio(
        "工作台",
        options=(
            "智能检索",
            "独立盲测",
            "批量入库",
            "资料库",
        ),
    )
    render_product_header(selected_library["name"])
    method_key = "quality_hybrid"
    top_k = 3
    if mode == "智能检索":
        method_key = st.sidebar.selectbox(
            "搜索模式",
            options=list(LIVE_METHOD_LABELS),
            format_func=LIVE_METHOD_LABELS.get,
            index=list(LIVE_METHOD_LABELS).index("quality_hybrid"),
        )
        top_k = st.sidebar.slider("显示Top-K", 1, 6, 3)

    if mode == "智能检索":
        render_live_search(
            method_key,
            top_k,
            selected_library,
            selected_library_dir,
        )
        st.caption(
            "模型在本机运行，不调用商业API；原图和查询不会上传网络。"
        )
        return
    if mode == "批量入库":
        render_upload(selected_library, selected_library_dir)
        st.caption("上传资料仅保存在本机，不调用商业API。")
        return
    if mode == "独立盲测":
        render_blind_query_collection(selected_library, selected_library_dir)
        st.caption(
            "盲测查询只保存在本机；采集阶段不会调用任何检索模型。"
        )
        return
    if mode == "资料库":
        render_library_manager(registry)
        return


if __name__ == "__main__":
    main()
