"""V18 校准集配对候选页相关性盲审网页。"""

from __future__ import annotations

import hashlib
import math
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import streamlit as st

from ocr_vlm_retrieval.evaluation.judgments import (
    NO_RELEVANT_CANDIDATE_IN_POOL,
    POOLED_RELEVANCE_TASK,
    RELEVANT_CANDIDATE_IN_POOL,
)
from scripts import v18_relevance_review_app as base

EXPECTED_SPLIT = os.environ.get("V18_REVIEW_SPLIT", "calibration").strip()


def packet_pairs(
    packets: Sequence[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Group the frozen positive/contrast query rows into 40 blind tasks."""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for packet in packets:
        group_id = str(packet.get("group_id", "")).strip()
        if not group_id:
            raise ValueError("Every paired-review packet must have a group_id")
        grouped.setdefault(group_id, []).append(packet)
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for group_id in sorted(grouped):
        rows = sorted(grouped[group_id], key=lambda row: str(row["query_id"]))
        if len(rows) != 2:
            raise ValueError(f"{group_id} must contain exactly two queries")
        if rows[0]["query_id"] == rows[1]["query_id"]:
            raise ValueError(f"{group_id} contains a duplicate query_id")
        pairs.append((rows[0], rows[1]))
    return pairs


def secondary_pairs(
    pairs: Sequence[tuple[dict[str, Any], dict[str, Any]]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    required = math.ceil(len(pairs) * base.SECONDARY_REVIEW_FRACTION)
    return sorted(
        pairs,
        key=lambda pair: hashlib.sha256(
            (
                f"{pair[0]['study_fingerprint']}\0secondary-pair\0"
                f"{pair[0]['group_id']}"
            ).encode()
        ).hexdigest(),
    )[:required]


def assigned_pairs(
    pairs: Sequence[tuple[dict[str, Any], dict[str, Any]]], reviewer_role: str
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    if reviewer_role == "secondary":
        return secondary_pairs(pairs)
    if reviewer_role not in {"primary", "adjudicator"}:
        raise ValueError("reviewer_role must be primary, secondary, or adjudicator")
    return list(pairs)


def union_candidates(
    pair: tuple[dict[str, Any], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep first-seen blind order while rendering shared images only once."""

    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for packet in pair:
        for candidate in packet["candidates"]:
            item_id = str(candidate["item_id"])
            if item_id in seen:
                continue
            seen.add(item_id)
            merged.append(candidate)
    return merged


def save_pair_judgments(judgments: Sequence[dict[str, Any]]) -> None:
    indexed = {
        (str(row["query_id"]), str(row["reviewer_id"])): row
        for row in base.read_jsonl(base.JUDGMENTS_PATH)
    }
    for judgment in judgments:
        key = (str(judgment["query_id"]), str(judgment["reviewer_id"]))
        indexed[key] = judgment
    base.write_jsonl_atomic(
        base.JUDGMENTS_PATH,
        [indexed[key] for key in sorted(indexed)],
    )


def _pair_complete(
    pair: tuple[dict[str, Any], dict[str, Any]], completed_ids: set[str]
) -> bool:
    return all(str(packet["query_id"]) in completed_ids for packet in pair)


def main() -> None:
    st.set_page_config(page_title="V18 配对盲审", layout="wide")
    base.install_translation_guard()
    split_label = "留出集" if EXPECTED_SPLIT == "holdout" else "校准集"
    st.title(f"V18 {split_label}配对候选页相关性盲审")
    st.caption(
        "当前页面仅显示本审核角色分配的配对任务；同一图片只展示一次。"
        "只根据查询和图片判断，不显示检索方法、分数、排名、来源文件名或系统答案。"
    )

    packets = base.read_jsonl(base.PACKETS_PATH)
    if not packets:
        st.error("尚未生成 V18 校准集候选池。")
        return
    if {str(row.get("split", "")) for row in packets} != {EXPECTED_SPLIT}:
        st.error(f"数据包不是纯{split_label}，已停止显示。")
        return
    fingerprints = {str(row.get("study_fingerprint", "")) for row in packets}
    if len(fingerprints) != 1 or not next(iter(fingerprints)):
        st.error("审核数据包的研究指纹不一致。")
        return
    active_fingerprint = next(iter(fingerprints))
    try:
        pairs = packet_pairs(packets)
    except ValueError as error:
        st.error(str(error))
        return

    reviewer_id = st.sidebar.text_input(
        "审核者 ID",
        value=base.FIXED_REVIEWER_ID,
        disabled=bool(base.FIXED_REVIEWER_ID),
    ).strip()
    if base.FIXED_REVIEWER_ROLE:
        reviewer_role = base.FIXED_REVIEWER_ROLE
        if reviewer_role == "primary":
            role_status = "当前任务：第一审核者"
        elif reviewer_role == "secondary":
            role_status = "当前任务：第二审核者（固定复核）"
        else:
            role_status = "当前任务：冲突裁决者"
        st.sidebar.info(f"{role_status}（{len(pairs)} 组来源池）")
    else:
        role_label = st.sidebar.radio(
            "审核任务",
            ("第一审核者（40 组）", "第二审核者（固定复核 12 组）"),
        )
        reviewer_role = "primary" if role_label.startswith("第一") else "secondary"
    if reviewer_role not in {"primary", "secondary", "adjudicator"}:
        st.error("审核角色配置错误。")
        return

    assigned = assigned_pairs(pairs, reviewer_role)
    judgments = [
        row
        for row in base.read_jsonl(base.JUDGMENTS_PATH)
        if row.get("study_fingerprint") == active_fingerprint
    ]
    completed_ids = {
        str(row["query_id"])
        for row in judgments
        if reviewer_id and row.get("reviewer_id") == reviewer_id
    }
    pending = [pair for pair in assigned if not _pair_complete(pair, completed_ids)]
    st.sidebar.metric("分配组数", len(assigned))
    st.sidebar.metric("已完成", len(assigned) - len(pending))
    st.sidebar.metric("待完成", len(pending))
    show_completed = st.sidebar.checkbox("显示已完成组", value=False)
    visible = assigned if show_completed else pending
    if not reviewer_id:
        st.info("请先在左侧填写审核者 ID。")
        return
    if not visible:
        st.success("当前 40 组审核任务已全部完成。")
        return

    cursor_key = (
        f"v18_pair_cursor::{reviewer_id}::{reviewer_role}::"
        f"{active_fingerprint}::{show_completed}"
    )
    if not 1 <= int(st.session_state.get(cursor_key, 1)) <= len(visible):
        st.session_state[cursor_key] = 1
    previous_col, position_col, next_col = st.columns([1, 2, 1])
    position = int(
        position_col.number_input(
            "配对任务序号",
            min_value=1,
            max_value=len(visible),
            step=1,
            key=cursor_key,
        )
    )
    previous_col.button(
        "← 上一组",
        disabled=position <= 1,
        on_click=base.move_cursor,
        args=(cursor_key, -1, len(visible)),
        width="stretch",
    )
    next_col.button(
        "下一组 →",
        disabled=position >= len(visible),
        on_click=base.move_cursor,
        args=(cursor_key, 1, len(visible)),
        width="stretch",
    )

    pair = visible[position - 1]
    labels = ("A", "B")
    existing = {
        str(packet["query_id"]): base.current_judgment(
            judgments, str(packet["query_id"]), reviewer_id
        )
        for packet in pair
    }
    st.markdown("#### 本组两条查询")
    for label, packet in zip(labels, pair, strict=True):
        st.info(f"查询 {label}：{packet['query']}")
    st.write(
        "对每张图分别判断 A、B：只有同时满足该查询的全部必要条件才勾选。"
        "只满足部分对象、颜色、场景、关系或文字仍不勾选。"
    )

    drafts: dict[str, str] = {}
    candidate_ids_by_query: dict[str, set[str]] = {}
    for packet in pair:
        query_id = str(packet["query_id"])
        candidates = list(packet["candidates"])
        candidate_ids_by_query[query_id] = {
            str(candidate["item_id"]) for candidate in candidates
        }
        draft_key = (
            f"v18_pair_draft::{reviewer_id}::{query_id}::{active_fingerprint}"
        )
        drafts[query_id] = draft_key
        if draft_key not in st.session_state:
            previous = dict(existing[query_id].get("candidate_relevance", {}))
            st.session_state[draft_key] = {
                str(candidate["item_id"]): bool(
                    previous.get(str(candidate["item_id"]), False)
                )
                for candidate in candidates
            }

    merged_candidates = union_candidates(pair)
    columns = st.columns(4)
    for offset, candidate in enumerate(merged_candidates):
        item_id = str(candidate["item_id"])
        display_id = f"候选 {offset + 1:02d}"
        with columns[offset % 4]:
            image_path = base.candidate_image_path(candidate)
            if image_path is None:
                st.warning(f"{display_id}：图片不可用")
            else:
                st.image(str(image_path), width="stretch")
            st.caption(display_id)
            for label, packet in zip(labels, pair, strict=True):
                query_id = str(packet["query_id"])
                if item_id not in candidate_ids_by_query[query_id]:
                    st.caption(f"查询 {label} 的池中无此图")
                    continue
                widget_key = (
                    f"v18_pair_relevant::{reviewer_id}::{query_id}::{item_id}"
                )
                if widget_key not in st.session_state:
                    st.session_state[widget_key] = bool(
                        st.session_state[drafts[query_id]].get(item_id, False)
                    )
                st.checkbox(
                    f"满足查询 {label}",
                    key=widget_key,
                    on_change=base.remember_relevance,
                    args=(drafts[query_id], item_id, widget_key),
                )

    st.caption(
        "保存时：某条查询只要勾选一张图，就记为“池中存在相关页”；"
        "完全不勾选则明确记为“池中无相关页”。"
    )
    note_values = {
        str(row.get("notes", ""))
        for row in existing.values()
        if str(row.get("notes", "")).strip()
    }
    notes = st.text_area(
        "本组备注（可选）",
        value=next(iter(note_values)) if len(note_values) == 1 else "",
    )
    if st.button("保存并完成本组", type="primary"):
        saved: list[dict[str, Any]] = []
        counts: list[int] = []
        reviewed_at = datetime.now(UTC).isoformat(timespec="seconds")
        for packet in pair:
            query_id = str(packet["query_id"])
            relevance = {
                str(candidate["item_id"]): bool(
                    st.session_state[drafts[query_id]].get(
                        str(candidate["item_id"]), False
                    )
                )
                for candidate in packet["candidates"]
            }
            relevant_count = sum(relevance.values())
            counts.append(relevant_count)
            pool_relevance = (
                RELEVANT_CANDIDATE_IN_POOL
                if relevant_count
                else NO_RELEVANT_CANDIDATE_IN_POOL
            )
            saved.append(
                {
                    "query_id": query_id,
                    "reviewer_id": reviewer_id,
                    "reviewer_role": reviewer_role,
                    "task_id": POOLED_RELEVANCE_TASK,
                    "pool_relevance": pool_relevance,
                    "pool_sha256": base.packet_pool_sha256(packet),
                    "candidate_relevance": relevance,
                    "notes": notes.strip(),
                    "reviewed_at": reviewed_at,
                    "study_fingerprint": active_fingerprint,
                }
            )
        save_pair_judgments(saved)
        st.success(f"本组已保存：查询 A {counts[0]} 张，查询 B {counts[1]} 张。")
        st.rerun()


if __name__ == "__main__":
    main()
