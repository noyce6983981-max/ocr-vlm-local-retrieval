"""Blinded local UI for V17 calibration relevance judgments."""

from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKETS_PATH = (
    PROJECT_ROOT / "data/evaluation/v17/human_study/calibration/review_packets.jsonl"
)
JUDGMENTS_PATH = (
    PROJECT_ROOT / "data/evaluation/v17/human_study/calibration/judgments.jsonl"
)
CANDIDATES_PER_PAGE = 20
SECONDARY_REVIEW_FRACTION = 0.30


def install_translation_guard() -> None:
    """Prevent browser translators from mutating Streamlit's React-owned DOM."""
    st.html(
        """
        <script>
        (() => {
          const doc = document;
          doc.documentElement.lang = "zh-CN";
          doc.documentElement.setAttribute("translate", "no");
          doc.documentElement.classList.add("notranslate");
          if (doc.body) {
            doc.body.setAttribute("translate", "no");
            doc.body.classList.add("notranslate");
          }
          let meta = doc.head.querySelector('meta[name="google"]');
          if (!meta) {
            meta = doc.createElement("meta");
            meta.name = "google";
            doc.head.appendChild(meta);
          }
          meta.content = "notranslate";
        })();
        </script>
        """,
        unsafe_allow_javascript=True,
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


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


def save_judgment(judgment: dict[str, Any]) -> None:
    indexed = {
        (str(row["query_id"]), str(row["reviewer_id"])): row
        for row in read_jsonl(JUDGMENTS_PATH)
    }
    key = (str(judgment["query_id"]), str(judgment["reviewer_id"]))
    indexed[key] = judgment
    write_jsonl_atomic(
        JUDGMENTS_PATH,
        [indexed[row_key] for row_key in sorted(indexed)],
    )


def secondary_packets(packets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    required = math.ceil(len(packets) * SECONDARY_REVIEW_FRACTION)
    return sorted(
        packets,
        key=lambda row: hashlib.sha256(
            (f"{row['study_fingerprint']}\0secondary\0{row['query_id']}").encode()
        ).hexdigest(),
    )[:required]


def packet_index(
    packets: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for packet in packets:
        query_id = str(packet.get("query_id", "")).strip()
        query = str(packet.get("query", "")).strip()
        if not query_id or not query or query_id in indexed:
            raise ValueError("Reviewer packets need unique query IDs and queries")
        indexed[query_id] = packet
    return indexed


def candidate_image_path(candidate: dict[str, Any]) -> Path | None:
    raw_path = str(candidate.get("review_metadata", {}).get("image_path", ""))
    if not raw_path:
        return None
    path = Path(raw_path)
    resolved = (path if path.is_absolute() else PROJECT_ROOT / path).resolve()
    return resolved if resolved.is_file() else None


def current_judgment(
    judgments: list[dict[str, Any]], query_id: str, reviewer_id: str
) -> dict[str, Any]:
    return next(
        (
            row
            for row in judgments
            if row.get("query_id") == query_id and row.get("reviewer_id") == reviewer_id
        ),
        {},
    )


def remember_relevance(draft_key: str, item_id: str, widget_key: str) -> None:
    """Persist checkbox changes outside Streamlit's transient widget state."""
    st.session_state[draft_key] = updated_relevance_draft(
        st.session_state.get(draft_key, {}),
        item_id,
        bool(st.session_state.get(widget_key, False)),
    )


def updated_relevance_draft(
    draft: dict[str, bool], item_id: str, checked: bool
) -> dict[str, bool]:
    updated = dict(draft)
    updated[item_id] = checked
    return updated


def moved_query_position(position: int, delta: int, total: int) -> int:
    if total < 1:
        raise ValueError("total must be positive")
    return min(max(position + delta, 1), total)


def move_query_position(cursor_key: str, delta: int, total: int) -> None:
    current = int(st.session_state.get(cursor_key, 1))
    st.session_state[cursor_key] = moved_query_position(current, delta, total)


def main() -> None:
    st.set_page_config(page_title="V17 相关性盲审", layout="wide")
    install_translation_guard()
    st.title("V17 校准集相关性盲审")
    st.caption(
        "判断候选页是否同时满足查询的全部必要条件。页面不会显示检索方法、"
        "分数、原始排名、系统接受决定或预期答案。"
    )
    st.warning(
        "为避免浏览器自动翻译破坏页面状态或改变冻结查询语义，本页已禁用自动"
        "翻译；查询始终显示审核冻结的英文原文。"
    )

    packets = read_jsonl(PACKETS_PATH)
    if not packets:
        st.error("尚未生成校准候选池 review_packets.jsonl。")
        return
    fingerprints = {str(row.get("study_fingerprint", "")) for row in packets}
    if len(fingerprints) != 1 or not next(iter(fingerprints)):
        st.error("审核数据包的 study_fingerprint 不一致。")
        return
    active_fingerprint = next(iter(fingerprints))
    judgments = [
        row
        for row in read_jsonl(JUDGMENTS_PATH)
        if row.get("study_fingerprint") == active_fingerprint
    ]
    reviewer_id = st.sidebar.text_input(
        "审核者 ID",
        help="两名审核者必须使用不同 ID，且不要共享审核结果。",
    ).strip()
    role = st.sidebar.radio(
        "审核任务",
        ("第一审核者（全部 40 条）", "第二审核者（固定复核 12 条）"),
    )
    assigned = packets if role.startswith("第一") else secondary_packets(packets)
    completed_ids = {
        str(row["query_id"])
        for row in judgments
        if reviewer_id and row.get("reviewer_id") == reviewer_id
    }
    pending = [row for row in assigned if row["query_id"] not in completed_ids]
    st.sidebar.metric("分配查询", len(assigned))
    st.sidebar.metric("已完成", len(assigned) - len(pending))
    st.sidebar.metric("待完成", len(pending))
    show_completed = st.sidebar.checkbox("显示已完成查询", value=False)
    visible = assigned if show_completed else pending

    if not reviewer_id:
        st.info("请先在左侧填写审核者 ID。")
        return
    if not visible:
        st.success("当前审核任务已全部完成。")
        return

    visible_by_id = packet_index(visible)
    query_ids = list(visible_by_id)
    cursor_key = (
        f"query_position::{reviewer_id}::{role}::{active_fingerprint}::{show_completed}"
    )
    current_position = int(st.session_state.get(cursor_key, 1))
    if not 1 <= current_position <= len(query_ids):
        st.session_state[cursor_key] = 1
    previous_column, position_column, next_column = st.columns([1, 2, 1])
    position = int(
        position_column.number_input(
            "查询序号",
            min_value=1,
            max_value=len(query_ids),
            step=1,
            key=cursor_key,
            help="输入序号或使用两侧按钮；当前审核对象以绑定卡片为准。",
        )
    )
    previous_column.button(
        "← 上一条",
        disabled=position <= 1,
        on_click=move_query_position,
        args=(cursor_key, -1, len(query_ids)),
        width="stretch",
    )
    next_column.button(
        "下一条 →",
        disabled=position >= len(query_ids),
        on_click=move_query_position,
        args=(cursor_key, 1, len(query_ids)),
        width="stretch",
    )
    selected_id = query_ids[position - 1]
    packet = visible_by_id[selected_id]
    if str(packet["query_id"]) != selected_id:
        st.error("查询选择器与审核数据包绑定不一致，已停止显示。")
        return
    existing = current_judgment(judgments, selected_id, reviewer_id)
    known_relevant_outside = list(
        existing.get("migration", {}).get("known_relevant_dropped_item_ids", [])
    )
    st.markdown("#### 当前审核对象")
    st.info(f"{selected_id} · {packet['query']}")
    st.caption(
        f"第 {position}/{len(query_ids)} 条 · 候选池绑定：{selected_id} · study "
        f"{active_fingerprint[:12]} · 以此处为当前生效选择"
    )
    st.write(
        "相关：候选页中的对象、属性、场景、关系和文字证据足以满足整条查询；"
        "只匹配其中一个显眼属性仍应判为不相关。"
    )
    if known_relevant_outside:
        st.info(
            f"这条已从旧候选池迁移：有 {len(known_relevant_outside)} 个已知相关项"
            "位于新的 20 项池外，原判断已保留，无需重做。"
        )

    candidates = list(packet["candidates"])
    page_count = math.ceil(len(candidates) / CANDIDATES_PER_PAGE)
    if page_count == 1:
        page_number = 1
        st.caption(f"本条共 {len(candidates)} 个候选，已在当前页全部展示。")
    else:
        page_number = st.select_slider(
            "候选页",
            options=list(range(1, page_count + 1)),
            value=1,
            format_func=lambda value: f"第 {value}/{page_count} 页",
        )
    seen_key = f"seen::{reviewer_id}::{selected_id}"
    if seen_key not in st.session_state:
        st.session_state[seen_key] = set()
    st.session_state[seen_key].add(page_number)

    start = (page_number - 1) * CANDIDATES_PER_PAGE
    page_candidates = candidates[start : start + CANDIDATES_PER_PAGE]
    columns = st.columns(4)
    existing_relevance = dict(existing.get("candidate_relevance", {}))
    draft_key = f"draft::{reviewer_id}::{selected_id}::{active_fingerprint}"
    if draft_key not in st.session_state:
        st.session_state[draft_key] = {
            str(candidate["item_id"]): bool(
                existing_relevance.get(str(candidate["item_id"]), False)
            )
            for candidate in candidates
        }
    for offset, candidate in enumerate(page_candidates):
        item_id = str(candidate["item_id"])
        candidate_id = str(candidate["candidate_id"])
        metadata = dict(candidate.get("review_metadata", {}))
        state_key = f"relevant::{reviewer_id}::{selected_id}::{item_id}"
        if state_key not in st.session_state:
            st.session_state[state_key] = bool(
                st.session_state[draft_key].get(item_id, False)
            )
        with columns[offset % 4]:
            image_path = candidate_image_path(candidate)
            if image_path is None:
                st.warning(f"{candidate_id}：图像文件不可用")
            else:
                st.image(str(image_path), width="stretch")
            st.checkbox(
                f"{candidate_id}：相关",
                key=state_key,
                on_change=remember_relevance,
                args=(draft_key, item_id, state_key),
            )
            st.caption(
                " · ".join(
                    str(value)
                    for value in (
                        metadata.get("title"),
                        metadata.get("source_name"),
                        metadata.get("source_relpath"),
                        (
                            f"第 {metadata['page_number']} 页"
                            if metadata.get("page_number") is not None
                            else None
                        ),
                    )
                    if value
                )
            )

    st.progress(len(st.session_state[seen_key]) / page_count)
    st.caption(
        f"已查看 {len(st.session_state[seen_key])}/{page_count} 个候选分页；"
        "提交前必须逐页查看。"
    )
    answerability_options = ("uncertain", "answerable", "no_answer", "excluded")
    current_answerability = str(existing.get("answerability", "uncertain"))
    answerability = st.radio(
        "整条查询的答案性",
        answerability_options,
        index=answerability_options.index(current_answerability),
        horizontal=True,
        help=(
            "answerable：池中至少一个完整相关候选；no_answer：池中没有完整相关"
            "候选；uncertain：证据不足；excluded：查询本身无效或无法判定。"
        ),
    )
    notes = st.text_area("审核备注（可选）", value=str(existing.get("notes", "")))
    if st.button(f"保存 {selected_id} 的判断", type="primary"):
        if len(st.session_state[seen_key]) != page_count:
            st.error("请先逐页查看全部候选，再保存判断。")
            return
        relevance = {
            str(candidate["item_id"]): bool(
                st.session_state[draft_key].get(str(candidate["item_id"]), False)
            )
            for candidate in candidates
        }
        relevant_count = sum(relevance.values())
        if (
            answerability == "answerable"
            and relevant_count == 0
            and not known_relevant_outside
        ):
            st.error("选择 answerable 时，至少应标记一个完整相关候选。")
            return
        if answerability == "no_answer" and relevant_count > 0:
            st.error("选择 no_answer 时，不能同时标记相关候选。")
            return
        save_judgment(
            {
                "query_id": selected_id,
                "reviewer_id": reviewer_id,
                "reviewer_role": "primary" if role.startswith("第一") else "secondary",
                "answerability": answerability,
                "candidate_relevance": relevance,
                "notes": notes.strip(),
                "reviewed_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "study_fingerprint": packet["study_fingerprint"],
                **(
                    {
                        "migration": existing["migration"],
                        "migrated_at": existing.get("migrated_at"),
                    }
                    if existing.get("migration")
                    else {}
                ),
            }
        )
        st.success(f"已保存：{relevant_count} 个完整相关候选。")
        st.rerun()


if __name__ == "__main__":
    main()
