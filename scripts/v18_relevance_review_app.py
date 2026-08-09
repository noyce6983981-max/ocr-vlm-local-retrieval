"""V18 校准集候选页相关性盲审网页。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import streamlit as st

from ocr_vlm_retrieval.evaluation.judgments import (
    NO_RELEVANT_CANDIDATE_IN_POOL,
    POOLED_RELEVANCE_TASK,
    RELEVANT_CANDIDATE_IN_POOL,
)

DEFAULT_STUDY_DIR = PROJECT_ROOT / "data/evaluation/v18/human_study/calibration"
PACKETS_PATH = Path(
    os.environ.get(
        "V18_REVIEW_PACKETS", str(DEFAULT_STUDY_DIR / "review_packets.jsonl")
    )
).resolve()
JUDGMENTS_PATH = Path(
    os.environ.get(
        "V18_REVIEW_JUDGMENTS", str(DEFAULT_STUDY_DIR / "judgments.jsonl")
    )
).resolve()
RUNTIME_ROOT = Path(os.environ.get("V18_RUNTIME_ROOT", str(PROJECT_ROOT))).resolve()
FIXED_REVIEWER_ID = os.environ.get("V18_REVIEWER_ID", "").strip()
FIXED_REVIEWER_ROLE = os.environ.get("V18_REVIEWER_ROLE", "").strip()
SECONDARY_REVIEW_FRACTION = 0.30


def install_translation_guard() -> None:
    """阻止浏览器翻译插件修改 Streamlit 的 React 节点。"""

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
            f"{row['study_fingerprint']}\0secondary\0{row['query_id']}".encode()
        ).hexdigest(),
    )[:required]


def assigned_packets(
    packets: list[dict[str, Any]], reviewer_role: str
) -> list[dict[str, Any]]:
    if reviewer_role == "secondary":
        return secondary_packets(packets)
    if reviewer_role != "primary":
        raise ValueError("reviewer_role must be primary or secondary")
    return packets


def candidate_image_path(candidate: dict[str, Any]) -> Path | None:
    raw_path = str(candidate.get("review_asset", {}).get("image_path", "")).strip()
    if not raw_path:
        return None
    path = Path(raw_path)
    resolved = (path if path.is_absolute() else RUNTIME_ROOT / path).resolve()
    return resolved if resolved.is_file() else None


def packet_pool_sha256(packet: dict[str, Any]) -> str:
    existing = str(packet.get("pool_sha256", "")).strip()
    if existing:
        return existing
    material = {
        "task_id": POOLED_RELEVANCE_TASK,
        "query_id": str(packet["query_id"]),
        "study_fingerprint": str(packet["study_fingerprint"]),
        "item_ids": sorted(
            str(candidate["item_id"]) for candidate in packet["candidates"]
        ),
    }
    return hashlib.sha256(
        json.dumps(
            material,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def remember_relevance(draft_key: str, item_id: str, widget_key: str) -> None:
    draft = dict(st.session_state.get(draft_key, {}))
    draft[item_id] = bool(st.session_state.get(widget_key, False))
    st.session_state[draft_key] = draft


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


def move_cursor(cursor_key: str, delta: int, total: int) -> None:
    current = int(st.session_state.get(cursor_key, 1))
    st.session_state[cursor_key] = min(max(current + delta, 1), total)


def main() -> None:
    st.set_page_config(page_title="V18 校准集盲审", layout="wide")
    install_translation_guard()
    st.title("V18 校准集候选页相关性盲审")
    st.caption(
        "只根据查询和候选图片判断。页面不会显示检索方法、分数、排名、来源文件名或系统答案。"
    )

    packets = read_jsonl(PACKETS_PATH)
    if not packets:
        st.error("尚未生成 V18 校准集候选池，请等待检索和池化完成。")
        return
    if {str(row.get("split", "")) for row in packets} != {"calibration"}:
        st.error("数据包不是纯校准集，已停止显示。")
        return
    fingerprints = {str(row.get("study_fingerprint", "")) for row in packets}
    if len(fingerprints) != 1 or not next(iter(fingerprints)):
        st.error("审核数据包的研究指纹不一致。")
        return
    active_fingerprint = next(iter(fingerprints))

    reviewer_id = st.sidebar.text_input(
        "审核者 ID",
        value=FIXED_REVIEWER_ID,
        disabled=bool(FIXED_REVIEWER_ID),
        help="两名审核者必须使用不同 ID，并独立完成判断。",
    ).strip()
    if FIXED_REVIEWER_ROLE:
        reviewer_role = FIXED_REVIEWER_ROLE
        st.sidebar.info(
            "当前任务：第一审核者（全部 80 条）"
            if reviewer_role == "primary"
            else "当前任务：第二审核者（固定复核 24 条）"
        )
    else:
        role_label = st.sidebar.radio(
            "审核任务",
            ("第一审核者（全部 80 条）", "第二审核者（固定复核 24 条）"),
        )
        reviewer_role = "primary" if role_label.startswith("第一") else "secondary"
    if reviewer_role not in {"primary", "secondary"}:
        st.error("审核角色配置错误。")
        return
    assigned = assigned_packets(packets, reviewer_role)
    judgments = [
        row
        for row in read_jsonl(JUDGMENTS_PATH)
        if row.get("study_fingerprint") == active_fingerprint
    ]
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

    cursor_key = (
        f"v18_cursor::{reviewer_id}::{reviewer_role}::"
        f"{active_fingerprint}::{show_completed}"
    )
    if not 1 <= int(st.session_state.get(cursor_key, 1)) <= len(visible):
        st.session_state[cursor_key] = 1
    previous_col, position_col, next_col = st.columns([1, 2, 1])
    position = int(
        position_col.number_input(
            "查询序号",
            min_value=1,
            max_value=len(visible),
            step=1,
            key=cursor_key,
        )
    )
    previous_col.button(
        "← 上一条",
        disabled=position <= 1,
        on_click=move_cursor,
        args=(cursor_key, -1, len(visible)),
        width="stretch",
    )
    next_col.button(
        "下一条 →",
        disabled=position >= len(visible),
        on_click=move_cursor,
        args=(cursor_key, 1, len(visible)),
        width="stretch",
    )

    packet = visible[position - 1]
    query_id = str(packet["query_id"])
    existing = current_judgment(judgments, query_id, reviewer_id)
    st.markdown("#### 当前查询")
    st.info(f"{query_id} · {packet['query']}")
    st.write(
        "相关：候选图片同时满足查询中的全部必要条件。只匹配对象、颜色、场景、关系或文字中的一部分，仍判为不相关。"
    )

    candidates = list(packet["candidates"])
    existing_relevance = dict(existing.get("candidate_relevance", {}))
    draft_key = f"v18_draft::{reviewer_id}::{query_id}::{active_fingerprint}"
    if draft_key not in st.session_state:
        st.session_state[draft_key] = {
            str(candidate["item_id"]): bool(
                existing_relevance.get(str(candidate["item_id"]), False)
            )
            for candidate in candidates
        }
    columns = st.columns(4)
    for offset, candidate in enumerate(candidates):
        item_id = str(candidate["item_id"])
        candidate_id = str(candidate["candidate_id"])
        widget_key = f"v18_relevant::{reviewer_id}::{query_id}::{item_id}"
        if widget_key not in st.session_state:
            st.session_state[widget_key] = bool(
                st.session_state[draft_key].get(item_id, False)
            )
        with columns[offset % 4]:
            image_path = candidate_image_path(candidate)
            if image_path is None:
                st.warning(f"{candidate_id}：图片不可用")
            else:
                st.image(str(image_path), width="stretch")
            st.checkbox(
                f"{candidate_id}：完整相关",
                key=widget_key,
                on_change=remember_relevance,
                args=(draft_key, item_id, widget_key),
            )

    options = (RELEVANT_CANDIDATE_IN_POOL, NO_RELEVANT_CANDIDATE_IN_POOL)
    current_decision = str(existing.get("pool_relevance", ""))
    decision_index = options.index(current_decision) if current_decision in options else None
    pool_relevance = st.radio(
        "20 个候选中是否存在至少一个完整相关页面？",
        options,
        index=decision_index,
        format_func=lambda value: (
            "存在完整相关页面"
            if value == RELEVANT_CANDIDATE_IN_POOL
            else "没有完整相关页面"
        ),
        horizontal=True,
    )
    notes = st.text_area("备注（可选）", value=str(existing.get("notes", "")))
    if st.button("保存并完成本条", type="primary"):
        relevance = {
            str(candidate["item_id"]): bool(
                st.session_state[draft_key].get(str(candidate["item_id"]), False)
            )
            for candidate in candidates
        }
        relevant_count = sum(relevance.values())
        if pool_relevance is None:
            st.error("请先选择候选池最终判断。")
            return
        if pool_relevance == RELEVANT_CANDIDATE_IN_POOL and relevant_count == 0:
            st.error("选择“存在”时，至少标记一个完整相关候选。")
            return
        if pool_relevance == NO_RELEVANT_CANDIDATE_IN_POOL and relevant_count > 0:
            st.error("选择“没有”时，不能同时标记相关候选。")
            return
        save_judgment(
            {
                "query_id": query_id,
                "reviewer_id": reviewer_id,
                "reviewer_role": reviewer_role,
                "task_id": POOLED_RELEVANCE_TASK,
                "pool_relevance": pool_relevance,
                "pool_sha256": packet_pool_sha256(packet),
                "candidate_relevance": relevance,
                "notes": notes.strip(),
                "reviewed_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "study_fingerprint": active_fingerprint,
            }
        )
        st.success(f"已保存：{relevant_count} 个完整相关候选。")
        st.rerun()


if __name__ == "__main__":
    main()
