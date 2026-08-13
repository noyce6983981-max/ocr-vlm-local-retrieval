"""Human review UI for the 12 V19.1 development query families only."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_DIR = ROOT / "records/private/v19_1/condition_completeness"
FAMILY_PATH = PRIVATE_DIR / "source_families_draft.jsonl"
DRAFT_PATH = PRIVATE_DIR / "development_query_drafts_machine.json"
ROLES = (
    "answerable_positive",
    "paraphrase_positive",
    "single_condition_hard_negative",
    "unanswerable_neighbor",
)
ROLE_LABELS = {
    "answerable_positive": "可回答正例（目标页满足）",
    "paraphrase_positive": "自然改写正例（目标页满足）",
    "single_condition_hard_negative": "单条件强负例（资料库不满足）",
    "unanswerable_neighbor": "近邻无答案（资料库不满足）",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def load_families() -> list[dict[str, Any]]:
    source_rows = [
        row for row in _read_jsonl(FAMILY_PATH) if row["split"] == "development"
    ]
    draft_payload = json.loads(DRAFT_PATH.read_text(encoding="utf-8-sig"))
    if draft_payload.get("holdout_ocr_opened") is not False:
        raise ValueError("holdout OCR must remain closed during development review")
    drafts = {str(row["family_id"]): row for row in draft_payload.get("families", [])}
    if (
        len(source_rows) != 12
        or {str(row["family_id"]) for row in source_rows} != drafts.keys()
    ):
        raise ValueError("expected 12 matched V19.1 development families")
    families: list[dict[str, Any]] = []
    for source in source_rows:
        family_id = str(source["family_id"])
        families.append({**source, **drafts[family_id]})
    return families


def _snapshot_sha256(family: Mapping[str, Any]) -> str:
    material = json.dumps(
        dict(family), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _review_path(reviewer_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", reviewer_id):
        raise ValueError("审核者 ID 只能包含字母、数字、下划线或连字符")
    return PRIVATE_DIR / f"{reviewer_id}_development_query_reviews.jsonl"


def _write_jsonl_atomic(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _load_reviews(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    rows = _read_jsonl(path)
    if len(rows) != len({str(row["family_id"]) for row in rows}):
        raise ValueError("审核记录包含重复家族")
    return {str(row["family_id"]): row for row in rows}


def _save_review(
    path: Path,
    *,
    family: Mapping[str, Any],
    reviewer_id: str,
    changed_condition_kind: str,
    query_texts: Mapping[str, str],
    notes: str,
) -> dict[str, dict[str, Any]]:
    cleaned = {role: " ".join(str(query_texts[role]).split()) for role in ROLES}
    if any(len(cleaned[role]) < 8 for role in ROLES):
        raise ValueError("四条查询均需至少 8 个字符")
    if len(set(cleaned.values())) != 4:
        raise ValueError("同一家族四条查询不能重复")
    reviews = _load_reviews(path)
    family_id = str(family["family_id"])
    reviews[family_id] = {
        "schema_version": 1,
        "family_id": family_id,
        "family_snapshot_sha256": _snapshot_sha256(family),
        "reviewer_id": reviewer_id,
        "reviewed_at_unix": round(time.time(), 6),
        "decision": "human_approved_development_family",
        "changed_condition_kind": changed_condition_kind.strip(),
        "query_texts": cleaned,
        "notes": notes.strip(),
    }
    _write_jsonl_atomic(path, [reviews[key] for key in sorted(reviews)])
    return reviews


def _image_path(snapshot: Mapping[str, Any]) -> Path:
    return ROOT / str(snapshot["image_path"])


def render() -> None:
    st.set_page_config(page_title="V19.1 开发查询审核", layout="wide")
    st.title("V19.1 必要条件完整性：开发查询审核")
    st.caption(
        "只审核 12 个 development 家族；holdout 图像、OCR 和查询尚未打开。"
        "每个家族确认一次四条配对查询即可。"
    )
    families = load_families()
    reviewer_id = st.sidebar.text_input("审核者 ID", value="reviewer_01").strip()
    try:
        review_path = _review_path(reviewer_id)
    except ValueError as error:
        st.error(str(error))
        st.stop()
    reviews = _load_reviews(review_path)
    st.sidebar.metric("已完成家族", f"{len(reviews)}/12")
    st.sidebar.info(
        "正例两条必须由左侧目标页完整满足；后两条必须确实无答案，且强负例只能改一个必要条件。"
    )

    options = [str(row["family_id"]) for row in families]
    default_index = next(
        (index for index, family_id in enumerate(options) if family_id not in reviews),
        0,
    )
    selection_key = "v19_1_selected_development_family"
    advance_key = "v19_1_advance_to_family"
    advance_to = st.session_state.pop(advance_key, None)
    if advance_to in options:
        st.session_state[selection_key] = advance_to
    elif st.session_state.get(selection_key) not in options:
        st.session_state[selection_key] = options[default_index]
    selected_id = st.selectbox(
        "选择家族",
        options,
        index=default_index,
        key=selection_key,
        format_func=lambda value: f"{'✓' if value in reviews else '○'} {value}",
    )
    family = next(row for row in families if row["family_id"] == selected_id)
    existing = reviews.get(selected_id)
    st.subheader(
        f"{selected_id} · {family['content_stratum']} · "
        f"{family['changed_condition_kind']}"
    )
    left, right = st.columns(2)
    with left:
        st.markdown("**目标页（正例应完整满足）**")
        st.image(str(_image_path(family["target"])), use_container_width=True)
        st.caption(family["target"]["item_id"])
    with right:
        st.markdown("**近邻页（用于构造无答案干扰）**")
        st.image(str(_image_path(family["neighbor"])), use_container_width=True)
        st.caption(family["neighbor"]["item_id"])

    defaults = existing["query_texts"] if existing else family["query_drafts"]
    query_texts: dict[str, str] = {}
    for role in ROLES:
        query_texts[role] = st.text_area(
            ROLE_LABELS[role],
            value=str(defaults[role]),
            key=f"{selected_id}_{role}",
            height=75,
        )
    changed_condition_kind = st.text_input(
        "强负例唯一改变的必要条件",
        value=str(
            existing["changed_condition_kind"]
            if existing
            else family["changed_condition_kind"]
        ),
        key=f"{selected_id}_condition_kind",
    )
    notes = st.text_area(
        "备注（可选）",
        value=str(existing.get("notes", "") if existing else ""),
        key=f"{selected_id}_notes",
        height=65,
    )
    attested = st.checkbox(
        "我已核对：两条正例完整满足；强负例只改变一个必要条件；近邻负例无答案",
        key=f"{selected_id}_attested",
    )
    if st.button("保存本家族", type="primary", disabled=not attested):
        try:
            reviews = _save_review(
                review_path,
                family=family,
                reviewer_id=reviewer_id,
                changed_condition_kind=changed_condition_kind,
                query_texts=query_texts,
                notes=notes,
            )
        except ValueError as error:
            st.error(str(error))
        else:
            st.success(f"已保存 {selected_id}；当前完成 {len(reviews)}/12")
            remaining = [family_id for family_id in options if family_id not in reviews]
            if remaining:
                st.session_state[advance_key] = remaining[0]
            st.rerun()
    if len(reviews) == 12:
        st.success("12 个 development 家族已全部完成，可以编译人工审核开发集。")
    st.caption(f"审核记录：{review_path}")


if __name__ == "__main__":
    render()
