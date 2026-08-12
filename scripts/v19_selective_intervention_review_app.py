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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FAMILY_PATH = (
    PROJECT_ROOT
    / "records/private/v19/selective_intervention/source_families_draft.jsonl"
)
PRIVATE_REVIEW_DIR = PROJECT_ROOT / "records/private/v19/selective_intervention"
ROLES = (
    "answerable_positive",
    "paraphrase_positive",
    "single_condition_hard_negative",
    "unanswerable_neighbor",
)
ROLE_LABELS = {
    "answerable_positive": "可回答正例",
    "paraphrase_positive": "自然改写正例",
    "single_condition_hard_negative": "单条件强负例",
    "unanswerable_neighbor": "近邻无答案",
}
CONDITION_KINDS = (
    "attribute",
    "color",
    "date",
    "entity",
    "identifier",
    "layout",
    "number",
    "object_count",
    "orientation",
    "position",
    "relation",
    "scene",
    "text",
    "time_scene",
    "topic",
)


def load_families(path: Path = FAMILY_PATH) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len(rows) != 50:
        raise ValueError(f"expected 50 source-bound families, got {len(rows)}")
    if len({str(row["family_id"]) for row in rows}) != len(rows):
        raise ValueError("family IDs must be unique")
    for row in rows:
        if set(row.get("query_drafts", {})) != set(ROLES):
            raise ValueError(f"family {row['family_id']} is missing query roles")
    return rows


def family_snapshot_sha256(family: Mapping[str, Any]) -> str:
    snapshot = {
        "family_id": family["family_id"],
        "split": family["split"],
        "content_stratum": family["content_stratum"],
        "target": family["target"],
        "neighbor": family["neighbor"],
        "query_drafts": family["query_drafts"],
        "changed_condition_kind": family["changed_condition_kind"],
    }
    material = json.dumps(
        snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def review_path_for(reviewer_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", reviewer_id):
        raise ValueError("审核者ID只能包含字母、数字、下划线或连字符")
    return PRIVATE_REVIEW_DIR / f"{reviewer_id}_family_corrections.jsonl"


def completion_path_for(reviewer_id: str) -> Path:
    review_path_for(reviewer_id)
    return PRIVATE_REVIEW_DIR / f"{reviewer_id}_completion.json"


def progress_path_for(reviewer_id: str) -> Path:
    review_path_for(reviewer_id)
    return PRIVATE_REVIEW_DIR / f"{reviewer_id}_browse_progress.json"


def load_reviews(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    family_ids = [str(row["family_id"]) for row in rows]
    if len(family_ids) != len(set(family_ids)):
        raise ValueError("review file contains duplicate family IDs")
    return {str(row["family_id"]): row for row in rows}


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


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _family_bundle_sha256(families: list[dict[str, Any]]) -> str:
    snapshot_material = "\n".join(
        family_snapshot_sha256(family) for family in families
    )
    return hashlib.sha256(snapshot_material.encode("utf-8")).hexdigest()


def mark_family_seen(
    path: Path,
    *,
    reviewer_id: str,
    families: list[dict[str, Any]],
    family_id: str,
) -> set[str]:
    expected = {str(row["family_id"]) for row in families}
    if family_id not in expected:
        raise ValueError("browse progress contains an unknown family")
    bundle_sha256 = _family_bundle_sha256(families)
    seen: set[str] = set()
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        if str(payload.get("reviewer_id", "")) != reviewer_id:
            raise ValueError("browse progress reviewer does not match")
        if str(payload.get("family_bundle_sha256", "")) != bundle_sha256:
            raise ValueError("family bundle changed after browse progress was saved")
        seen = {str(value) for value in payload.get("seen_family_ids", [])}
        if not seen <= expected:
            raise ValueError("browse progress contains unknown family IDs")
    seen.add(family_id)
    _write_json_atomic(
        path,
        {
            "schema_version": 1,
            "reviewer_id": reviewer_id,
            "family_bundle_sha256": bundle_sha256,
            "seen_family_ids": sorted(seen),
            "seen_family_count": len(seen),
            "updated_at_unix": round(time.time(), 6),
        },
    )
    return seen


def save_review(
    path: Path,
    *,
    family: Mapping[str, Any],
    query_texts: Mapping[str, str],
    changed_condition_kind: str,
    pair_decision: str,
    notes: str,
    reviewer_id: str,
) -> dict[str, dict[str, Any]]:
    if set(query_texts) != set(ROLES):
        raise ValueError("all four query roles are required")
    if any(len(str(query_texts[role]).strip()) < 8 for role in ROLES):
        raise ValueError("each query must contain at least eight characters")
    if len({str(query_texts[role]).strip() for role in ROLES}) != len(ROLES):
        raise ValueError("the four queries in a family must be unique")
    if changed_condition_kind not in CONDITION_KINDS:
        raise ValueError("changed condition kind is invalid")
    if pair_decision not in {"accept", "replace_pair"}:
        raise ValueError("pair decision is invalid")
    reviews = load_reviews(path)
    family_id = str(family["family_id"])
    snapshot = family_snapshot_sha256(family)
    existing = reviews.get(family_id)
    if existing and str(existing["family_snapshot_sha256"]) != snapshot:
        raise ValueError("family content changed after an earlier correction")
    reviews[family_id] = {
        "family_id": family_id,
        "family_snapshot_sha256": snapshot,
        "reviewer_id": reviewer_id,
        "reviewed_at_unix": round(time.time(), 6),
        "pair_decision": pair_decision,
        "changed_condition_kind": changed_condition_kind,
        "query_texts": {role: str(query_texts[role]).strip() for role in ROLES},
        "notes": notes.strip(),
    }
    _write_jsonl_atomic(path, [reviews[key] for key in sorted(reviews)])
    return reviews


def save_completion(
    path: Path,
    *,
    reviewer_id: str,
    families: list[dict[str, Any]],
    seen_family_ids: set[str],
) -> None:
    expected = {str(row["family_id"]) for row in families}
    if seen_family_ids != expected:
        raise ValueError("必须完整浏览50个家族后才能总确认")
    _write_json_atomic(
        path,
        {
            "schema_version": 1,
            "reviewer_id": reviewer_id,
            "completed_at_unix": round(time.time(), 6),
            "family_count": len(families),
            "seen_family_ids": sorted(seen_family_ids),
            "family_bundle_sha256": _family_bundle_sha256(families),
            "attestation": (
                "Reviewer browsed every family and accepted untouched drafts; "
                "saved corrections override individual drafts."
            ),
            "eligible_for_freeze": False,
        },
    )


def load_ocr_excerpt(library_root: Path, item_id: str) -> str:
    path = library_root / "outputs/user_library/ocr/json" / f"{item_id}.json"
    if not path.is_file():
        return "（无 OCR 摘要）"
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return " ".join(str(value) for value in payload.get("rec_texts", []))[:1000]


def _library_root() -> Path:
    default = os.environ.get("OCR_VLM_LIBRARY_ROOT", str(PROJECT_ROOT))
    return Path(
        st.sidebar.text_input("资料库根目录", value=default).strip()
    ).expanduser()


def render() -> None:
    st.set_page_config(page_title="V19端到端查询审核", layout="wide")
    st.title("V19 端到端选择性干预查询审核")
    st.caption(
        "目标页与近邻页已绑定；发现错误后点击保存，写入采用原子替换。"
        "未完成总确认前，任何数据都不能冻结或用于最终结论。"
    )
    families = load_families()
    reviewer_id = st.sidebar.text_input("审核者ID", value="reviewer_01").strip()
    try:
        review_path = review_path_for(reviewer_id)
        completion_path = completion_path_for(reviewer_id)
        progress_path = progress_path_for(reviewer_id)
    except ValueError as error:
        st.error(str(error))
        st.stop()
    reviews = load_reviews(review_path)
    library_root = _library_root()
    if not (library_root / "outputs/user_library/manifest.jsonl").is_file():
        st.error("资料库根目录无效：未找到 outputs/user_library/manifest.jsonl")
        st.stop()

    if "v19si_index" not in st.session_state:
        st.session_state.v19si_index = 0
    index = max(0, min(int(st.session_state.v19si_index), len(families) - 1))
    family = families[index]
    family_id = str(family["family_id"])
    try:
        seen_family_ids = mark_family_seen(
            progress_path,
            reviewer_id=reviewer_id,
            families=families,
            family_id=family_id,
        )
    except ValueError as error:
        st.error(str(error))
        st.stop()
    st.sidebar.metric("已持久化浏览", f"{len(seen_family_ids)}/50")
    st.sidebar.metric("已保存修正", len(reviews))
    st.sidebar.progress(len(seen_family_ids) / len(families))
    st.sidebar.caption(f"修正记录：{review_path}")
    st.sidebar.caption(f"浏览进度：{progress_path}")

    left, middle, right = st.columns([1, 3, 1])
    with left:
        if st.button("← 上一家族", disabled=index == 0, width="stretch"):
            st.session_state.v19si_index = index - 1
            st.rerun()
    with middle:
        selected = st.number_input(
            "浏览位置",
            min_value=1,
            max_value=len(families),
            value=index + 1,
            label_visibility="collapsed",
        )
        if int(selected) - 1 != index:
            st.session_state.v19si_index = int(selected) - 1
            st.rerun()
        st.markdown(
            f"### {index + 1}/50 · `{family_id}` · {family['content_stratum']}"
        )
    with right:
        if st.button(
            "下一家族 →",
            disabled=index == len(families) - 1,
            width="stretch",
        ):
            st.session_state.v19si_index = index + 1
            st.rerun()

    image_columns = st.columns(2)
    for column, role, label in zip(
        image_columns, ("target", "neighbor"), ("目标页", "近邻干扰页"), strict=True
    ):
        info = family[role]
        image_path = library_root / str(info["image_path"])
        with column:
            st.markdown(f"#### {label} · `{info['item_id']}`")
            st.image(str(image_path), width="stretch")
            st.caption(
                f"{info['source_dataset']} · {info['source_file']} · {info['license']}"
            )
            with st.expander(f"{label} OCR 摘要"):
                st.text(load_ocr_excerpt(library_root, str(info["item_id"])))

    saved = reviews.get(family_id, {})
    saved_queries = saved.get("query_texts", family["query_drafts"])
    st.markdown("#### 四条查询草案（发现错误时直接修改）")
    query_texts: dict[str, str] = {}
    for role in ROLES:
        query_texts[role] = st.text_area(
            ROLE_LABELS[role],
            value=str(saved_queries[role]),
            key=f"{family_id}_{role}",
            height=80,
        )
    condition_default = str(
        saved.get("changed_condition_kind", family["changed_condition_kind"])
    )
    changed_condition = st.selectbox(
        "强负例唯一改变的条件",
        CONDITION_KINDS,
        index=CONDITION_KINDS.index(condition_default),
        key=f"{family_id}_condition",
    )
    pair_options = ("accept", "replace_pair")
    pair_default = str(saved.get("pair_decision", "accept"))
    pair_decision = st.radio(
        "来源对是否适合作为目标页/近邻干扰页？",
        pair_options,
        index=pair_options.index(pair_default),
        format_func=lambda value: (
            "适合，保留" if value == "accept" else "不适合，后续替换这一对"
        ),
        horizontal=True,
        key=f"{family_id}_pair",
    )
    notes = st.text_area(
        "备注（可选）",
        value=str(saved.get("notes", "")),
        key=f"{family_id}_notes",
        height=70,
    )
    if st.button("保存本家族修改", type="primary"):
        try:
            save_review(
                review_path,
                family=family,
                query_texts=query_texts,
                changed_condition_kind=changed_condition,
                pair_decision=pair_decision,
                notes=notes,
                reviewer_id=reviewer_id,
            )
        except ValueError as error:
            st.error(str(error))
        else:
            st.success("已原子保存；换页或退出不会丢失。")

    st.divider()
    complete = len(seen_family_ids) == len(families)
    attest = st.checkbox(
        "我已完整浏览50个家族；未保存修改的草案视为通过，已保存修改作为最终审核意见。",
        disabled=not complete,
    )
    if st.button("完成本轮人工审核", disabled=not (complete and attest)):
        try:
            save_completion(
                completion_path,
                reviewer_id=reviewer_id,
                families=families,
                seen_family_ids=seen_family_ids,
            )
        except ValueError as error:
            st.error(str(error))
        else:
            st.success("人工审核已记录，但仍未冻结；下一步由验收脚本检查。")


if __name__ == "__main__":
    render()
