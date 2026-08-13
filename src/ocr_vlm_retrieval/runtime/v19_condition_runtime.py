"""Pure payload helpers for the optional V19 condition-evidence runtime."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def ordered_candidate_ids(
    candidate_item_ids: Sequence[str], selected_item_id: str | None
) -> list[str]:
    """Promote a verified candidate while preserving retrieval order otherwise."""

    values = [str(value) for value in candidate_item_ids]
    if selected_item_id is None or selected_item_id not in values:
        return values
    return [selected_item_id, *(value for value in values if value != selected_item_id)]


def build_condition_ranking(
    manifest_by_id: Mapping[str, Mapping[str, Any]],
    candidate_item_ids: Sequence[str],
    scores_by_id: Mapping[str, float],
    candidate_evidence: Sequence[Mapping[str, Any]],
    *,
    selected_item_id: str | None,
) -> list[dict[str, Any]]:
    """Build UI-compatible rows with complete-condition audit fields."""

    evidence_by_id = {str(row["item_id"]): row for row in candidate_evidence}
    rows: list[dict[str, Any]] = []
    for display_rank, item_id in enumerate(
        ordered_candidate_ids(candidate_item_ids, selected_item_id), start=1
    ):
        if item_id not in manifest_by_id:
            raise ValueError(f"candidate missing from manifest: {item_id}")
        evidence = evidence_by_id.get(item_id, {})
        rows.append(
            {
                **dict(manifest_by_id[item_id]),
                "score": round(float(scores_by_id[item_id]), 6),
                "raw_visual_score": round(float(scores_by_id[item_id]), 6),
                "text_weight": 0.0,
                "bm25_query_weight": 0.0,
                "retrieval_route": "v19_condition_evidence",
                "v19_display_rank": display_rank,
                "v19_original_retrieval_rank": int(
                    evidence.get(
                        "retrieval_rank", candidate_item_ids.index(item_id) + 1
                    )
                ),
                "v19_all_constraints_matched": bool(
                    evidence.get("all_constraints_matched", False)
                ),
                "v19_matched_constraint_count": int(
                    evidence.get("matched_constraint_count", 0)
                ),
                "v19_constraint_count": int(evidence.get("constraint_count", 0)),
                "v19_condition_evidence": list(evidence.get("evidence", [])),
            }
        )
    return rows


def alias_fallback_payload(
    payload: Mapping[str, Any],
    *,
    reason: str,
    fallback_method: str = "quality_hybrid",
) -> dict[str, Any]:
    """Expose a legacy result under the V19 UI key with explicit provenance."""

    rankings = {
        key: list(value) for key, value in dict(payload.get("rankings", {})).items()
    }
    acceptance = {
        key: dict(value)
        for key, value in dict(payload.get("acceptance", {})).items()
    }
    rankings["v19_condition"] = list(rankings.get(fallback_method, []))
    fallback_decision = dict(
        acceptance.get(
            fallback_method,
            {
                "accepted": bool(rankings["v19_condition"]),
                "query_mode": payload.get("query_mode", "mixed"),
                "reason": "沿用稳定检索链路。",
            },
        )
    )
    fallback_decision["reason"] = (
        f"V19必要条件核验未介入：{reason}；" + str(fallback_decision["reason"])
    )
    fallback_decision["v19_condition_intervened"] = False
    acceptance["v19_condition"] = fallback_decision
    return {
        **dict(payload),
        "requested_method": "v19_condition",
        "rankings": rankings,
        "acceptance": acceptance,
        "v19_condition_runtime": {
            "status": "stable_fallback",
            "intervened": False,
            "reason": reason,
            "fallback_method": fallback_method,
        },
    }
