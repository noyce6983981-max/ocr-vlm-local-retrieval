"""Store auditable assistant category proposals separately from human truth."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.taxonomy import CATEGORY_LABELS


PROPOSAL_FIELDS = [
    "item_id",
    "proposed_category",
    "assistant_notes",
    "proposal_round",
    "model_current_category",
    "model_predicted_category",
    "model_confidence",
    "status",
    "proposed_at",
    "reviewed_at",
]
ACTIVE_PROPOSAL_STATUSES = {"pending"}
RESOLVED_PROPOSAL_STATUSES = {"confirmed", "revised", "rejected"}
VALID_PROPOSAL_STATUSES = (
    ACTIVE_PROPOSAL_STATUSES | RESOLVED_PROPOSAL_STATUSES
)


def read_assistant_proposals(
    path: Path | None,
) -> dict[str, dict[str, str]]:
    if path is None or not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    proposals: dict[str, dict[str, str]] = {}
    for row in rows:
        item_id = str(row.get("item_id", "")).strip()
        if not item_id:
            continue
        if item_id in proposals:
            raise ValueError(f"Duplicate assistant proposal: {item_id}")
        category = str(row.get("proposed_category", "")).strip()
        if category not in CATEGORY_LABELS:
            raise ValueError(
                f"Unsupported assistant proposal category for "
                f"{item_id}: {category}"
            )
        status = str(row.get("status", "pending")).strip() or "pending"
        if status not in VALID_PROPOSAL_STATUSES:
            raise ValueError(
                f"Unsupported assistant proposal status for "
                f"{item_id}: {status}"
            )
        normalized = {
            field: str(row.get(field, "")).strip()
            for field in PROPOSAL_FIELDS
        }
        normalized["status"] = status
        proposals[item_id] = normalized
    return proposals


def is_active_assistant_proposal(
    proposal: dict[str, str] | None,
) -> bool:
    return bool(
        proposal
        and proposal.get("status", "pending")
        in ACTIVE_PROPOSAL_STATUSES
        and proposal.get("proposed_category") in CATEGORY_LABELS
    )


def write_assistant_proposals(
    path: Path,
    proposals: dict[str, dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=PROPOSAL_FIELDS)
        writer.writeheader()
        for proposal in proposals.values():
            writer.writerow(
                {
                    field: str(proposal.get(field, ""))
                    for field in PROPOSAL_FIELDS
                }
            )
    temporary_path.replace(path)


def resolve_assistant_proposal(
    path: Path,
    *,
    item_id: str,
    human_category: str | None,
    now: datetime | None = None,
) -> None:
    proposals = read_assistant_proposals(path)
    proposal = proposals.get(item_id)
    if proposal is None:
        return
    proposed_category = proposal["proposed_category"]
    if human_category is None:
        proposal["status"] = "rejected"
    elif human_category == proposed_category:
        proposal["status"] = "confirmed"
    else:
        proposal["status"] = "revised"
    timestamp = now or datetime.now(timezone.utc)
    proposal["reviewed_at"] = timestamp.isoformat(timespec="seconds")
    write_assistant_proposals(path, proposals)
