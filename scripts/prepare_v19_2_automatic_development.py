"""Select fresh V19.2 development families without a human review queue."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_v19_selective_intervention_review import (  # noqa: E402
    _atomic_jsonl,
    _load_jsonl,
    build_source_families,
    load_excluded_item_ids,
    load_ocr_character_counts,
)
from scripts.generate_v19_2_automatic_development_queries import (  # noqa: E402
    phrase_candidates,
)
from ocr_vlm_retrieval.gating.candidate_verification import (  # noqa: E402
    load_ocr_lines,
)

CONFIG = ROOT / "config/studies/v19_2_automatic_optimization.json"
V18_ROOT = ROOT / "data/evaluation/v18"
V19_QUERIES = ROOT / "records/private/v19/selective_intervention/reviewed_queries.jsonl"
V19_1_FAMILIES = (
    ROOT / "records/private/v19_1/condition_completeness/source_families_draft.jsonl"
)
PRIVATE_DIR = ROOT / "records/private/v19_2/automatic_optimization"
DEFAULT_OUTPUT = PRIVATE_DIR / "development_source_families.jsonl"
DEFAULT_RECEIPT = PRIVATE_DIR / "development_source_selection_receipt.json"
ALL_STRATA = (
    "ocr_literal_lookup",
    "pure_visual",
    "layout_table",
    "entity_identifier",
    "visual_compositional",
    "text_visual_compositional",
    "topic_discovery",
    "relationship_scene",
)


def _source_ids(path: Path, roles: tuple[str, ...]) -> set[str]:
    result: set[str] = set()
    for row in _load_jsonl(path):
        for role in roles:
            value: Any = row.get(role)
            if isinstance(value, dict):
                item_id = str(value.get("item_id", "")).strip()
            else:
                item_id = str(row.get(role, "")).strip()
            if item_id:
                result.add(item_id)
    return result


def load_evidence_readable_item_ids(
    manifest: list[dict[str, Any]], ocr_root: Path
) -> set[str]:
    """Return pages with enough readable OCR spans for a two-condition contract."""

    result: set[str] = set()
    for row in manifest:
        item_id = str(row.get("item_id", "")).strip()
        path = ocr_root / f"{item_id}.json"
        if not item_id or not path.is_file():
            continue
        lines = load_ocr_lines(path, minimum_confidence=0.35)
        if len(phrase_candidates(lines)) >= 2:
            result.add(item_id)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--v18-evaluation-root", type=Path, default=V18_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--seed", default="v19-2-automatic-development-v1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    manifest = _load_jsonl(ROOT / "outputs/user_library/manifest.jsonl")
    v18 = load_excluded_item_ids(args.v18_evaluation_root.resolve())
    if not v18:
        raise ValueError("V18 exclusion list is empty")
    v19 = _source_ids(V19_QUERIES, ("target_item_id", "neighbor_item_id"))
    v19_1 = _source_ids(V19_1_FAMILIES, ("target", "neighbor"))
    excluded = v18 | v19 | v19_1
    raw_ocr_counts = load_ocr_character_counts(
        ROOT / "outputs/user_library/ocr/summary.csv"
    )
    evidence_readable = load_evidence_readable_item_ids(
        manifest, ROOT / "outputs/user_library/ocr/json"
    )
    evidence_qualified_counts = {
        item_id: count if item_id in evidence_readable else 0
        for item_id, count in raw_ocr_counts.items()
    }
    requested = config["development_family_counts"]
    counts = {name: int(requested.get(name, 0)) for name in ALL_STRATA}
    families = build_source_families(
        manifest,
        stratum_counts=counts,
        excluded_item_ids=excluded,
        ocr_character_counts=evidence_qualified_counts,
        seed=args.seed,
    )
    per_stratum: Counter[str] = Counter()
    for family in families:
        stratum = str(family["content_stratum"])
        per_stratum[stratum] += 1
        family["family_id"] = f"v19_2_auto_{stratum}_{per_stratum[stratum]:02d}"
        family["split"] = "automatic_development_only"
        family["selection_status"] = "fresh_machine_development_source"
        family["review_status"] = "machine_evidence_contract_pending"
    selected_ids = {
        str(family[role]["item_id"])
        for family in families
        for role in ("target", "neighbor")
    }
    if selected_ids & excluded:
        raise AssertionError("V19.2 development overlaps a predecessor source")
    _atomic_jsonl(args.output, families)
    receipt = {
        "schema_version": 1,
        "study_id": config["study_id"],
        "status": "fresh_automatic_development_sources_selected",
        "seed": args.seed,
        "family_count": len(families),
        "source_item_count": len(selected_ids),
        "stratum_counts": dict(sorted(per_stratum.items())),
        "excluded_v18_item_count": len(v18),
        "excluded_v19_item_count": len(v19),
        "excluded_v19_1_item_count": len(v19_1),
        "selected_predecessor_overlap_count": 0,
        "evidence_readable_item_count": len(evidence_readable),
        "evidence_unreadable_item_count": len(raw_ocr_counts) - len(evidence_readable),
        "minimum_readable_evidence_phrases_per_source": 2,
        "human_review_required": False,
        "machine_evidence_contract_required": True,
        "future_holdout_opened": False,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
