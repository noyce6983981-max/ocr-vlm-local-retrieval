from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRIVATE_DIR = PROJECT_ROOT / "records/private/v19/selective_intervention"
DEFAULT_FAMILIES = PRIVATE_DIR / "source_families_draft.jsonl"
DEFAULT_CATALOG = PRIVATE_DIR / "query_draft_catalog.json"
DEFAULT_QUERIES = PRIVATE_DIR / "queries_draft.jsonl"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_catalog(path: Path) -> dict[str, tuple[str, str, str, str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("query draft catalog must be a JSON object")
    result: dict[str, tuple[str, str, str, str, str]] = {}
    for family_id, raw_spec in payload.items():
        if (
            not isinstance(family_id, str)
            or not isinstance(raw_spec, list)
            or len(raw_spec) != 5
            or not all(isinstance(value, str) and value.strip() for value in raw_spec)
        ):
            raise ValueError(f"invalid query draft catalog row: {family_id}")
        result[family_id] = (
            raw_spec[0],
            raw_spec[1],
            raw_spec[2],
            raw_spec[3],
            raw_spec[4],
        )
    return result


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
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


def build_query_rows(
    families: list[dict[str, Any]],
    catalog: Mapping[str, tuple[str, str, str, str, str]],
) -> list[dict[str, Any]]:
    family_ids = {str(row["family_id"]) for row in families}
    if family_ids != set(catalog):
        missing = sorted(family_ids - set(catalog))
        extra = sorted(set(catalog) - family_ids)
        raise ValueError(f"draft family mismatch: missing={missing}, extra={extra}")
    query_rows: list[dict[str, Any]] = []
    for family in families:
        family_id = str(family["family_id"])
        positive, paraphrase, hard_negative, neighbor_negative, condition = catalog[
            family_id
        ]
        target_id = str(family["target"]["item_id"])
        neighbor_id = str(family["neighbor"]["item_id"])
        role_specs: tuple[tuple[str, str, bool, list[str]], ...] = (
            ("answerable_positive", positive, True, [target_id]),
            ("paraphrase_positive", paraphrase, True, [target_id]),
            ("single_condition_hard_negative", hard_negative, False, []),
            ("unanswerable_neighbor", neighbor_negative, False, []),
        )
        family["query_drafts"] = {
            role: query_text for role, query_text, _answerable, _gold in role_specs
        }
        family["changed_condition_kind"] = condition
        family["selection_status"] = "source_bound_query_drafted"
        for index, (role, query_text, answerable, gold_ids) in enumerate(
            role_specs, start=1
        ):
            query_rows.append(
                {
                    "query_id": f"{family_id}_q{index}",
                    "family_id": family_id,
                    "split": family["split"],
                    "content_stratum": family["content_stratum"],
                    "query_role": role,
                    "query_text": query_text,
                    "gold_answerable": answerable,
                    "gold_relevant_item_ids": gold_ids,
                    "target_item_id": target_id,
                    "neighbor_item_id": neighbor_id,
                    "changed_condition_kind": condition,
                    "status": "codex_draft_pending_human_review_not_frozen",
                }
            )
    texts = [str(row["query_text"]) for row in query_rows]
    if len(families) != 50 or len(query_rows) != 200:
        raise ValueError("selective-intervention draft must be 50 families / 200 queries")
    if len(texts) != len(set(texts)):
        raise ValueError("draft query text must be globally unique")
    if sum(bool(row["gold_answerable"]) for row in query_rows) != 100:
        raise ValueError("draft must contain 100 answerable and 100 unanswerable queries")
    return query_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile private V19 source bindings and query drafts."
    )
    parser.add_argument("--families", type=Path, default=DEFAULT_FAMILIES)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_QUERIES)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.catalog.is_file():
        raise SystemExit(
            "Private query draft catalog is missing; author drafts before compiling"
        )
    families = _load_jsonl(args.families)
    catalog = load_catalog(args.catalog)
    queries = build_query_rows(families, catalog)
    _write_jsonl_atomic(args.families, families)
    _write_jsonl_atomic(args.output, queries)
    print("Generated 50 source-bound draft families and 200 draft queries")
    print("Status remains pending human review; no freeze receipt was created")
    print(f"Wrote {args.families}")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
