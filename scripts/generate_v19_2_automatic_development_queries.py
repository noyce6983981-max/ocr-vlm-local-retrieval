"""Generate and validate V19.2 development queries without human review."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ocr_vlm_retrieval.gating.candidate_verification import (  # noqa: E402
    load_ocr_lines,
    normalize_ocr_text,
)
from ocr_vlm_retrieval.gating.ocr_literals_v19_2 import (  # noqa: E402
    extract_v19_2_literal_groups,
)
from ocr_vlm_retrieval.routing.rule_router import RuleRouter  # noqa: E402
from ocr_vlm_retrieval.runtime.cache import write_json_atomic  # noqa: E402

PRIVATE_DIR = ROOT / "records/private/v19_2/automatic_optimization"
DEFAULT_FAMILIES = PRIVATE_DIR / "development_source_families.jsonl"
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/evaluation/v19_2/automatic_optimization"
    / "development_assignments_machine.json"
)
OCR_ROOT = ROOT / "outputs/user_library/ocr/json"
MANIFEST = ROOT / "outputs/user_library/manifest.jsonl"
TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9'&./-]*|[\u3400-\u4dbf\u4e00-\u9fff]+")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def _clean_phrase(value: str) -> str:
    return " ".join(value.replace("“", " ").replace("”", " ").split()).strip(
        " ,.;:!?()[]{}'\""
    )


def phrase_candidates(lines: Iterable[str]) -> list[str]:
    """Return deterministic, readable OCR spans suitable for evidence contracts."""

    values: list[str] = []
    for raw in lines:
        line = _clean_phrase(str(raw))
        tokens = TOKEN.findall(line)
        latin = [token for token in tokens if re.search(r"[A-Za-z]", token)]
        if 8 <= len(line) <= 56 and len(tokens) >= 2:
            values.append(line)
        if len(latin) >= 4:
            for size in (5, 4, 3):
                if len(latin) < size:
                    continue
                values.extend(
                    " ".join(latin[start : start + size])
                    for start in range(0, len(latin) - size + 1)
                )
        for chinese in re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]{6,}", line):
            values.append(chinese[:20])
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = _clean_phrase(value)
        normalized = normalize_ocr_text(cleaned)
        alphanumeric = [character for character in cleaned if character.isalnum()]
        alphabetic = sum(character.isalpha() for character in alphanumeric)
        punctuation = sum(
            not character.isalnum() and not character.isspace()
            for character in cleaned
        )
        if (
            6 <= len(normalized.replace(" ", "")) <= 64
            and normalized not in seen
            and not normalized.isdigit()
            and alphanumeric
            and alphabetic / len(alphanumeric) >= 0.45
            and punctuation / len(cleaned) <= 0.2
        ):
            seen.add(normalized)
            result.append(cleaned)
    return result


def _corpus_index() -> dict[str, str]:
    result: dict[str, str] = {}
    for row in _read_jsonl(MANIFEST):
        item_id = str(row.get("item_id", ""))
        path = OCR_ROOT / f"{item_id}.json"
        if item_id and path.is_file():
            result[item_id] = normalize_ocr_text(
                " ".join(load_ocr_lines(path, minimum_confidence=0.35))
            )
    return result


def _document_frequency(phrase: str, corpus: Mapping[str, str]) -> int:
    needle = normalize_ocr_text(phrase)
    return sum(needle in text for text in corpus.values())


def select_phrases(
    item_id: str,
    other_item_id: str,
    *,
    count: int,
    corpus: Mapping[str, str],
) -> list[str]:
    lines = load_ocr_lines(OCR_ROOT / f"{item_id}.json", minimum_confidence=0.35)
    other = corpus[other_item_id]
    ranked = sorted(
        phrase_candidates(lines),
        key=lambda phrase: (
            _document_frequency(phrase, corpus),
            -len(normalize_ocr_text(phrase)),
            normalize_ocr_text(phrase),
        ),
    )
    selected: list[str] = []
    for phrase in ranked:
        normalized = normalize_ocr_text(phrase)
        if normalized not in corpus[item_id]:
            continue
        if normalized in other:
            continue
        if _document_frequency(phrase, corpus) > 2:
            continue
        if any(
            normalized in normalize_ocr_text(old)
            or normalize_ocr_text(old) in normalized
            for old in selected
        ):
            continue
        selected.append(phrase)
        if len(selected) == count:
            return selected
    raise ValueError(f"not enough distinctive OCR phrases for {item_id}")


def satisfying_items(phrases: Iterable[str], corpus: Mapping[str, str]) -> list[str]:
    needles = [normalize_ocr_text(value) for value in phrases]
    return sorted(
        item_id
        for item_id, text in corpus.items()
        if all(needle in text for needle in needles)
    )


def _query(first: str, second: str, *, paraphrase: bool) -> str:
    if paraphrase:
        return f"查找同时包含“{first}”和“{second}”的页面。"
    return f"哪份资料必须包含“{first}”，并同时包含“{second}”？"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", type=Path, default=DEFAULT_FAMILIES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    families = _read_jsonl(args.families)
    if len(families) != 12:
        raise ValueError("expected 12 automatic development families")
    corpus = _corpus_index()
    router = RuleRouter.legacy()
    assignments: list[dict[str, Any]] = []
    contract_rows: list[dict[str, Any]] = []
    for family in families:
        family_id = str(family["family_id"])
        target = str(family["target"]["item_id"])
        neighbor = str(family["neighbor"]["item_id"])
        target_phrases = select_phrases(
            target, neighbor, count=2, corpus=corpus
        )
        neighbor_phrases = select_phrases(
            neighbor, target, count=2, corpus=corpus
        )
        specs: tuple[
            tuple[str, Sequence[str], bool, list[str]],
            ...,
        ] = (
            ("answerable_positive", target_phrases[:2], False, [target]),
            (
                "paraphrase_positive",
                (target_phrases[1], target_phrases[0]),
                True,
                [target],
            ),
            (
                "single_condition_hard_negative",
                (target_phrases[0], neighbor_phrases[0]),
                False,
                [],
            ),
            (
                "unanswerable_neighbor",
                (target_phrases[1], neighbor_phrases[1]),
                True,
                [],
            ),
        )
        for index, (role, phrases, paraphrase, relevant) in enumerate(specs, start=1):
            matches = satisfying_items(phrases, corpus)
            if matches != relevant:
                raise ValueError(
                    f"machine evidence contract failed for {family_id}/{role}: {matches}"
                )
            query = _query(phrases[0], phrases[1], paraphrase=paraphrase)
            groups = [
                group
                for group in extract_v19_2_literal_groups(query)
                if group.source == "explicit_required"
            ]
            if [group.label for group in groups] != list(phrases):
                raise ValueError(f"query parser contract failed: {family_id}/{role}")
            answerable = bool(relevant)
            assignment = {
                "query_id": f"{family_id}_q{index}",
                "family_id": family_id,
                "query": query,
                "query_role": role,
                "content_stratum": family["content_stratum"],
                "gold_answerable": answerable,
                "gold_relevant_item_ids": relevant,
                "source_item_id": target,
                "neighbor_item_id": neighbor,
                "changed_condition_kind": "explicit_required_phrase",
                "legacy_route": router.route(query).route,
                "review_status": "machine_evidence_contract_passed",
            }
            assignments.append(assignment)
            contract_rows.append(
                {
                    "query_id": assignment["query_id"],
                    "phrases": list(phrases),
                    "full_corpus_satisfying_item_ids": matches,
                }
            )
    if len(assignments) != 48 or len({row["query"] for row in assignments}) != 48:
        raise AssertionError("expected 48 unique automatic development queries")
    payload = {
        "schema_version": 1,
        "study_id": "v19-2-automatic-condition-aggregation",
        "status": "machine_authored_evidence_contract_development",
        "split": "v19_2_automatic_development_only",
        "eligible_for_final_claim": False,
        "human_review_used": False,
        "machine_evidence_contract": "exact normalized OCR conjunction over all indexed pages",
        "source_family_sha256": hashlib.sha256(args.families.read_bytes()).hexdigest(),
        "family_count": len(families),
        "query_count": len(assignments),
        "role_counts": dict(Counter(row["query_role"] for row in assignments)),
        "future_holdout_opened": False,
        "assignments": assignments,
        "evidence_contract_rows": contract_rows,
    }
    write_json_atomic(args.output, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "family_count": payload["family_count"],
                "query_count": payload["query_count"],
                "role_counts": payload["role_counts"],
                "human_review_used": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
