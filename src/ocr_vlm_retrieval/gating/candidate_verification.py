"""Model-independent helpers for V17 candidate verification."""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterable, Mapping
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from ocr_vlm_retrieval.gating.attribute_coverage import AttributePlan

FULL_QUERY_INSTRUCTION = (
    "Verify exact full-query satisfaction in this one candidate image. "
    "Every necessary object, visual attribute, spatial or directional "
    "relation, and requested text condition must hold in the same image. "
    "A partial or near-neighbor match is negative."
)

ATTRIBUTE_INSTRUCTION = (
    "Verify only the stated necessary visual condition in this candidate. "
    "For directional relations and attribute bindings, require the exact "
    "direction and the same referenced object; partial evidence is negative."
)


def verification_prompt_payload() -> dict[str, str]:
    """Return prompt text as data so the later method lock can hash it."""

    return {
        "full_query_instruction": FULL_QUERY_INSTRUCTION,
        "attribute_instruction": ATTRIBUTE_INSTRUCTION,
    }


def normalize_ocr_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = re.sub(r"[^\w\u3400-\u4dbf\u4e00-\u9fff]+", " ", normalized)
    return " ".join(normalized.split())


def _compact(value: str) -> str:
    return normalize_ocr_text(value).replace(" ", "")


def load_ocr_lines(path: Path, *, minimum_confidence: float = 0.35) -> list[str]:
    """Read PaddleOCR text lines while applying the recorded confidence."""

    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    texts = payload.get("rec_texts", [])
    scores = payload.get("rec_scores", [])
    if not isinstance(texts, list) or not isinstance(scores, list):
        raise ValueError(f"Invalid OCR payload: {path}")
    if len(texts) != len(scores):
        raise ValueError(f"OCR text/score length mismatch: {path}")
    return [
        str(text).strip()
        for text, score in zip(texts, scores, strict=True)
        if str(text).strip() and float(score) >= minimum_confidence
    ]


def _best_fuzzy_similarity(term: str, lines: Iterable[str]) -> float:
    compact_term = _compact(term)
    compact_lines = [_compact(line) for line in lines]
    compact_lines = [line for line in compact_lines if line]
    if not compact_term or not compact_lines:
        return 0.0
    candidates = list(compact_lines)
    joined = "".join(compact_lines)
    width = len(compact_term)
    if width and len(joined) >= width:
        window_count = len(joined) - width + 1
        candidates.extend(
            joined[index : index + width] for index in range(window_count)
        )
    return max(
        SequenceMatcher(None, compact_term, candidate).ratio()
        for candidate in candidates
    )


def evaluate_ocr_requirement(
    term: str,
    lines: Iterable[str],
    *,
    fuzzy_threshold: float = 0.88,
) -> dict[str, Any]:
    """Prefer exact OCR, then fuzzy OCR, and explicitly mark VLM fallback."""

    normalized_term = normalize_ocr_text(term)
    if not normalized_term:
        raise ValueError("OCR requirement term must not be empty")
    collected = [str(line) for line in lines]
    normalized_corpus = normalize_ocr_text(" ".join(collected))
    compact_term = _compact(term)
    compact_corpus = _compact(" ".join(collected))
    exact = bool(
        normalized_term in normalized_corpus
        or (compact_term and compact_term in compact_corpus)
    )
    similarity = 1.0 if exact else _best_fuzzy_similarity(term, collected)
    if exact:
        level = "exact"
        score = 1.0
        source = "deterministic_ocr_exact"
    elif similarity >= fuzzy_threshold:
        level = "fuzzy"
        score = similarity
        source = "deterministic_ocr_fuzzy"
    else:
        level = "none"
        score = 0.0
        source = "vlm_fallback"
    return {
        "term": term,
        "normalized_term": normalized_term,
        "match_level": level,
        "score": round(score, 8),
        "similarity": round(similarity, 8),
        "source": source,
        "ocr_line_count": len(collected),
    }


def build_ocr_evidence(
    plan: AttributePlan,
    lines: Iterable[str],
    *,
    fuzzy_threshold: float = 0.88,
) -> dict[str, dict[str, Any]]:
    collected = list(lines)
    return {
        requirement.requirement_id: evaluate_ocr_requirement(
            requirement.value,
            collected,
            fuzzy_threshold=fuzzy_threshold,
        )
        for requirement in plan.requirements
        if requirement.kind == "ocr"
    }


def resolve_requirement_scores(
    model_scores: Mapping[str, float],
    ocr_evidence: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, float], dict[str, str]]:
    """Overlay deterministic OCR when available, retaining VLM as fallback."""

    resolved = {key: float(value) for key, value in model_scores.items()}
    sources = {key: "visual_verifier" for key in resolved}
    for requirement_id, evidence in ocr_evidence.items():
        if requirement_id not in resolved:
            raise ValueError(f"OCR evidence references unknown {requirement_id!r}")
        level = str(evidence.get("match_level", "none"))
        if level in {"exact", "fuzzy"}:
            resolved[requirement_id] = float(evidence["score"])
            sources[requirement_id] = str(evidence["source"])
        else:
            sources[requirement_id] = "visual_verifier_fallback"
    return resolved, sources
