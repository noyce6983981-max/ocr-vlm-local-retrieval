from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from ocr_vlm_retrieval.gating.attribute_coverage import (
    decompose_visual_query,
    load_attribute_policy,
)
from ocr_vlm_retrieval.gating.candidate_verification import (
    build_ocr_evidence,
    evaluate_ocr_requirement,
    load_ocr_lines,
    resolve_requirement_scores,
)
from ocr_vlm_retrieval.gating.contrastive_relations import (
    build_relation_counterfactual,
    counterfactual_prompt,
    relation_evidence_passes,
)


def test_directional_counterfactual_uses_longest_marker() -> None:
    row = build_relation_counterfactual("logo text on the back of the shirt")

    assert row is not None
    assert row.positive_marker == "on the back of"
    assert row.negative_value == "logo text on the front of the shirt"
    assert row.negative_value in counterfactual_prompt(row)


def test_relation_requires_absolute_score_and_margin() -> None:
    assert relation_evidence_passes(
        positive_score=0.61,
        negative_score=0.50,
        absolute_threshold=0.44,
        margin_threshold=0.10,
    )
    assert not relation_evidence_passes(
        positive_score=0.61,
        negative_score=0.58,
        absolute_threshold=0.44,
        margin_threshold=0.10,
    )
    assert not relation_evidence_passes(
        positive_score=0.40,
        negative_score=0.10,
        absolute_threshold=0.44,
        margin_threshold=0.10,
    )


def test_ocr_precedence_exact_then_fuzzy_then_vlm() -> None:
    exact = evaluate_ocr_requirement("实验室安全", ["实验室", "安全"])
    fuzzy = evaluate_ocr_requirement("HELLO", ["HE1LO"], fuzzy_threshold=0.75)
    missing = evaluate_ocr_requirement("HELLO", ["WORLD"])

    assert exact["match_level"] == "exact"
    assert fuzzy["match_level"] == "fuzzy"
    assert missing["source"] == "vlm_fallback"

    resolved, sources = resolve_requirement_scores(
        {"exact": 0.2, "fuzzy": 0.3, "missing": 0.7},
        {
            "exact": exact,
            "fuzzy": fuzzy,
            "missing": missing,
        },
    )
    assert resolved["exact"] == 1.0
    assert resolved["fuzzy"] == pytest.approx(float(fuzzy["score"]))
    assert resolved["missing"] == 0.7
    assert sources["missing"] == "visual_verifier_fallback"


def test_build_ocr_evidence_only_targets_explicit_terms() -> None:
    policy = load_attribute_policy(PROJECT_ROOT / "config/v17_attribute_coverage.json")
    plan = decompose_visual_query('find a sign saying "LAB SAFETY"', policy)

    evidence = build_ocr_evidence(plan, ["LAB", "SAFETY"])

    assert len(evidence) == 1
    assert next(iter(evidence.values()))["match_level"] == "exact"


def test_load_ocr_lines_filters_low_confidence_and_validates(tmp_path: Path) -> None:
    path = tmp_path / "ocr.json"
    path.write_text(
        '{"rec_texts": ["keep", "drop"], "rec_scores": [0.9, 0.2]}',
        encoding="utf-8",
    )
    assert load_ocr_lines(path) == ["keep"]
    assert load_ocr_lines(tmp_path / "missing.json") == []

    path.write_text(
        '{"rec_texts": ["bad"], "rec_scores": []}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="length mismatch"):
        load_ocr_lines(path)
