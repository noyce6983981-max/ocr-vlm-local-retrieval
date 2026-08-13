from __future__ import annotations

from scripts.generate_v19_2_automatic_development_queries import (
    phrase_candidates,
    satisfying_items,
)


def test_phrase_candidates_rejects_numeric_ocr_gibberish() -> None:
    assert phrase_candidates(["8", "888 88", "(11 6020)4115338335 039"]) == []


def test_phrase_candidates_keeps_readable_multitoken_span() -> None:
    candidates = phrase_candidates(["Federal Trade Commission research agreement"])
    assert "Federal Trade Commission research agreement" in candidates


def test_satisfying_items_requires_same_page_conjunction() -> None:
    corpus = {
        "left": "alpha project",
        "right": "budget 2026",
        "complete": "alpha project budget 2026",
    }
    assert satisfying_items(["Alpha Project", "Budget 2026"], corpus) == [
        "complete"
    ]
