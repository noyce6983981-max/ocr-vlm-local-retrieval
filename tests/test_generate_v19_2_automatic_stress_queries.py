from __future__ import annotations

import pytest

from scripts.generate_v19_2_automatic_stress_queries import TEMPLATES, stress_query


def test_every_stress_template_preserves_both_values() -> None:
    for template in TEMPLATES:
        query = stress_query(template, ["Alpha Project", "Budget 2026"])
        assert "Alpha Project" in query
        assert "Budget 2026" in query


def test_stress_query_requires_exactly_two_phrases() -> None:
    with pytest.raises(ValueError, match="exactly two"):
        stress_query(TEMPLATES[0], ["only one"])
