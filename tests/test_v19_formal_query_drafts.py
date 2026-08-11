from __future__ import annotations

from collections import Counter

from scripts.generate_v19_formal_query_drafts import (
    build_rows,
    validate_rows,
)


def test_formal_query_drafts_are_family_isolated_and_balanced() -> None:
    families, queries = build_rows()
    validate_rows(families, queries)
    assert len(families) == 60
    assert len(queries) == 240
    assert Counter(row["split"] for row in queries) == {
        "calibration": 120,
        "holdout": 120,
    }
    assert all(len(row["queries"]) == 4 for row in families)
    assert all(
        len({row["split"] for row in queries if row["family_id"] == family_id})
        == 1
        for family_id in {row["family_id"] for row in queries}
    )
