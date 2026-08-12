from __future__ import annotations

from scripts.generate_v19_selective_intervention_query_drafts import build_query_rows


def _catalog() -> dict[str, tuple[str, str, str, str, str]]:
    return {
        f"family_{index:02d}": (
            f"第{index}组可回答正例查询文本",
            f"第{index}组自然改写正例查询文本",
            f"第{index}组单条件强负例查询文本",
            f"第{index}组近邻无答案查询文本",
            "color",
        )
        for index in range(50)
    }


def test_query_rows_preserve_source_binding_and_answerability() -> None:
    families = []
    catalog = _catalog()
    for family_id in catalog:
        families.append(
            {
                "family_id": family_id,
                "split": "development",
                "content_stratum": "pure_visual",
                "target": {"item_id": f"target_{family_id}"},
                "neighbor": {"item_id": f"neighbor_{family_id}"},
            }
        )

    rows = build_query_rows(families, catalog)

    assert len(rows) == 200
    assert sum(bool(row["gold_answerable"]) for row in rows) == 100
    assert all(
        row["gold_relevant_item_ids"] == [row["target_item_id"]]
        for row in rows
        if row["gold_answerable"]
    )
    assert all(
        not row["gold_relevant_item_ids"]
        for row in rows
        if not row["gold_answerable"]
    )
