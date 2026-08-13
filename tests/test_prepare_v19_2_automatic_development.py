from __future__ import annotations

from scripts.prepare_v19_2_automatic_development import (
    _source_ids,
    load_evidence_readable_item_ids,
)


def test_source_ids_reads_nested_and_flat_roles(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "rows.jsonl"
    path.write_text(
        '{"target": {"item_id": "a"}, "neighbor": {"item_id": "b"}}\n'
        '{"target_item_id": "c", "neighbor_item_id": "d"}\n',
        encoding="utf-8",
    )
    assert _source_ids(path, ("target", "neighbor")) == {"a", "b"}
    assert _source_ids(path, ("target_item_id", "neighbor_item_id")) == {"c", "d"}


def test_load_evidence_readable_item_ids_rejects_ocr_gibberish(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    (tmp_path / "good.json").write_text(
        '{"rec_texts":["Federal Trade Commission",'
        '"Research consulting agreement"],"rec_scores":[0.99,0.98]}',
        encoding="utf-8",
    )
    (tmp_path / "bad.json").write_text(
        '{"rec_texts":["8","n~","a"],"rec_scores":[0.9,0.9,0.9]}',
        encoding="utf-8",
    )
    manifest = [{"item_id": "good"}, {"item_id": "bad"}]
    assert load_evidence_readable_item_ids(manifest, tmp_path) == {"good"}
