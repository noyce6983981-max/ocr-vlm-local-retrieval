from __future__ import annotations

import json
import ast
from pathlib import Path

from scripts.score_library_text_batch import read_queries as read_text_queries


def write_reviewed_queries(path: Path) -> None:
    rows = [
        {
            "query_id": "dev",
            "split": "development",
            "query_text": "找右上角带校徽的封面",
            "status": "reviewed_not_frozen",
        },
        {
            "query_id": "held",
            "split": "holdout",
            "query_text": "留出查询不得读取",
            "status": "reviewed_not_frozen",
        },
    ]
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_batch_scorers_accept_reviewed_query_schema_and_isolate_development(
    tmp_path: Path,
) -> None:
    path = tmp_path / "reviewed.jsonl"
    write_reviewed_queries(path)
    rows = read_text_queries(path, {"development"}, "reviewed_not_frozen")
    assert len(rows) == 1
    assert rows[0]["query_id"] == "dev"
    assert rows[0]["query"] == "找右上角带校徽的封面"
    assert rows[0]["review_status"] == "reviewed_not_frozen"


def test_visual_batch_reader_contains_the_same_reviewed_schema_bridge() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts/score_library_visual_batch.py"
    ).read_text(encoding="utf-8")
    assert 'row.get("query") or row.get("query_text")' in source
    assert 'row.get("review_status") or row.get("status")' in source


def test_text_batch_reader_does_not_import_gpu_runtime_at_module_load() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts/score_library_text_batch.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    top_level_imports = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        str(node.module)
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
    }
    assert "torch" not in top_level_imports
    assert "FlagEmbedding" not in top_level_imports
