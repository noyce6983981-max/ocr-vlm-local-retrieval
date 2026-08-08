from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_module():
    path = (
        Path(__file__).resolve().parents[1] / "scripts/prepare_v18_query_authoring.py"
    )
    spec = importlib.util.spec_from_file_location("prepare_v18_query_authoring", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_protocol_amendment_preserves_source_bound_authoring_ids() -> None:
    module = _load_module()
    previous = [
        {"authoring_id": "source_001", "source_item_id": "item_a"},
        {"authoring_id": "source_002", "source_item_id": "item_b"},
    ]
    amended = [
        {"authoring_id": "temporary_001", "source_item_id": "item_b"},
        {"authoring_id": "temporary_002", "source_item_id": "item_a"},
    ]
    assert module.preserve_authoring_ids(amended, previous) == [
        {"authoring_id": "source_001", "source_item_id": "item_a"},
        {"authoring_id": "source_002", "source_item_id": "item_b"},
    ]


def test_authoring_ids_are_not_preserved_if_source_set_changes() -> None:
    module = _load_module()
    with pytest.raises(ValueError, match="source set changed"):
        module.preserve_authoring_ids(
            [{"authoring_id": "new", "source_item_id": "item_b"}],
            [{"authoring_id": "old", "source_item_id": "item_a"}],
        )
