from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_module():
    path = PROJECT_ROOT / "scripts/v18_live_search.py"
    spec = importlib.util.spec_from_file_location("v18_live_search", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_entity_route_is_retrieved_as_mixed_without_labels() -> None:
    module = _load_module()
    module._resolve_search_intent = lambda method, query: (
        "entity_exact",
        False,
        query,
        "person name",
    )
    assert module.resolve_v18_search_intent("quality_hybrid", "query") == (
        "mixed",
        False,
        "query",
        None,
    )
