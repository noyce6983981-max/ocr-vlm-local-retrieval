from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


def _load_module():
    path = PROJECT_ROOT / "scripts/audit_v18_condition_decomposition.py"
    spec = importlib.util.spec_from_file_location(
        "audit_v18_condition_decomposition", path
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pair_change_is_detected_without_relevance_labels() -> None:
    module = _load_module()
    policy = json.loads(
        (PROJECT_ROOT / "config/v17_attribute_coverage.json").read_text(
            encoding="utf-8"
        )
    )
    queries = [
        {
            "query_id": "positive",
            "query": "绿色树木位于岩石前方。",
            "group_id": "g1",
            "query_role": "positive",
        },
        {
            "query_id": "negative",
            "query": "黄色树木位于岩石前方。",
            "group_id": "g1",
            "query_role": "single_condition_hard_negative",
        },
    ]
    audit = module.audit_plans(queries, policy)
    assert audit["compositional_query_count"] == 2
    assert audit["changed_condition_detected_group_count"] == 1
    assert audit["failed_changed_condition_group_ids"] == []
