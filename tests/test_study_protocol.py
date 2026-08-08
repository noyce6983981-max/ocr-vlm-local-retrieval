from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from ocr_vlm_retrieval.studies.protocol import (
    ProtocolError,
    StudyProtocol,
    load_study_protocol,
    protocol_fingerprint,
)


def test_v18_protocol_has_balanced_frozen_design() -> None:
    protocol = load_study_protocol(PROJECT_ROOT / "config/studies/v18.json")
    assert protocol.study_id == "v18-condition-aware-listwise-160"
    assert protocol.predecessor_release == "v17.0.1"
    assert protocol.query_design.total_queries == 160
    assert protocol.query_design.source_group_count == 80
    assert protocol.query_design.splits == {"calibration": 80, "holdout": 80}
    assert len(protocol_fingerprint(protocol)) == 64


def test_protocol_rejects_inconsistent_counts() -> None:
    payload = json.loads(
        (PROJECT_ROOT / "config/studies/v18.json").read_text(encoding="utf-8")
    )
    payload["query_design"]["total_queries"] = 159
    with pytest.raises(ProtocolError, match="queries per group"):
        StudyProtocol.from_mapping(payload)


def test_protocol_requires_two_independent_reviewers() -> None:
    payload = json.loads(
        (PROJECT_ROOT / "config/studies/v18.json").read_text(encoding="utf-8")
    )
    payload["review_design"]["independent_relevance_reviewers"] = 1
    with pytest.raises(ProtocolError, match="at least two"):
        StudyProtocol.from_mapping(payload)
