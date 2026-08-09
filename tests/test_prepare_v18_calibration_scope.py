from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from ocr_vlm_retrieval.studies.query_split import query_set_fingerprint


def _load_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts/prepare_v18_calibration_scope.py"
    )
    spec = importlib.util.spec_from_file_location(
        "prepare_v18_calibration_scope", path
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, split in enumerate(("calibration", "holdout"), start=1):
        group_id = f"group_{index}"
        for role_index, role in enumerate(
            ("positive", "single_condition_hard_negative"), start=1
        ):
            rows.append(
                {
                    "query_id": f"v18_query_{(index - 1) * 2 + role_index:03d}",
                    "query": f"第{index}组测试查询角色{role_index}",
                    "split": split,
                    "group_id": group_id,
                    "query_role": role,
                    "review_status": "human_query_approved",
                }
            )
    return rows


def _receipt(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "study_id": "study",
        "protocol_sha256": "protocol",
        "query_count": 4,
        "query_set_sha256": query_set_fingerprint(rows),
        "retrieval_executed": False,
        "holdout_results_opened": False,
        "v17_artifacts_modified": False,
    }


def test_scope_exports_calibration_and_never_holdout() -> None:
    module = _load_module()
    rows = _rows()
    calibration = module.isolate_calibration_rows(
        rows,
        _receipt(rows),
        study_id="study",
        protocol_sha256="protocol",
        expected_total=4,
        expected_calibration=2,
        expected_holdout=2,
    )
    assert len(calibration) == 2
    assert {row["split"] for row in calibration} == {"calibration"}


def test_scope_refuses_prior_holdout_access() -> None:
    module = _load_module()
    rows = _rows()
    receipt = _receipt(rows)
    receipt["holdout_results_opened"] = True
    with pytest.raises(ValueError, match="holdout was already marked as opened"):
        module.isolate_calibration_rows(
            rows,
            receipt,
            study_id="study",
            protocol_sha256="protocol",
            expected_total=4,
            expected_calibration=2,
            expected_holdout=2,
        )
