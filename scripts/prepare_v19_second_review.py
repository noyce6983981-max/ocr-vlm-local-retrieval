from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.v19_formal_family_review_app import (  # noqa: E402
    ROUTE_LABELS,
    family_snapshot_sha256,
    load_families,
)

METHOD_LOCK = ROOT / "data/evaluation/v19/formal/method_lock.json"
DEFAULT_OUTPUT = (
    ROOT / "data/evaluation/v19/formal/second_review_manifest.json"
)


def select_families(
    families: list[dict[str, Any]], seed: str
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for route_index, route in enumerate(ROUTE_LABELS):
        calibration_count = 2 if route_index % 2 == 0 else 1
        holdout_count = 3 - calibration_count
        for split, count in (
            ("calibration", calibration_count),
            ("holdout", holdout_count),
        ):
            pool = [
                family
                for family in families
                if family["proposed_route"] == route and family["split"] == split
            ]
            ranked = sorted(
                pool,
                key=lambda family: hashlib.sha256(
                    f"{seed}:{family['family_id']}".encode()
                ).hexdigest(),
            )
            if len(ranked) < count:
                raise ValueError(f"insufficient {route}/{split} families")
            selected.extend(ranked[:count])
    if len(selected) != 18 or len({row["family_id"] for row in selected}) != 18:
        raise ValueError("second review must select 18 unique families")
    return selected


def build_manifest(
    families: list[dict[str, Any]], method_lock_sha256: str
) -> dict[str, Any]:
    selected = select_families(families, method_lock_sha256)
    split_counts = Counter(str(row["split"]) for row in selected)
    route_counts = Counter(str(row["proposed_route"]) for row in selected)
    if split_counts != {"calibration": 9, "holdout": 9}:
        raise ValueError(f"unexpected second-review split counts: {split_counts}")
    if set(route_counts.values()) != {3}:
        raise ValueError(f"unexpected second-review route counts: {route_counts}")
    return {
        "schema_version": 1,
        "study_id": "v19-local-llm-structured-intent-routing",
        "review_stage": "independent_blind_second_review",
        "method_lock_sha256": method_lock_sha256,
        "family_count": 18,
        "query_count": 72,
        "double_review_fraction": 0.3,
        "split_counts": dict(split_counts),
        "selection_strata_verified": True,
        "primary_labels_exposed": False,
        "families": [
            {
                "family_id": str(row["family_id"]),
                "family_snapshot_sha256": family_snapshot_sha256(row),
                "split": str(row["split"]),
                "query_ids": list(row["query_ids"]),
                "queries": list(row["queries"]),
            }
            for row in selected
        ],
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def main() -> int:
    method_lock_sha256 = hashlib.sha256(METHOD_LOCK.read_bytes()).hexdigest()
    manifest = build_manifest(load_families(), method_lock_sha256)
    write_json_atomic(DEFAULT_OUTPUT, manifest)
    print(
        "V19 blind second-review manifest prepared: "
        f"families={manifest['family_count']}, queries={manifest['query_count']}"
    )
    print(f"Wrote {DEFAULT_OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
