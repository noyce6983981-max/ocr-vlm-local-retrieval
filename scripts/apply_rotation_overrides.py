"""Install accepted rotation-retry OCR texts as reversible retrieval overrides."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path(
            "data/evaluation/public_dataset_200_rotation_retry.csv"
        ),
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path("work/rotation_retry_001"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/user_library/ocr/overrides"),
    )
    return parser.parse_args()


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def main() -> None:
    args = parse_args()
    results_path = project_path(args.results)
    artifacts_dir = project_path(args.artifacts)
    output_dir = project_path(args.output)
    with results_path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    accepted = [
        row
        for row in rows
        if parse_bool(row["is_best"])
        and parse_bool(row["rotation_accepted"])
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    installed = []
    for row in accepted:
        angle = int(row["angle"])
        source_path = (
            artifacts_dir
            / f"{row['item_id']}_rotation_{angle}.json"
        )
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        payload["_retrieval_override"] = {
            "method": "selective_four_rotation_ocr",
            "rotation_degrees": angle,
            "source_experiment": results_path.relative_to(
                PROJECT_ROOT
            ).as_posix(),
            "baseline_preserved": True,
        }
        output_path = output_dir / f"{row['item_id']}.json"
        temp_path = output_path.with_suffix(".json.tmp")
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(temp_path, output_path)
        installed.append(
            {
                "item_id": row["item_id"],
                "filename": row["filename"],
                "rotation_degrees": angle,
                "character_count": int(row["character_count"]),
                "mean_confidence": float(row["mean_confidence"]),
            }
        )
    print(
        json.dumps(
            {
                "status": "success",
                "installed_overrides": len(installed),
                "items": installed,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
