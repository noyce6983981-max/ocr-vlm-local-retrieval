"""Attach current Top-3 results to open-set human-review queries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queue",
        type=Path,
        default=Path(
            "data/evaluation/"
            "public_dataset_1500_retrieval_query_queue_100.csv"
        ),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "data/evaluation/"
            "public_dataset_1500_retrieval_query_protocol.json"
        ),
    )
    parser.add_argument(
        "--library-dir",
        type=Path,
        default=Path("outputs/user_library"),
    )
    parser.add_argument(
        "--backup",
        type=Path,
        default=Path(
            "data/evaluation/archive/"
            "public_dataset_1500_retrieval_query_queue_100_before_candidates.csv"
        ),
    )
    return parser.parse_args()


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    args = parse_args()
    queue_path = project_path(args.queue)
    protocol_path = project_path(args.protocol)
    library_dir = project_path(args.library_dir)
    backup_path = project_path(args.backup)
    with queue_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0])
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(queue_path, backup_path)
    open_rows = [row for row in rows if row["query_type"] == "no_answer"]

    acceptance_counts: Counter[str] = Counter()
    for index, row in enumerate(open_rows, start=1):
        output_path = (
            PROJECT_ROOT
            / "outputs/retrieval_query_review_cache"
            / f"{row['query_id']}.json"
        )
        completed = subprocess.run(
            [
                str(PROJECT_ROOT / ".venv/Scripts/python.exe"),
                str(PROJECT_ROOT / "scripts/live_search.py"),
                row["query"],
                "--library-dir",
                str(library_dir),
                "--output",
                str(output_path),
            ],
            cwd=PROJECT_ROOT,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=300,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"{row['query_id']} retrieval failed:\n"
                + (completed.stderr or completed.stdout)[-1800:]
            )
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        ranking = payload["rankings"]["quality_hybrid"]
        decision = payload["acceptance"]["quality_hybrid"]
        row["candidate_item_ids"] = ";".join(
            result["item_id"] for result in ranking[:3]
        )
        row["system_acceptance"] = str(bool(decision["accepted"]))
        row["system_acceptance_reason"] = decision["reason"]
        row["system_top1_score"] = str(ranking[0]["score"])
        acceptance_counts[row["system_acceptance"]] += 1
        print(
            f"[{index}/{len(open_rows)}] {row['query_id']} "
            f"accepted={row['system_acceptance']}",
            flush=True,
        )

    temporary = queue_path.with_suffix(queue_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(queue_path)

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["open_set_candidates_attached"] = len(open_rows)
    protocol["open_set_system_acceptance_counts"] = dict(acceptance_counts)
    protocol["queue_sha256"] = hashlib.sha256(queue_path.read_bytes()).hexdigest()
    protocol_path.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(protocol, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
