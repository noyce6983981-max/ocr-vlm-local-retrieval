from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


PRIMARY_ARTIFACTS = (
    "category_feedback_model.joblib",
    "active_learning_queue.csv",
    "cross_validation_confusion_matrix.csv",
    "feedback_training_report.json",
)
EVALUATION_ARTIFACTS = {
    "validation": (
        "validation_report.json",
        "validation_predictions.csv",
        "validation_confusion_matrix.csv",
        "validation_per_class_metrics.csv",
    ),
    "test": (
        "test_report.json",
        "test_predictions.csv",
        "test_confusion_matrix.csv",
        "test_per_class_metrics.csv",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--snapshot-name", required=True)
    args = parser.parse_args()

    formal_dir = args.formal_dir.resolve()
    candidate_dir = args.candidate_dir.resolve()
    snapshot_dir = formal_dir / "frozen" / args.snapshot_name
    if snapshot_dir.exists():
        raise FileExistsError(f"snapshot already exists: {snapshot_dir}")
    for name in PRIMARY_ARTIFACTS:
        if not (formal_dir / name).is_file():
            raise FileNotFoundError(formal_dir / name)
        if not (candidate_dir / name).is_file():
            raise FileNotFoundError(candidate_dir / name)
    for split, names in EVALUATION_ARTIFACTS.items():
        source_dir = args.validation_dir if split == "validation" else args.test_dir
        for name in names:
            if not (source_dir / name).is_file():
                raise FileNotFoundError(source_dir / name)

    old_hash = sha256(formal_dir / "category_feedback_model.joblib")
    candidate_hash = sha256(candidate_dir / "category_feedback_model.joblib")
    candidate_report = json.loads(
        (candidate_dir / "feedback_training_report.json").read_text(encoding="utf-8-sig")
    )
    validation_report = json.loads(
        (args.validation_dir / "validation_report.json").read_text(encoding="utf-8-sig")
    )
    test_report = json.loads(
        (args.test_dir / "test_report.json").read_text(encoding="utf-8-sig")
    )
    if candidate_report.get("status") != "success":
        raise ValueError("candidate training report is not successful")
    if validation_report.get("evaluated_pages") != 95:
        raise ValueError("validation evaluation is incomplete")
    if test_report.get("evaluated_pages") != 95:
        raise ValueError("test evaluation is incomplete")

    snapshot_dir.mkdir(parents=True)
    for name in PRIMARY_ARTIFACTS:
        shutil.copy2(formal_dir / name, snapshot_dir / name)
    if (formal_dir / "evaluation").is_dir():
        shutil.copytree(
            formal_dir / "evaluation",
            snapshot_dir / "evaluation",
        )

    for name in PRIMARY_ARTIFACTS:
        if name != "feedback_training_report.json":
            atomic_copy(candidate_dir / name, formal_dir / name)

    deployed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    candidate_report["artifacts"] = {
        "model": "outputs\\user_library\\feedback_model\\category_feedback_model.joblib",
        "queue": "outputs\\user_library\\feedback_model\\active_learning_queue.csv",
        "confusion_matrix": (
            "outputs\\user_library\\feedback_model\\"
            "cross_validation_confusion_matrix.csv"
        ),
    }
    candidate_report["deployment"] = {
        "status": "deployed",
        "deployed_at": deployed_at,
        "selection_rule": (
            "validation accuracy and macro_f1 tied; selected lower validation ECE"
        ),
        "validation_metrics": validation_report["metrics"],
        "test_metrics": test_report["metrics"],
        "previous_model_sha256": old_hash,
        "deployed_model_sha256": candidate_hash,
        "snapshot": str(snapshot_dir),
    }
    write_json(formal_dir / "feedback_training_report.json", candidate_report)

    evaluation_dir = formal_dir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    formal_model_path = str(formal_dir / "category_feedback_model.joblib")
    for split, names in EVALUATION_ARTIFACTS.items():
        source_dir = args.validation_dir if split == "validation" else args.test_dir
        for name in names:
            source = source_dir / name
            destination = evaluation_dir / name
            if name.endswith("_report.json"):
                report = json.loads(source.read_text(encoding="utf-8-sig"))
                report["model_path"] = formal_model_path
                report["artifacts"] = {
                    key: str(evaluation_dir / Path(value).name)
                    for key, value in report.get("artifacts", {}).items()
                }
                write_json(destination, report)
            else:
                atomic_copy(source, destination)

    deployed_hash = sha256(formal_dir / "category_feedback_model.joblib")
    if deployed_hash != candidate_hash:
        raise RuntimeError("deployed model hash does not match candidate")
    print(
        json.dumps(
            {
                "status": "deployed",
                "previous_hash": old_hash,
                "deployed_hash": deployed_hash,
                "snapshot": str(snapshot_dir),
                "validation": validation_report["metrics"],
                "test": test_report["metrics"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
