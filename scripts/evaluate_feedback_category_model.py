"""Evaluate a frozen feedback classifier without retraining it."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_category_review_splits import parse_bool
from scripts.taxonomy import CATEGORY_LABELS
from scripts.train_feedback_category_model import (
    CATEGORY_KEYS,
    aligned_probabilities,
    build_feature_matrix,
    effective_category,
    is_category_review,
    load_ocr_features,
    load_text_features,
    load_visual_features,
    original_category,
    read_jsonl,
    read_reviews,
)


DEFAULT_LIBRARY_DIR = PROJECT_ROOT / "outputs/user_library"
DEFAULT_MODEL_PATH = (
    DEFAULT_LIBRARY_DIR
    / "feedback_model/category_feedback_model.joblib"
)
DEFAULT_REVIEWS_PATH = (
    PROJECT_ROOT
    / "data/evaluation/public_dataset_1500_quality_human_reviews.csv"
)
DEFAULT_SPLITS_PATH = (
    PROJECT_ROOT
    / "data/evaluation/public_dataset_1500_category_review_splits.csv"
)
DEFAULT_OUTPUT_DIR = (
    DEFAULT_LIBRARY_DIR / "feedback_model/evaluation"
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def expected_calibration_error(
    probabilities: np.ndarray,
    truth: np.ndarray,
    predictions: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    confidence = probabilities.max(axis=1)
    correct = predictions == truth
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for index in range(bins):
        lower = edges[index]
        upper = edges[index + 1]
        in_bin = (
            (confidence >= lower)
            & (
                confidence <= upper
                if index == bins - 1
                else confidence < upper
            )
        )
        if not in_bin.any():
            continue
        error += float(in_bin.mean()) * abs(
            float(correct[in_bin].mean())
            - float(confidence[in_bin].mean())
        )
    return error


def ensure_split_allowed(split: str, allow_test: bool) -> None:
    if split == "test" and not allow_test:
        raise ValueError(
            "Test evaluation is locked. Complete validation and freeze the "
            "full method before rerunning with --allow-test."
        )


def conservative_predictions(
    baseline: np.ndarray,
    model_predictions: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    confidence = probabilities.max(axis=1)
    override_mask = (
        (model_predictions != baseline)
        & (confidence >= threshold)
    )
    predictions = baseline.copy()
    predictions[override_mask] = model_predictions[override_mask]
    return predictions, override_mask


def evaluate_frozen_model(
    *,
    library_dir: Path,
    model_path: Path,
    reviews_path: Path,
    splits_path: Path,
    output_dir: Path,
    split: str,
    allow_test: bool,
    override_threshold: float,
) -> dict[str, Any]:
    ensure_split_allowed(split, allow_test)
    manifest = read_jsonl(library_dir / "manifest.jsonl")
    reviews = read_reviews(reviews_path)
    split_rows = read_csv(splits_path)
    required_ids = {
        row["item_id"]
        for row in split_rows
        if row.get("split") == split
        and parse_bool(row.get("review_required", False))
    }
    missing_reviews = sorted(required_ids - set(reviews))
    if missing_reviews:
        raise ValueError(
            f"{split} review is incomplete: "
            f"{len(missing_reviews)} required pages remain."
        )

    model_bundle = joblib.load(model_path)
    if tuple(model_bundle.get("category_keys", ())) != CATEGORY_KEYS:
        raise ValueError("Model taxonomy does not match current taxonomy.")

    visual_features = load_visual_features(library_dir)
    text_features = load_text_features(library_dir)
    ocr_features = load_ocr_features(library_dir)
    rows, matrix = build_feature_matrix(
        manifest,
        visual_features,
        text_features,
        ocr_features,
    )
    eligible_indices = [
        index
        for index, row in enumerate(rows)
        if row["item_id"] in required_ids
        and is_category_review(reviews.get(row["item_id"]))
    ]
    excluded_ids = sorted(
        required_ids
        - {rows[index]["item_id"] for index in eligible_indices}
    )
    if not eligible_indices:
        raise ValueError(f"No usable human labels in {split} split.")

    evaluation_matrix = matrix[eligible_indices]
    truth = np.array(
        [
            effective_category(
                rows[index],
                reviews.get(rows[index]["item_id"]),
            )
            for index in eligible_indices
        ],
        dtype=object,
    )
    current_baseline = np.array(
        [str(rows[index]["category"]) for index in eligible_indices],
        dtype=object,
    )
    legacy_baseline = np.array(
        [original_category(rows[index]) for index in eligible_indices],
        dtype=object,
    )
    probabilities = aligned_probabilities(
        model_bundle["model"], evaluation_matrix
    )
    predicted_indices = np.argmax(probabilities, axis=1)
    model_predictions = np.array(
        [CATEGORY_KEYS[index] for index in predicted_indices],
        dtype=object,
    )
    predictions, override_mask = conservative_predictions(
        current_baseline,
        model_predictions,
        probabilities,
        threshold=override_threshold,
    )

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            truth,
            predictions,
            labels=list(CATEGORY_KEYS),
            zero_division=0,
        )
    )
    per_class_rows = [
        {
            "category": category,
            "label": CATEGORY_LABELS[category],
            "precision": round(float(precision[index]), 6),
            "recall": round(float(recall[index]), 6),
            "f1": round(float(f1[index]), 6),
            "support": int(support[index]),
        }
        for index, category in enumerate(CATEGORY_KEYS)
    ]
    confusion = confusion_matrix(
        truth, predictions, labels=list(CATEGORY_KEYS)
    )
    confusion_rows = [
        {
            "true_category": truth_category,
            **{
                predicted_category: int(
                    confusion[row_index, column_index]
                )
                for column_index, predicted_category in enumerate(
                    CATEGORY_KEYS
                )
            },
        }
        for row_index, truth_category in enumerate(CATEGORY_KEYS)
    ]
    prediction_rows = []
    for local_index, source_index in enumerate(eligible_indices):
        row = rows[source_index]
        prediction_rows.append(
            {
                "item_id": row["item_id"],
                "truth": truth[local_index],
                "prediction": predictions[local_index],
                "model_prediction": model_predictions[local_index],
                "current_baseline": current_baseline[local_index],
                "legacy_baseline": legacy_baseline[local_index],
                "confidence": round(
                    float(probabilities[local_index].max()), 6
                ),
                "correct": str(
                    predictions[local_index] == truth[local_index]
                ).lower(),
                "override_applied": str(
                    override_mask[local_index]
                ).lower(),
                "source_file_name": row.get("source_file_name", ""),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{split}_report.json"
    prediction_path = output_dir / f"{split}_predictions.csv"
    confusion_path = output_dir / f"{split}_confusion_matrix.csv"
    per_class_path = output_dir / f"{split}_per_class_metrics.csv"
    report = {
        "status": "success",
        "split": split,
        "model_path": str(model_path),
        "model_trained_at": model_bundle.get("trained_at", ""),
        "required_review_pages": len(required_ids),
        "evaluated_pages": len(eligible_indices),
        "excluded_review_ids": excluded_ids,
        "metrics": {
            "accuracy": round(
                float(accuracy_score(truth, predictions)), 6
            ),
            "macro_f1": round(
                float(
                    f1_score(
                        truth,
                        predictions,
                        labels=list(CATEGORY_KEYS),
                        average="macro",
                        zero_division=0,
                    )
                ),
                6,
            ),
            "current_baseline_accuracy": round(
                float(accuracy_score(truth, current_baseline)), 6
            ),
            "standalone_model_accuracy": round(
                float(
                    accuracy_score(truth, model_predictions)
                ),
                6,
            ),
            "standalone_model_macro_f1": round(
                float(
                    f1_score(
                        truth,
                        model_predictions,
                        labels=list(CATEGORY_KEYS),
                        average="macro",
                        zero_division=0,
                    )
                ),
                6,
            ),
            "legacy_baseline_accuracy": round(
                float(accuracy_score(truth, legacy_baseline)), 6
            ),
            "override_threshold": override_threshold,
            "override_count": int(override_mask.sum()),
            "mean_confidence": round(
                float(probabilities.max(axis=1).mean()), 6
            ),
            "ece_10_bins": round(
                expected_calibration_error(
                    probabilities,
                    truth,
                    model_predictions,
                    bins=10,
                ),
                6,
            ),
        },
        "per_class": per_class_rows,
        "artifacts": {
            "predictions": str(prediction_path),
            "confusion_matrix": str(confusion_path),
            "per_class_metrics": str(per_class_path),
        },
        "guard": (
            "This script loaded a frozen model and did not fit or update it."
        ),
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(
        prediction_path,
        prediction_rows,
        [
            "item_id",
            "truth",
            "prediction",
            "model_prediction",
            "current_baseline",
            "legacy_baseline",
            "confidence",
            "correct",
            "override_applied",
            "source_file_name",
        ],
    )
    write_csv(
        confusion_path,
        confusion_rows,
        ["true_category", *CATEGORY_KEYS],
    )
    write_csv(
        per_class_path,
        per_class_rows,
        ["category", "label", "precision", "recall", "f1", "support"],
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--library-dir", type=Path, default=DEFAULT_LIBRARY_DIR
    )
    parser.add_argument(
        "--model", type=Path, default=DEFAULT_MODEL_PATH
    )
    parser.add_argument(
        "--reviews", type=Path, default=DEFAULT_REVIEWS_PATH
    )
    parser.add_argument(
        "--splits", type=Path, default=DEFAULT_SPLITS_PATH
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR
    )
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="validation",
    )
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument(
        "--override-threshold", type=float, default=0.80
    )
    return parser.parse_args()


def resolved(path: Path) -> Path:
    return (
        path.resolve()
        if path.is_absolute()
        else (PROJECT_ROOT / path).resolve()
    )


def main() -> None:
    args = parse_args()
    report = evaluate_frozen_model(
        library_dir=resolved(args.library_dir),
        model_path=resolved(args.model),
        reviews_path=resolved(args.reviews),
        splits_path=resolved(args.splits),
        output_dir=resolved(args.output_dir),
        split=args.split,
        allow_test=args.allow_test,
        override_threshold=args.override_threshold,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
