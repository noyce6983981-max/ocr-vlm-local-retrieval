"""Train a weakly supervised category model with human-review overrides."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import faiss
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_category_review_splits import build_groups
from scripts.assistant_category_proposals import (
    is_active_assistant_proposal,
    read_assistant_proposals,
)
from scripts.taxonomy import CATEGORY_LABELS, normalize_category


DEFAULT_LIBRARY_DIR = PROJECT_ROOT / "outputs/user_library"
DEFAULT_REVIEWS_PATH = (
    PROJECT_ROOT
    / "data/evaluation/public_dataset_1500_quality_human_reviews.csv"
)
DEFAULT_REVIEW_SPLITS_PATH = (
    PROJECT_ROOT
    / "data/evaluation/public_dataset_1500_category_review_splits.csv"
)
DEFAULT_ASSISTANT_PROPOSALS_PATH = (
    PROJECT_ROOT
    / "data/evaluation/public_dataset_1500_category_assistant_proposals.csv"
)
CATEGORY_KEYS = tuple(CATEGORY_LABELS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--library-dir",
        type=Path,
        default=DEFAULT_LIBRARY_DIR,
    )
    parser.add_argument(
        "--reviews",
        type=Path,
        default=DEFAULT_REVIEWS_PATH,
    )
    parser.add_argument(
        "--review-splits",
        type=Path,
        default=DEFAULT_REVIEW_SPLITS_PATH,
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--review-weight", type=float, default=5.0)
    parser.add_argument(
        "--assistant-proposals",
        type=Path,
        default=None,
        help=(
            "Optional AI proposal CSV. It is never treated as human truth "
            "and is only used on the frozen training split."
        ),
    )
    parser.add_argument("--assistant-weight", type=float, default=1.5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional artifact directory; defaults to feedback_model.",
    )
    parser.add_argument("--queue-size", type=int, default=300)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_reviews(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            row["item_id"]: row
            for row in csv.DictReader(handle)
            if row.get("item_id")
        }


def read_review_splits(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(
            "Frozen category-review split not found. Run "
            "`python scripts/build_category_review_splits.py` first."
        )
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assignments: dict[str, str] = {}
    for row in rows:
        item_id = str(row.get("item_id", "")).strip()
        split = str(row.get("split", "")).strip()
        if not item_id:
            continue
        if item_id in assignments:
            raise ValueError(
                f"Duplicate item ID in review split: {item_id}"
            )
        if split not in {"train", "validation", "test"}:
            raise ValueError(
                f"Unsupported review split for {item_id}: {split}"
            )
        assignments[item_id] = split
    return assignments


def is_category_review(review: dict[str, str] | None) -> bool:
    return bool(
        review
        and review.get("decision") in {"accepted", "reclassified"}
    )


def resolve_row_splits(
    rows: list[dict[str, Any]],
    review_splits: dict[str, str],
) -> np.ndarray:
    missing_assignments = sorted(
        row["item_id"]
        for row in rows
        if row["item_id"] not in review_splits
    )
    if missing_assignments:
        raise ValueError(
            "Active pages missing from frozen review split: "
            + ", ".join(missing_assignments[:5])
        )
    return np.array(
        [review_splits[row["item_id"]] for row in rows],
        dtype=object,
    )


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def load_visual_features(
    library_dir: Path,
) -> dict[str, np.ndarray]:
    metadata = read_jsonl(library_dir / "visual_index/metadata.jsonl")
    embeddings = np.load(
        library_dir / "visual_index/embeddings.npy",
        mmap_mode="r",
    )
    if len(metadata) != len(embeddings):
        raise ValueError("Visual metadata and embeddings are misaligned.")
    return {
        row["item_id"]: np.asarray(embeddings[index], dtype=np.float32)
        for index, row in enumerate(metadata)
    }


def load_text_features(
    library_dir: Path,
) -> dict[str, np.ndarray]:
    metadata = read_jsonl(library_dir / "text_index/metadata.jsonl")
    index_bytes = np.frombuffer(
        (library_dir / "text_index/index.faiss").read_bytes(),
        dtype=np.uint8,
    )
    index = faiss.deserialize_index(index_bytes)
    if len(metadata) != index.ntotal:
        raise ValueError("Text metadata and FAISS index are misaligned.")
    vectors = index.reconstruct_n(0, index.ntotal).astype(
        np.float32, copy=False
    )
    sums: dict[str, np.ndarray] = {}
    counts: Counter[str] = Counter()
    for row, vector in zip(metadata, vectors, strict=True):
        item_id = row["item_id"]
        if item_id not in sums:
            sums[item_id] = np.zeros(index.d, dtype=np.float32)
        sums[item_id] += vector
        counts[item_id] += 1
    item_ids = list(sums)
    matrix = np.vstack(
        [sums[item_id] / counts[item_id] for item_id in item_ids]
    )
    matrix = l2_normalize(matrix)
    return dict(zip(item_ids, matrix, strict=True))


def load_ocr_features(library_dir: Path) -> dict[str, np.ndarray]:
    path = library_dir / "ocr/summary.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    features: dict[str, np.ndarray] = {}
    for row in rows:
        features[row["item_id"]] = np.array(
            [
                float(row.get("mean_confidence", 0.0) or 0.0),
                math.log1p(
                    int(float(row.get("character_count", 0) or 0))
                )
                / 10.0,
                math.log1p(
                    int(float(row.get("text_box_count", 0) or 0))
                )
                / 8.0,
                float(row.get("empty_rate", 0.0) or 0.0),
            ],
            dtype=np.float32,
        )
    return features


def effective_category(
    row: dict[str, Any],
    review: dict[str, str] | None,
    assistant_proposal: dict[str, str] | None = None,
) -> str:
    if (
        review
        and review.get("decision") == "reclassified"
        and review.get("revised_category") in CATEGORY_KEYS
    ):
        return review["revised_category"]
    if review and review.get("decision") == "accepted":
        return str(row.get("category", ""))
    if is_active_assistant_proposal(assistant_proposal):
        return assistant_proposal["proposed_category"]
    return str(row.get("category", ""))


def original_category(row: dict[str, Any]) -> str:
    return normalize_category(
        str(
            row.get("taxonomy_v1_category")
            or row.get("original_category")
            or row.get("category", "")
        )
    )


def build_feature_matrix(
    manifest: list[dict[str, Any]],
    visual_features: dict[str, np.ndarray],
    text_features: dict[str, np.ndarray],
    ocr_features: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], np.ndarray]:
    active_rows = [
        row
        for row in manifest
        if bool(row.get("search_enabled", True))
        and row.get("category") in CATEGORY_KEYS
        and row["item_id"] in visual_features
    ]
    if not active_rows:
        raise ValueError("No active pages have visual features.")
    if not text_features:
        raise ValueError("Text feature index is empty.")
    text_dimension = len(next(iter(text_features.values())))
    ocr_dimension = 4
    matrix = np.vstack(
        [
            np.concatenate(
                [
                    l2_normalize(
                        visual_features[row["item_id"]][None, :]
                    )[0],
                    text_features.get(
                        row["item_id"],
                        np.zeros(text_dimension, dtype=np.float32),
                    ),
                    ocr_features.get(
                        row["item_id"],
                        np.zeros(ocr_dimension, dtype=np.float32),
                    ),
                ]
            )
            for row in active_rows
        ]
    ).astype(np.float32, copy=False)
    return active_rows, matrix


def new_classifier() -> LogisticRegression:
    return LogisticRegression(
        C=0.7,
        class_weight="balanced",
        max_iter=700,
        solver="lbfgs",
        random_state=42,
    )


def aligned_probabilities(
    model: LogisticRegression,
    matrix: np.ndarray,
) -> np.ndarray:
    probabilities = model.predict_proba(matrix)
    aligned = np.zeros(
        (len(matrix), len(CATEGORY_KEYS)),
        dtype=np.float32,
    )
    category_to_index = {
        category: index for index, category in enumerate(CATEGORY_KEYS)
    }
    for source_index, category in enumerate(model.classes_):
        aligned[:, category_to_index[str(category)]] = probabilities[
            :, source_index
        ]
    return aligned


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def train_feedback_model(
    *,
    library_dir: Path,
    reviews_path: Path,
    review_splits_path: Path,
    assistant_proposals_path: Path | None,
    folds: int,
    review_weight: float,
    assistant_weight: float,
    queue_size: int,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    manifest = read_jsonl(library_dir / "manifest.jsonl")
    reviews = read_reviews(reviews_path)
    review_splits = read_review_splits(review_splits_path)
    assistant_proposals = read_assistant_proposals(
        assistant_proposals_path
    )
    visual_features = load_visual_features(library_dir)
    text_features = load_text_features(library_dir)
    ocr_features = load_ocr_features(library_dir)
    rows, matrix = build_feature_matrix(
        manifest,
        visual_features,
        text_features,
        ocr_features,
    )
    labels = np.array(
        [
            effective_category(
                row,
                reviews.get(row["item_id"]),
                (
                    assistant_proposals.get(row["item_id"])
                    if review_splits.get(row["item_id"]) == "train"
                    else None
                ),
            )
            for row in rows
        ],
        dtype=object,
    )
    frozen_group_by_item, _ = build_groups(manifest)
    groups = np.array(
        [frozen_group_by_item[row["item_id"]] for row in rows],
        dtype=object,
    )
    splits = resolve_row_splits(rows, review_splits)
    train_mask = splits == "train"
    reviewed_mask = np.array(
        [
            is_category_review(reviews.get(row["item_id"]))
            for row in rows
        ],
        dtype=bool,
    )
    assistant_mask = np.array(
        [
            (
                not reviewed_mask[index]
                and review_splits.get(row["item_id"]) == "train"
                and is_active_assistant_proposal(
                    assistant_proposals.get(row["item_id"])
                )
            )
            for index, row in enumerate(rows)
        ],
        dtype=bool,
    )
    weights = np.where(
        reviewed_mask,
        review_weight,
        np.where(assistant_mask, assistant_weight, 1.0),
    )

    class_counts = Counter(str(value) for value in labels[train_mask])
    if set(class_counts) != set(CATEGORY_KEYS):
        missing = sorted(set(CATEGORY_KEYS) - set(class_counts))
        raise ValueError(
            "Training split is missing active categories: "
            + ", ".join(missing)
        )
    if folds < 2 or min(class_counts.values()) < folds:
        raise ValueError("Not enough samples per category for cross-validation.")

    fit_indices = np.flatnonzero(train_mask)
    train_matrix = matrix[fit_indices]
    train_labels = labels[fit_indices]
    train_groups = groups[fit_indices]
    train_weights = weights[fit_indices]
    splitter = StratifiedGroupKFold(
        n_splits=folds,
        shuffle=True,
        random_state=42,
    )
    train_out_of_fold = np.zeros(
        (len(fit_indices), len(CATEGORY_KEYS)),
        dtype=np.float32,
    )
    for fold_train_indices, fold_validation_indices in splitter.split(
        train_matrix,
        train_labels,
        train_groups,
    ):
        model = new_classifier()
        model.fit(
            train_matrix[fold_train_indices],
            train_labels[fold_train_indices],
            sample_weight=train_weights[fold_train_indices],
        )
        train_out_of_fold[
            fold_validation_indices
        ] = aligned_probabilities(
            model,
            train_matrix[fold_validation_indices],
        )

    train_predicted_indices = np.argmax(train_out_of_fold, axis=1)
    train_predictions = np.array(
        [CATEGORY_KEYS[index] for index in train_predicted_indices],
        dtype=object,
    )
    train_reviewed_local_indices = np.flatnonzero(
        reviewed_mask[train_mask]
    )
    train_reviewed_global_indices = fit_indices[
        train_reviewed_local_indices
    ]
    reviewed_labels = train_labels[train_reviewed_local_indices]
    reviewed_predictions = train_predictions[
        train_reviewed_local_indices
    ]
    reviewed_original = np.array(
        [
            original_category(rows[index])
            for index in train_reviewed_global_indices
        ],
        dtype=object,
    )

    final_model = new_classifier()
    final_model.fit(
        train_matrix,
        train_labels,
        sample_weight=train_weights,
    )
    full_probabilities = aligned_probabilities(final_model, matrix)
    full_predicted_indices = np.argmax(full_probabilities, axis=1)
    full_predictions = np.array(
        [CATEGORY_KEYS[index] for index in full_predicted_indices],
        dtype=object,
    )

    queue_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if (
            reviewed_mask[index]
            or assistant_mask[index]
            or splits[index] != "train"
        ):
            continue
        probabilities = full_probabilities[index]
        order = np.argsort(-probabilities)
        predicted_category = CATEGORY_KEYS[int(order[0])]
        confidence = float(probabilities[order[0]])
        margin = float(
            probabilities[order[0]] - probabilities[order[1]]
        )
        current_category = str(labels[index])
        disagreement = predicted_category != current_category
        priority_score = min(
            1.0,
            (1.0 - confidence) * 0.60
            + (1.0 - margin) * 0.15
            + (0.25 if disagreement else 0.0),
        )
        queue_rows.append(
            {
                "item_id": row["item_id"],
                "source_file_name": row.get("source_file_name", ""),
                "current_category": current_category,
                "predicted_category": predicted_category,
                "predicted_label": CATEGORY_LABELS[predicted_category],
                "confidence": round(confidence, 6),
                "margin": round(margin, 6),
                "priority_score": round(priority_score, 6),
                "priority_reason": (
                    "模型与当前类别冲突"
                    if disagreement
                    else "模型不确定"
                ),
                "perceptual_group": row.get("perceptual_group", ""),
            }
        )
    queue_rows.sort(
        key=lambda row: (
            -float(row["priority_score"]),
            str(row["item_id"]),
        )
    )
    queue_rows = queue_rows[:queue_size]

    output_dir = output_dir or (library_dir / "feedback_model")
    output_dir.mkdir(parents=True, exist_ok=True)
    trained_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    model_path = output_dir / "category_feedback_model.joblib"
    joblib.dump(
        {
            "model": final_model,
            "category_keys": CATEGORY_KEYS,
            "category_labels": CATEGORY_LABELS,
            "feature_order": (
                "qwen3_vl_visual_embedding",
                "bge_m3_mean_text_embedding",
                "ocr_quality_features",
            ),
            "trained_at": trained_at,
            "training_protocol": {
                "split_file": str(
                    review_splits_path.relative_to(PROJECT_ROOT)
                ),
                "fit_split": "train",
                "excluded_splits": ["validation", "test"],
                "train_page_count": int(train_mask.sum()),
                "assistant_proposals_file": (
                    str(
                        assistant_proposals_path.relative_to(
                            PROJECT_ROOT
                        )
                    )
                    if assistant_proposals_path
                    and assistant_proposals_path.is_relative_to(
                        PROJECT_ROOT
                    )
                    else str(assistant_proposals_path or "")
                ),
                "assistant_proposal_count": int(
                    assistant_mask.sum()
                ),
                "assistant_weight": assistant_weight,
            },
        },
        model_path,
    )

    confusion = confusion_matrix(
        train_labels,
        train_predictions,
        labels=list(CATEGORY_KEYS),
    )
    confusion_rows = []
    for row_index, truth in enumerate(CATEGORY_KEYS):
        confusion_rows.append(
            {
                "true_category": truth,
                **{
                    predicted: int(confusion[row_index, column_index])
                    for column_index, predicted in enumerate(CATEGORY_KEYS)
                },
            }
        )
    write_csv(
        output_dir / "cross_validation_confusion_matrix.csv",
        confusion_rows,
        ["true_category", *CATEGORY_KEYS],
    )
    write_csv(
        output_dir / "active_learning_queue.csv",
        queue_rows,
        [
            "item_id",
            "source_file_name",
            "current_category",
            "predicted_category",
            "predicted_label",
            "confidence",
            "margin",
            "priority_score",
            "priority_reason",
            "perceptual_group",
        ],
    )

    def heldout_metrics(split: str) -> dict[str, Any]:
        split_mask = (splits == split) & reviewed_mask
        indices = np.flatnonzero(split_mask)
        result: dict[str, Any] = {
            "reviewed_count": int(len(indices)),
            "assigned_page_count": int((splits == split).sum()),
        }
        if not len(indices):
            result.update({"accuracy": None, "macro_f1": None})
            return result
        result.update(
            {
                "accuracy": round(
                    float(
                        accuracy_score(
                            labels[indices],
                            full_predictions[indices],
                        )
                    ),
                    6,
                ),
                "macro_f1": round(
                    float(
                        f1_score(
                            labels[indices],
                            full_predictions[indices],
                            average="macro",
                        )
                    ),
                    6,
                ),
            }
        )
        return result

    report = {
        "status": "success",
        "trained_at": trained_at,
        "active_page_count": len(rows),
        "training_page_count": int(train_mask.sum()),
        "validation_page_count": int(
            (splits == "validation").sum()
        ),
        "test_page_count": int((splits == "test").sum()),
        "feature_dimension": int(matrix.shape[1]),
        "category_count": len(CATEGORY_KEYS),
        "cross_validation_folds": folds,
        "weak_label_metrics": {
            "accuracy": round(
                float(
                    accuracy_score(
                        train_labels,
                        train_predictions,
                    )
                ),
                6,
            ),
            "macro_f1": round(
                float(
                    f1_score(
                        train_labels,
                        train_predictions,
                        labels=list(CATEGORY_KEYS),
                        average="macro",
                    )
                ),
                6,
            ),
        },
        "human_feedback_metrics": {
            "reviewed_count": int(
                (reviewed_mask & train_mask).sum()
            ),
            "all_reviewed_count": int(reviewed_mask.sum()),
            "review_weight": review_weight,
            "original_label_accuracy": (
                round(
                    float(
                        accuracy_score(
                            reviewed_labels,
                            reviewed_original,
                        )
                    ),
                    6,
                )
                if len(train_reviewed_local_indices)
                else None
            ),
            "out_of_fold_model_accuracy": (
                round(
                    float(
                        accuracy_score(
                            reviewed_labels,
                            reviewed_predictions,
                        )
                    ),
                    6,
                )
                if len(train_reviewed_local_indices)
                else None
            ),
            "reviewed_class_counts": dict(
                sorted(Counter(reviewed_labels).items())
            ),
        },
        "assistant_feedback_metrics": {
            "active_training_proposal_count": int(
                assistant_mask.sum()
            ),
            "assistant_weight": assistant_weight,
            "human_review_precedence": True,
            "proposal_status": "pending_user_confirmation",
        },
        "heldout_metrics": {
            "validation": heldout_metrics("validation"),
            "test": heldout_metrics("test"),
            "note": (
                "验证集和测试集页面及其人工标签均未参与模型拟合。"
            ),
        },
        "label_provenance": {
            "human_review_overrides": int(
                (reviewed_mask & train_mask).sum()
            ),
            "assistant_proposal_overrides": int(
                (assistant_mask & train_mask).sum()
            ),
            "remaining_weak_source_labels": int(
                train_mask.sum()
                - (reviewed_mask & train_mask).sum()
                - (assistant_mask & train_mask).sum()
            ),
            "warning": (
                "训练集交叉验证主要衡量弱标签拟合；"
                "独立结论应读取 heldout_metrics.test。"
            ),
        },
        "active_learning_queue_size": len(queue_rows),
        "queue_disagreement_count": sum(
            row["priority_reason"] == "模型与当前类别冲突"
            for row in queue_rows
        ),
        "artifacts": {
            "model": str(model_path.relative_to(PROJECT_ROOT)),
            "queue": str(
                (
                    output_dir / "active_learning_queue.csv"
                ).relative_to(PROJECT_ROOT)
            ),
            "confusion_matrix": str(
                (
                    output_dir
                    / "cross_validation_confusion_matrix.csv"
                ).relative_to(PROJECT_ROOT)
            ),
        },
    }
    (output_dir / "feedback_training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    args = parse_args()
    library_dir = (
        args.library_dir
        if args.library_dir.is_absolute()
        else PROJECT_ROOT / args.library_dir
    ).resolve()
    reviews_path = (
        args.reviews
        if args.reviews.is_absolute()
        else PROJECT_ROOT / args.reviews
    ).resolve()
    review_splits_path = (
        args.review_splits
        if args.review_splits.is_absolute()
        else PROJECT_ROOT / args.review_splits
    ).resolve()
    assistant_proposals_path = (
        (
            args.assistant_proposals
            if args.assistant_proposals.is_absolute()
            else PROJECT_ROOT / args.assistant_proposals
        ).resolve()
        if args.assistant_proposals is not None
        else None
    )
    output_dir = (
        (
            args.output_dir
            if args.output_dir.is_absolute()
            else PROJECT_ROOT / args.output_dir
        ).resolve()
        if args.output_dir is not None
        else None
    )
    report = train_feedback_model(
        library_dir=library_dir,
        reviews_path=reviews_path,
        review_splits_path=review_splits_path,
        assistant_proposals_path=assistant_proposals_path,
        folds=args.folds,
        review_weight=args.review_weight,
        assistant_weight=args.assistant_weight,
        queue_size=args.queue_size,
        output_dir=output_dir,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
