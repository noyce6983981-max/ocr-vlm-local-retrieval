"""Learn whether OCR or vision is preferable, then tune a conservative cap."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_learned_gate import (  # noqa: E402
    FEATURE_NAMES,
    feature_vector,
    normalize_rows,
    retrieval_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path(
            "data/evaluation/"
            "public_dataset_200_query_all_nonempty_pending.csv"
        ),
    )
    parser.add_argument(
        "--quality",
        type=Path,
        default=Path(
            "data/evaluation/public_dataset_200_ocr_quality.csv"
        ),
    )
    parser.add_argument(
        "--text-scores",
        type=Path,
        default=Path(
            "outputs/evaluation/public200_gate/text/"
            "text_score_matrix.npz"
        ),
    )
    parser.add_argument(
        "--visual-scores",
        type=Path,
        default=Path(
            "outputs/evaluation/public200_gate/visual/"
            "visual_score_matrix.npz"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/evaluation/"
            "public_dataset_200_preference_gate_diagnostic.json"
        ),
    )
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260729)
    return parser.parse_args()


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def expected_ranks(
    scores: np.ndarray, target_columns: np.ndarray
) -> np.ndarray:
    order = np.argsort(-scores, axis=1)
    return np.asarray(
        [
            int(np.where(row == target)[0][0]) + 1
            for row, target in zip(order, target_columns)
        ],
        dtype=np.int32,
    )


def modality_preference_labels(
    text_scores: np.ndarray,
    visual_scores: np.ndarray,
    target_columns: np.ndarray,
) -> np.ndarray:
    text_ranks = expected_ranks(text_scores, target_columns)
    visual_ranks = expected_ranks(visual_scores, target_columns)
    return np.where(
        text_ranks < visual_ranks,
        1,
        np.where(visual_ranks < text_ranks, 0, -1),
    ).astype(np.int64)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    with project_path(args.queries).open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        queries = list(csv.DictReader(handle))
    with project_path(args.quality).open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        quality_rows = list(csv.DictReader(handle))
    quality_by_id = {row["item_id"]: row for row in quality_rows}

    text_archive = np.load(project_path(args.text_scores))
    visual_archive = np.load(project_path(args.visual_scores))
    query_ids = text_archive["query_ids"].tolist()
    item_ids = text_archive["item_ids"].tolist()
    if query_ids != visual_archive["query_ids"].tolist():
        raise ValueError("Text and visual query orders differ.")
    if item_ids != visual_archive["item_ids"].tolist():
        raise ValueError("Text and visual item orders differ.")
    query_by_id = {row["query_id"]: row for row in queries}
    item_column = {item_id: index for index, item_id in enumerate(item_ids)}
    targets = np.asarray(
        [
            item_column[query_by_id[query_id]["expected_item_id"]]
            for query_id in query_ids
        ],
        dtype=np.int64,
    )
    text = normalize_rows(
        np.asarray(text_archive["scores"], dtype=np.float32)
    )
    visual = normalize_rows(
        np.asarray(visual_archive["scores"], dtype=np.float32)
    )
    labels = modality_preference_labels(text, visual, targets)
    split_indices = {
        split: np.asarray(
            [
                index
                for index, query_id in enumerate(query_ids)
                if query_by_id[query_id]["split"] == split
            ],
            dtype=np.int64,
        )
        for split in ("train", "validation", "test")
    }

    features = np.asarray(
        [feature_vector(quality_by_id[item_id]) for item_id in item_ids],
        dtype=np.float32,
    )
    train_item_mask = np.asarray(
        [quality_by_id[item_id]["split"] == "train" for item_id in item_ids]
    )
    feature_mean = features[train_item_mask].mean(axis=0)
    feature_std = np.maximum(
        features[train_item_mask].std(axis=0), 1e-6
    )
    standardized = (features - feature_mean) / feature_std
    query_feature_rows = standardized[targets]

    labeled_split_indices = {
        split: indices[labels[indices] >= 0]
        for split, indices in split_indices.items()
    }
    train_indices = labeled_split_indices["train"]
    validation_indices = labeled_split_indices["validation"]
    if len(train_indices) < 10 or len(validation_indices) < 2:
        raise ValueError("Insufficient non-tied modality labels.")
    train_labels = labels[train_indices]
    positive_count = int(np.sum(train_labels == 1))
    negative_count = int(np.sum(train_labels == 0))
    pos_weight = negative_count / max(positive_count, 1)

    model = nn.Linear(len(FEATURE_NAMES), 1)
    nn.init.zeros_(model.weight)
    nn.init.zeros_(model.bias)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-3
    )
    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, dtype=torch.float32)
    )
    feature_tensor = torch.from_numpy(query_feature_rows)
    label_tensor = torch.from_numpy(labels.astype(np.float32))
    train_tensor = torch.from_numpy(train_indices).long()
    validation_tensor = torch.from_numpy(validation_indices).long()
    best_state: dict[str, torch.Tensor] | None = None
    best_validation_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(feature_tensor).squeeze(1)
        loss = loss_function(
            logits[train_tensor], label_tensor[train_tensor]
        )
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_logits = model(feature_tensor).squeeze(1)
            validation_loss = loss_function(
                validation_logits[validation_tensor],
                label_tensor[validation_tensor],
            ).item()
        if validation_loss < best_validation_loss - 1e-6:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("Preference gate did not produce a valid state.")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        page_probabilities = (
            torch.sigmoid(
                model(torch.from_numpy(standardized))
            )
            .squeeze(1)
            .numpy()
        )

    validation_query_indices = split_indices["validation"]
    cap_candidates = np.linspace(0.2, 1.0, 17)
    cap_metrics = []
    for cap in cap_candidates:
        weights = page_probabilities * cap
        fused = (
            text * weights[None, :]
            + visual * (1.0 - weights[None, :])
        )
        metrics = retrieval_metrics(
            fused[validation_query_indices],
            targets[validation_query_indices],
        )
        cap_metrics.append((float(cap), metrics))
    best_cap, _ = max(
        cap_metrics,
        key=lambda row: (
            row[1]["mrr"],
            row[1]["recall_at_1"],
            -row[0],
        ),
    )
    learned_weights = page_probabilities * best_cap
    heuristic_weights = np.clip(
        0.6
        * np.asarray(
            [
                float(quality_by_id[item_id]["mean_confidence"])
                for item_id in item_ids
            ],
            dtype=np.float32,
        ),
        0.0,
        1.0,
    )
    methods = {
        "text": text,
        "visual": visual,
        "fixed_0_5": 0.5 * text + 0.5 * visual,
        "heuristic_gate": (
            text * heuristic_weights[None, :]
            + visual * (1.0 - heuristic_weights[None, :])
        ),
        "preference_gate": (
            text * learned_weights[None, :]
            + visual * (1.0 - learned_weights[None, :])
        ),
    }
    metrics = {
        split: {
            name: retrieval_metrics(scores[indices], targets[indices])
            for name, scores in methods.items()
        }
        for split, indices in split_indices.items()
    }
    category_weights: dict[str, list[float]] = {}
    for item_id, weight in zip(item_ids, learned_weights):
        category_weights.setdefault(
            quality_by_id[item_id]["category"], []
        ).append(float(weight))
    label_stats = {
        split: dict(Counter(labels[indices].tolist()))
        for split, indices in split_indices.items()
    }
    report: dict[str, Any] = {
        "status": "diagnostic_unreviewed_queries",
        "warning": (
            "The modality labels come from automatically generated "
            "queries; formal claims require human-reviewed queries."
        ),
        "training": {
            "objective": "predict whether text rank beats visual rank",
            "best_epoch": best_epoch,
            "best_validation_loss": round(best_validation_loss, 6),
            "positive_text_better": positive_count,
            "negative_visual_better": negative_count,
            "positive_class_weight": round(pos_weight, 6),
            "feature_names": list(FEATURE_NAMES),
            "coefficients_standardized": {
                name: round(float(value), 6)
                for name, value in zip(
                    FEATURE_NAMES,
                    model.weight.detach().numpy()[0],
                )
            },
            "bias": round(float(model.bias.detach().item()), 6),
            "selected_text_weight_cap": round(best_cap, 6),
            "validation_cap_search": [
                {"cap": round(cap, 3), **values}
                for cap, values in cap_metrics
            ],
        },
        "modality_label_counts": label_stats,
        "weight_summary": {
            "minimum": round(float(learned_weights.min()), 6),
            "maximum": round(float(learned_weights.max()), 6),
            "mean": round(float(learned_weights.mean()), 6),
            "by_category": {
                category: round(float(np.mean(weights)), 6)
                for category, weights in sorted(category_weights.items())
            },
        },
        "metrics": metrics,
        "learned_text_weights": {
            item_id: round(float(weight), 6)
            for item_id, weight in zip(item_ids, learned_weights)
        },
    }
    output_path = project_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "best_epoch": best_epoch,
                "selected_cap": best_cap,
                "validation": metrics["validation"],
                "test": metrics["test"],
                "weight_by_category": report["weight_summary"][
                    "by_category"
                ],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
