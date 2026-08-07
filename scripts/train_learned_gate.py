"""Fit a tiny OCR-feature gate on train queries and diagnose held-out splits."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_NAMES = (
    "mean_confidence",
    "p10_confidence",
    "low_confidence_fraction",
    "zero_confidence_fraction",
    "log_character_count",
    "log_text_box_count",
    "log_chars_per_box",
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
            "public_dataset_200_learned_gate_diagnostic.json"
        ),
    )
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260729)
    return parser.parse_args()


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    minimum = matrix.min(axis=1, keepdims=True)
    maximum = matrix.max(axis=1, keepdims=True)
    return (matrix - minimum) / np.maximum(maximum - minimum, 1e-8)


def feature_vector(row: dict[str, str]) -> list[float]:
    return [
        float(row["mean_confidence"]),
        float(row["p10_confidence"]),
        float(row["low_confidence_fraction"]),
        float(row["zero_confidence_fraction"]),
        math.log1p(float(row["character_count"])),
        math.log1p(float(row["text_box_count"])),
        math.log1p(float(row["chars_per_box"])),
    ]


def retrieval_metrics(
    scores: np.ndarray,
    target_columns: np.ndarray,
) -> dict[str, Any]:
    order = np.argsort(-scores, axis=1)
    ranks = np.array(
        [
            int(np.where(row == target)[0][0]) + 1
            for row, target in zip(order, target_columns)
        ],
        dtype=np.int32,
    )
    return {
        "query_count": int(len(ranks)),
        "recall_at_1": round(float(np.mean(ranks == 1)), 6),
        "recall_at_3": round(float(np.mean(ranks <= 3)), 6),
        "mrr": round(float(np.mean(1.0 / ranks)), 6),
        "median_rank": float(np.median(ranks)),
    }


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
    if set(query_ids) != set(query_by_id):
        raise ValueError("Query CSV and score matrices differ.")
    item_column = {item_id: index for index, item_id in enumerate(item_ids)}

    text = normalize_rows(
        np.asarray(text_archive["scores"], dtype=np.float32)
    )
    visual = normalize_rows(
        np.asarray(visual_archive["scores"], dtype=np.float32)
    )
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
    targets = np.asarray(
        [
            item_column[query_by_id[query_id]["expected_item_id"]]
            for query_id in query_ids
        ],
        dtype=np.int64,
    )
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
    if any(len(indices) == 0 for indices in split_indices.values()):
        raise ValueError("Train, validation, and test queries are required.")

    device = torch.device("cpu")
    feature_tensor = torch.from_numpy(standardized).to(device)
    text_tensor = torch.from_numpy(text).to(device)
    visual_tensor = torch.from_numpy(visual).to(device)
    target_tensor = torch.from_numpy(targets).to(device)
    model = nn.Linear(len(FEATURE_NAMES), 1).to(device)
    nn.init.zeros_(model.weight)
    nn.init.zeros_(model.bias)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    train_index = torch.from_numpy(split_indices["train"]).long()
    validation_index = torch.from_numpy(
        split_indices["validation"]
    ).long()
    best_validation_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        weights = torch.sigmoid(model(feature_tensor)).squeeze(1)
        fused = (
            text_tensor * weights.unsqueeze(0)
            + visual_tensor * (1.0 - weights.unsqueeze(0))
        )
        train_loss = nn.functional.cross_entropy(
            fused[train_index] / args.temperature,
            target_tensor[train_index],
        )
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            weights = torch.sigmoid(model(feature_tensor)).squeeze(1)
            fused = (
                text_tensor * weights.unsqueeze(0)
                + visual_tensor * (1.0 - weights.unsqueeze(0))
            )
            validation_loss = nn.functional.cross_entropy(
                fused[validation_index] / args.temperature,
                target_tensor[validation_index],
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
        raise RuntimeError("Learned gate did not produce a valid state.")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        learned_weights = (
            torch.sigmoid(model(feature_tensor))
            .squeeze(1)
            .cpu()
            .numpy()
        )

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
    method_scores = {
        "text": text,
        "visual": visual,
        "fixed_0_5": 0.5 * text + 0.5 * visual,
        "heuristic_gate": (
            text * heuristic_weights[None, :]
            + visual * (1.0 - heuristic_weights[None, :])
        ),
        "learned_gate": (
            text * learned_weights[None, :]
            + visual * (1.0 - learned_weights[None, :])
        ),
    }
    metrics = {}
    for split, indices in split_indices.items():
        metrics[split] = {
            name: retrieval_metrics(
                scores[indices], targets[indices]
            )
            for name, scores in method_scores.items()
        }

    category_weights: dict[str, list[float]] = {}
    for item_id, weight in zip(item_ids, learned_weights):
        category = quality_by_id[item_id]["category"]
        category_weights.setdefault(category, []).append(float(weight))
    report = {
        "status": "diagnostic_unreviewed_queries",
        "warning": (
            "The gate is trained on automatically generated queries. "
            "Held-out metrics are diagnostic until human query review."
        ),
        "data": {
            "query_count": len(query_ids),
            "train_queries": len(split_indices["train"]),
            "validation_queries": len(split_indices["validation"]),
            "test_queries": len(split_indices["test"]),
            "candidate_pages": len(item_ids),
        },
        "training": {
            "seed": args.seed,
            "temperature": args.temperature,
            "learning_rate": args.learning_rate,
            "best_epoch": best_epoch,
            "best_validation_loss": round(best_validation_loss, 6),
            "feature_names": list(FEATURE_NAMES),
            "feature_mean": [
                round(float(value), 6) for value in feature_mean
            ],
            "feature_std": [
                round(float(value), 6) for value in feature_std
            ],
            "coefficients_standardized": {
                name: round(float(value), 6)
                for name, value in zip(
                    FEATURE_NAMES,
                    model.weight.detach().cpu().numpy()[0],
                )
            },
            "bias": round(float(model.bias.detach().cpu().item()), 6),
        },
        "learned_weight_summary": {
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
                "validation": metrics["validation"],
                "test": metrics["test"],
                "weight_by_category": report[
                    "learned_weight_summary"
                ]["by_category"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
