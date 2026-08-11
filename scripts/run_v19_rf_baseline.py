from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from ocr_vlm_retrieval.routing.evaluator import (  # noqa: E402
    RoutingSample,
    evaluate_routing,
)
from ocr_vlm_retrieval.routing.rule_router import RuleRouter  # noqa: E402
from ocr_vlm_retrieval.routing.schema import validate_route  # noqa: E402
from ocr_vlm_retrieval.runtime.cache import write_json_atomic  # noqa: E402
from scripts.run_intent_routing_study import read_queries  # noqa: E402

DEFAULT_QUERIES = ROOT / "data/evaluation/v19/pilot/queries_reviewed.csv"
DEFAULT_OUTPUT = ROOT / "outputs/evaluation/v19/pilot/rf_diagnostic.json"
RANDOM_STATE = 20260811


def build_classifier() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(2, 5),
                    max_features=2000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=500,
                    class_weight="balanced_subsample",
                    max_features="sqrt",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def run_rf_diagnostic(rows: list[dict[str, str]]) -> dict[str, Any]:
    queries = [row["query_text"] for row in rows]
    labels = [validate_route(row["gold_route"]) for row in rows]
    folds = StratifiedKFold(
        n_splits=6,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    predicted: list[str | None] = [None] * len(rows)
    fold_by_index: list[int | None] = [None] * len(rows)
    fit_predict_ms: list[float] = []
    for fold_index, (train_indices, test_indices) in enumerate(
        folds.split(queries, labels), start=1
    ):
        classifier = build_classifier()
        started = time.perf_counter()
        classifier.fit(
            [queries[index] for index in train_indices],
            [labels[index] for index in train_indices],
        )
        fold_predictions = classifier.predict(
            [queries[index] for index in test_indices]
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        fit_predict_ms.append(elapsed_ms)
        for index, route in zip(test_indices, fold_predictions, strict=True):
            predicted[index] = validate_route(str(route))
            fold_by_index[index] = fold_index

    if any(route is None for route in predicted):
        raise RuntimeError("out-of-fold prediction coverage is incomplete")

    rule_router = RuleRouter.legacy()
    samples: list[RoutingSample] = []
    prediction_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        gold_route = labels[index]
        predicted_route = validate_route(str(predicted[index]))
        rule_route = rule_router.route(row["query_text"]).route
        samples.append(
            RoutingSample(
                gold_route=gold_route,
                predicted_route=predicted_route,
                rule_route=rule_route,
                llm_invoked=False,
            )
        )
        prediction_rows.append(
            {
                "query_id": row["query_id"],
                "family_id": row["family_id"],
                "query_text": row["query_text"],
                "gold_route": gold_route,
                "predicted_route": predicted_route,
                "correct": predicted_route == gold_route,
                "fold": fold_by_index[index],
            }
        )
    metrics = evaluate_routing(samples)
    return {
        "study_id": "v19-local-llm-structured-intent-routing",
        "split": "human_reviewed_pilot_not_final",
        "method": "RF_character_tfidf_random_forest",
        "diagnostic_only": True,
        "evaluation": "six_fold_stratified_out_of_fold",
        "split_unit": "query_no_paraphrase_families_available",
        "random_state": RANDOM_STATE,
        "query_count": len(rows),
        "gold_route_counts": dict(Counter(labels)),
        "metrics": metrics.to_mapping(),
        "operational": {
            "total_fit_predict_ms": sum(fit_predict_ms),
            "mean_fold_fit_predict_ms": sum(fit_predict_ms) / len(fit_predict_ms),
        },
        "predictions": prediction_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = run_rf_diagnostic(read_queries(args.queries))
    write_json_atomic(args.output, payload)
    metrics = payload["metrics"]
    print(
        "V19 RF diagnostic: "
        f"n={payload['query_count']}, accuracy={metrics['accuracy']:.4f}, "
        f"macro_f1={metrics['macro_f1']:.4f}"
    )
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
