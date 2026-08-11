from __future__ import annotations

from scripts.finalize_v19_holdout_results import (
    macro_f1,
    paired_family_bootstrap,
)


def test_family_bootstrap_preserves_perfect_candidate_gain() -> None:
    baseline = []
    candidate = []
    routes = ("text_evidence", "visual_discovery")
    for family_index in range(30):
        gold = routes[family_index % 2]
        wrong = routes[(family_index + 1) % 2]
        for paraphrase in range(4):
            common = {
                "query_id": f"q{family_index}_{paraphrase}",
                "family_id": f"f{family_index}",
                "gold_route": gold,
            }
            baseline.append({**common, "predicted_route": wrong})
            candidate.append({**common, "predicted_route": gold})
    result = paired_family_bootstrap(
        baseline, candidate, repetitions=100, seed=1
    )
    assert result["candidate_accuracy_95ci"] == {
        "lower_95": 1.0,
        "upper_95": 1.0,
    }
    assert macro_f1(candidate) < 1.0
