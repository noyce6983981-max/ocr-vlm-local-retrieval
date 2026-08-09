from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts/evaluate_v18_locked_holdout.py"
    )
    spec = importlib.util.spec_from_file_location("evaluate_v18_locked_holdout", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _outcomes(correct: tuple[bool, bool]) -> list[dict[str, object]]:
    return [
        {
            "query_id": f"q{index}",
            "group_id": "g1",
            "query_role": role,
            "answerable_in_pool": role == "positive",
            "correct": value,
            "false_accept": role != "positive" and not value,
            "false_reject": role == "positive" and not value,
            "selected_relevant": role == "positive" and value,
            "selected_rank": 1 if value else None,
        }
        for index, (role, value) in enumerate(
            zip(("positive", "single_condition_hard_negative"), correct, strict=True)
        )
    ]


def test_paired_bootstrap_is_deterministic() -> None:
    module = _load_module()
    baseline = _outcomes((True, False))
    selected = _outcomes((True, True))
    first = module.paired_group_bootstrap(
        baseline, selected, repetitions=20, seed=17
    )
    second = module.paired_group_bootstrap(
        baseline, selected, repetitions=20, seed=17
    )
    assert first == second
    assert first["grouped_pair_accuracy"]["ci95"] == [1.0, 1.0]


def test_verification_index_materializes_l1_evidence() -> None:
    module = _load_module()
    indexed = module.verification_index(
        {
            "status": "complete",
            "scope": "v18_holdout_locked_top_k_candidates_only",
            "judgments_read": False,
            "results": [
                {
                    "query_id": "q1",
                    "candidates": [
                        {
                            "item_id": "item-1",
                            "retrieval_rank": 1,
                            "ranking_score": 0.4,
                            "full_query_score": 0.7,
                        }
                    ],
                }
            ],
        }
    )
    evidence = indexed["q1"][0]
    assert evidence.full_query_score == 0.7
    assert evidence.resolved_requirement_scores == {}
    assert evidence.contrastive_relation_margins == {}
