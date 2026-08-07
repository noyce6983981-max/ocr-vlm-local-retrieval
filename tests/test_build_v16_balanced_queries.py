from __future__ import annotations

from collections import Counter

from scripts.build_v16_balanced_queries import (
    NO_ANSWER_SPECS,
    ROUTES,
    TEXT_SPECS,
    TOPIC_SPECS,
    VISUAL_CLASS_SPECS,
)
from scripts.query_routing import infer_retrieval_route


def test_v16_specs_are_route_balanced() -> None:
    for split in ("calibration", "holdout"):
        answerable_counts = Counter(
            {
                "text_evidence": len(TEXT_SPECS[split]),
                "visual_metadata": len(
                    VISUAL_CLASS_SPECS[split]["visual_metadata"]
                ),
                "visual_discovery": len(
                    VISUAL_CLASS_SPECS[split]["visual_discovery"]
                ),
                "mixed": 6,
                "topic_discovery": len(TOPIC_SPECS[split]),
            }
        )
        assert answerable_counts == Counter({route: 6 for route in ROUTES})
        assert Counter(
            {
                route: len(NO_ANSWER_SPECS[split][route])
                for route in ROUTES
            }
        ) == Counter({route: 4 for route in ROUTES})


def test_v16_negative_specs_follow_declared_routes() -> None:
    for split in ("calibration", "holdout"):
        for route in ROUTES:
            for query, _ in NO_ANSWER_SPECS[split][route]:
                assert infer_retrieval_route(query) == route


def test_v16_visual_answerable_specs_follow_declared_routes() -> None:
    for split in ("calibration", "holdout"):
        for route in ("visual_metadata", "visual_discovery"):
            for _, query in VISUAL_CLASS_SPECS[split][route]:
                assert infer_retrieval_route(query) == route
