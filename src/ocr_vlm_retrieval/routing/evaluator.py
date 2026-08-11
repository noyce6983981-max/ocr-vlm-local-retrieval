"""Dependency-free metrics for the V19 intent-routing study."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Final

from ocr_vlm_retrieval.routing.schema import Route

ROUTE_ORDER: Final[tuple[Route, ...]] = (
    "text_evidence",
    "visual_discovery",
    "visual_metadata",
    "entity_exact",
    "topic_discovery",
    "mixed",
)


@dataclass(frozen=True, slots=True)
class RoutingSample:
    gold_route: Route
    predicted_route: Route | None
    rule_route: Route | None
    llm_invoked: bool
    used_fallback: bool = False
    fallback_error_type: str | None = None


@dataclass(frozen=True, slots=True)
class RouteMetrics:
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True, slots=True)
class RoutingMetrics:
    query_count: int
    accuracy: float
    macro_f1: float
    llm_call_rate: float
    fallback_rate: float
    invalid_json_rate_per_llm_call: float
    disagreement_rate_vs_rule: float
    per_route: dict[Route, RouteMetrics]

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)


def _safe_divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate_routing(samples: list[RoutingSample]) -> RoutingMetrics:
    """Compute macro metrics while counting missing predictions as errors."""

    if not samples:
        raise ValueError("samples must not be empty")

    per_route: dict[Route, RouteMetrics] = {}
    for route in ROUTE_ORDER:
        true_positive = sum(
            sample.gold_route == route and sample.predicted_route == route
            for sample in samples
        )
        predicted_positive = sum(
            sample.predicted_route == route for sample in samples
        )
        support = sum(sample.gold_route == route for sample in samples)
        precision = _safe_divide(true_positive, predicted_positive)
        recall = _safe_divide(true_positive, support)
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_route[route] = RouteMetrics(
            precision=precision,
            recall=recall,
            f1=f1,
            support=support,
        )

    query_count = len(samples)
    llm_calls = sum(sample.llm_invoked for sample in samples)
    comparable_to_rule = [
        sample
        for sample in samples
        if sample.rule_route is not None and sample.predicted_route is not None
    ]
    disagreements = sum(
        sample.predicted_route != sample.rule_route
        for sample in comparable_to_rule
    )
    invalid_json = sum(
        sample.fallback_error_type == "IntentSchemaError" for sample in samples
    )
    return RoutingMetrics(
        query_count=query_count,
        accuracy=_safe_divide(
            sum(
                sample.predicted_route == sample.gold_route
                for sample in samples
            ),
            query_count,
        ),
        macro_f1=sum(metric.f1 for metric in per_route.values())
        / len(ROUTE_ORDER),
        llm_call_rate=_safe_divide(llm_calls, query_count),
        fallback_rate=_safe_divide(
            sum(sample.used_fallback for sample in samples), query_count
        ),
        invalid_json_rate_per_llm_call=_safe_divide(invalid_json, llm_calls),
        disagreement_rate_vs_rule=_safe_divide(
            disagreements, len(comparable_to_rule)
        ),
        per_route=per_route,
    )
