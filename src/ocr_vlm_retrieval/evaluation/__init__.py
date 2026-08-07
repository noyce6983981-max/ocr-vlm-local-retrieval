"""Evaluation primitives for reproducible retrieval studies."""

from .human_evaluation import (
    agreement_report,
    blind_candidate_pool,
    grouped_paired_bootstrap,
    pool_ranked_runs,
    validate_annotation_coverage,
)

__all__ = [
    "agreement_report",
    "blind_candidate_pool",
    "grouped_paired_bootstrap",
    "pool_ranked_runs",
    "validate_annotation_coverage",
]
