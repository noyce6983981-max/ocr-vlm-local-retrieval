"""Evaluation primitives for reproducible retrieval studies."""

from .human_evaluation import (
    agreement_report,
    blind_candidate_pool,
    grouped_paired_bootstrap,
    pool_ranked_runs,
    validate_annotation_coverage,
)
from .public_holdout import (
    analyze_public_paired_records,
    build_public_paired_records,
    summarize_verification_runtime,
    validate_public_paired_records,
)

__all__ = [
    "agreement_report",
    "analyze_public_paired_records",
    "blind_candidate_pool",
    "build_public_paired_records",
    "grouped_paired_bootstrap",
    "pool_ranked_runs",
    "summarize_verification_runtime",
    "validate_annotation_coverage",
    "validate_public_paired_records",
]
