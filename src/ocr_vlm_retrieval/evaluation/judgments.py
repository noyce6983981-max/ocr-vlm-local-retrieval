"""Task-specific judgment schemas for the V17 study."""

from __future__ import annotations

import string
from collections.abc import Mapping
from typing import Any

POOLED_RELEVANCE_TASK = "pooled_relevance"
CORPUS_ANSWERABILITY_TASK = "corpus_answerability"

RELEVANT_CANDIDATE_IN_POOL = "relevant_candidate_in_pool"
NO_RELEVANT_CANDIDATE_IN_POOL = "no_relevant_candidate_in_pool"
VALID_POOL_RELEVANCE = {
    RELEVANT_CANDIDATE_IN_POOL,
    NO_RELEVANT_CANDIDATE_IN_POOL,
    "excluded",
    "uncertain",
}

CORPUS_ANSWERABLE = "corpus_answerable"
CORPUS_NO_ANSWER_AFTER_HIGH_RECALL_POOLING = (
    "corpus_no_answer_after_high_recall_pooling"
)
VALID_CORPUS_ANSWERABILITY = {
    CORPUS_ANSWERABLE,
    CORPUS_NO_ANSWER_AFTER_HIGH_RECALL_POOLING,
    "excluded",
    "uncertain",
}

LEGACY_POOL_RELEVANCE = {
    "answerable": RELEVANT_CANDIDATE_IN_POOL,
    "no_answer": NO_RELEVANT_CANDIDATE_IN_POOL,
    "excluded": "excluded",
    "uncertain": "uncertain",
}


def pool_relevance_decision(row: Mapping[str, Any]) -> str:
    """Read the canonical pool decision while accepting audited legacy rows."""

    task_id = str(row.get("task_id", POOLED_RELEVANCE_TASK)).strip()
    if task_id != POOLED_RELEVANCE_TASK:
        raise ValueError(f"Expected task_id={POOLED_RELEVANCE_TASK!r}, got {task_id!r}")
    decision = str(row.get("pool_relevance", "")).strip()
    if not decision:
        legacy = str(row.get("answerability", "")).strip()
        decision = LEGACY_POOL_RELEVANCE.get(legacy, "")
    if decision not in VALID_POOL_RELEVANCE:
        raise ValueError(f"Invalid pooled-relevance decision: {decision!r}")
    return decision


def normalize_pool_judgment(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return a canonical pooled-relevance row without changing the source."""

    relevance_source = row.get("candidate_relevance", {})
    if not isinstance(relevance_source, Mapping):
        raise ValueError("candidate_relevance must be an object")
    relevance = {
        str(candidate_id).strip(): bool(value)
        for candidate_id, value in relevance_source.items()
        if str(candidate_id).strip()
    }
    decision = pool_relevance_decision(row)
    any_relevant = any(relevance.values())
    if decision == RELEVANT_CANDIDATE_IN_POOL and not any_relevant:
        raise ValueError("A relevant-in-pool decision needs a relevant candidate")
    if decision == NO_RELEVANT_CANDIDATE_IN_POOL and any_relevant:
        raise ValueError("A no-relevant-in-pool decision cannot mark relevance")
    return {
        **dict(row),
        "task_id": POOLED_RELEVANCE_TASK,
        "pool_relevance": decision,
        "candidate_relevance": relevance,
    }


def validate_corpus_no_answer_record(row: Mapping[str, Any]) -> None:
    """Require audit evidence before a corpus-level no-answer label is valid."""

    task_id = str(row.get("task_id", "")).strip()
    if task_id != CORPUS_ANSWERABILITY_TASK:
        raise ValueError(f"Expected task_id={CORPUS_ANSWERABILITY_TASK!r}")
    decision = str(row.get("corpus_answerability", "")).strip()
    if decision not in VALID_CORPUS_ANSWERABILITY:
        raise ValueError(f"Invalid corpus-answerability decision: {decision!r}")
    if decision != CORPUS_NO_ANSWER_AFTER_HIGH_RECALL_POOLING:
        return

    pool_sha256 = str(row.get("pool_sha256", "")).strip().lower()
    if len(pool_sha256) != 64 or any(
        character not in string.hexdigits.lower() for character in pool_sha256
    ):
        raise ValueError("A corpus no-answer record needs a valid pool_sha256")
    retrieval_runs = row.get("retrieval_runs")
    if not isinstance(retrieval_runs, list) or not retrieval_runs:
        raise ValueError("A corpus no-answer record needs retrieval_runs")
    candidate_count = row.get("candidate_count")
    if not isinstance(candidate_count, int) or candidate_count < 1:
        raise ValueError("A corpus no-answer record needs candidate_count")
    reviewer_count = row.get("reviewer_count")
    if not isinstance(reviewer_count, int) or reviewer_count < 2:
        raise ValueError("Corpus no-answer requires two independent reviewers")
    if row.get("adjudicated") is not True:
        raise ValueError("Corpus no-answer requires completed adjudication")
    if row.get("known_relevant_outside_pool") is not False:
        raise ValueError(
            "Corpus no-answer requires known_relevant_outside_pool=false"
        )
