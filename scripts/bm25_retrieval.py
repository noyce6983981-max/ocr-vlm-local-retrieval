"""Dependency-free BM25 helpers for mixed Chinese and Latin OCR text."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any


TOKEN_PATTERN = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff]+|[a-zA-Z0-9]+(?:[._:/+-][a-zA-Z0-9]+)*"
)


def tokenize_mixed(text: str) -> list[str]:
    """Tokenize Latin words and emit CJK unigrams plus character bigrams."""
    tokens: list[str] = []
    for match in TOKEN_PATTERN.finditer(text.lower()):
        value = match.group(0)
        if "\u3400" <= value[0] <= "\u9fff":
            characters = list(value)
            tokens.extend(characters)
            tokens.extend(
                characters[index] + characters[index + 1]
                for index in range(len(characters) - 1)
            )
        else:
            tokens.append(value)
    return tokens


def bm25_query_weight(query: str) -> float:
    """Give short visual-style queries less sparse-retrieval influence."""
    compact = "".join(
        match.group(0) for match in TOKEN_PATTERN.finditer(query)
    )
    length = len(compact)
    if length <= 2:
        return 0.05
    if length <= 6:
        return 0.15
    return 0.3


def build_bm25_payload(
    texts: list[str],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> dict[str, Any]:
    if not texts:
        raise ValueError("BM25 corpus cannot be empty.")
    if k1 <= 0:
        raise ValueError("k1 must be positive.")
    if not 0 <= b <= 1:
        raise ValueError("b must be between 0 and 1.")

    postings: dict[str, list[list[int]]] = defaultdict(list)
    document_lengths: list[int] = []
    for document_index, text in enumerate(texts):
        frequencies = Counter(tokenize_mixed(text))
        document_lengths.append(sum(frequencies.values()))
        for token, frequency in frequencies.items():
            postings[token].append([document_index, int(frequency)])

    return {
        "version": 1,
        "algorithm": "BM25Okapi",
        "k1": float(k1),
        "b": float(b),
        "document_count": len(texts),
        "average_document_length": (
            sum(document_lengths) / len(document_lengths)
        ),
        "document_lengths": document_lengths,
        "postings": dict(sorted(postings.items())),
    }


def score_bm25(query: str, payload: dict[str, Any]) -> list[float]:
    document_count = int(payload["document_count"])
    scores = [0.0] * document_count
    if document_count == 0:
        return scores
    average_length = max(
        float(payload["average_document_length"]), 1e-12
    )
    lengths = payload["document_lengths"]
    postings = payload["postings"]
    k1 = float(payload["k1"])
    b = float(payload["b"])

    query_frequencies = Counter(tokenize_mixed(query))
    for token, query_frequency in query_frequencies.items():
        token_postings = postings.get(token)
        if not token_postings:
            continue
        document_frequency = len(token_postings)
        inverse_document_frequency = math.log(
            1.0
            + (
                document_count - document_frequency + 0.5
            )
            / (document_frequency + 0.5)
        )
        query_weight = 1.0 + math.log(query_frequency)
        for document_index, term_frequency in token_postings:
            denominator = term_frequency + k1 * (
                1.0
                - b
                + b
                * float(lengths[document_index])
                / average_length
            )
            scores[document_index] += (
                inverse_document_frequency
                * term_frequency
                * (k1 + 1.0)
                / denominator
                * query_weight
            )
    return scores
