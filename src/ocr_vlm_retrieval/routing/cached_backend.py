"""Bounded in-memory cache for persistent V19 intent backends."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock

from ocr_vlm_retrieval.routing.llm_router import IntentBackend


@dataclass(frozen=True, slots=True)
class CacheStats:
    hits: int
    misses: int
    entries: int


class CachedIntentBackend:
    """Cache exact normalized queries without changing backend responses."""

    def __init__(self, backend: IntentBackend, *, max_entries: int = 1024) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self._backend = backend
        self._max_entries = max_entries
        self._cache: OrderedDict[str, str] = OrderedDict()
        self._lock = Lock()
        self._hits = 0
        self._misses = 0

    @property
    def name(self) -> str:
        return f"cached:{self._backend.name}"

    @property
    def stats(self) -> CacheStats:
        with self._lock:
            return CacheStats(
                hits=self._hits,
                misses=self._misses,
                entries=len(self._cache),
            )

    def generate_intent_json(self, query: str) -> str:
        normalized = " ".join(query.split())
        if not normalized:
            raise ValueError("query must not be empty")
        with self._lock:
            cached = self._cache.get(normalized)
            if cached is not None:
                self._cache.move_to_end(normalized)
                self._hits += 1
                return cached

        generated = self._backend.generate_intent_json(normalized)
        with self._lock:
            self._misses += 1
            self._cache[normalized] = generated
            self._cache.move_to_end(normalized)
            while len(self._cache) > self._max_entries:
                self._cache.popitem(last=False)
        return generated
