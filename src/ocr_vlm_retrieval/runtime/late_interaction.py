"""Utilities for resumable multi-vector page retrieval experiments."""

from __future__ import annotations

import os
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        query_limited_information = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(
            query_limited_information, False, pid
        )
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


@contextmanager
def exclusive_process_lock(path: Path) -> Iterator[None]:
    """Prevent concurrent GPU jobs and recover a lock left by a crashed process."""

    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(2):
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                owner_pid = int(path.read_text(encoding="ascii").strip())
            except (OSError, ValueError):
                owner_pid = -1
            if _pid_is_running(owner_pid):
                raise RuntimeError(
                    f"another process already owns the GPU job lock: pid={owner_pid}"
                ) from None
            if attempt == 0:
                path.unlink(missing_ok=True)
                continue
            raise RuntimeError(
                f"could not recover stale process lock: {path}"
            ) from None
        else:
            try:
                os.write(descriptor, str(os.getpid()).encode("ascii"))
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            break
    else:  # pragma: no cover - defensive guard for future loop changes
        raise RuntimeError(f"could not acquire process lock: {path}")
    try:
        yield
    finally:
        try:
            if int(path.read_text(encoding="ascii").strip()) == os.getpid():
                path.unlink(missing_ok=True)
        except (OSError, ValueError):
            pass


def atomic_save_embedding(path: Path, embedding: np.ndarray[Any, Any]) -> None:
    """Persist one page embedding without exposing a partial shard."""

    array = np.asarray(embedding, dtype=np.float16)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("page embedding must be a non-empty rank-2 array")
    if not np.isfinite(array).all():
        raise ValueError("page embedding contains non-finite values")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    try:
        with temporary.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_embedding(
    path: Path, *, expected_dim: int | None = None
) -> np.ndarray[Any, Any]:
    """Load and validate one untrusted local NumPy embedding shard."""

    with path.open("rb") as handle:
        array = np.load(handle, allow_pickle=False)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"invalid embedding shape in {path}: {array.shape}")
    if expected_dim is not None and int(array.shape[1]) != expected_dim:
        raise ValueError(
            f"embedding dimension mismatch in {path}: "
            f"{array.shape[1]} != {expected_dim}"
        )
    if not np.issubdtype(array.dtype, np.floating):
        raise ValueError(f"embedding must be floating point: {path}")
    if not np.isfinite(array).all():
        raise ValueError(f"embedding contains non-finite values: {path}")
    return np.asarray(array, dtype=np.float16)


def positive_retrieval_summary(
    results: Sequence[Mapping[str, Any]],
    *,
    cutoffs: Sequence[int] = (1, 3, 5, 10, 20, 50),
) -> dict[str, Any]:
    """Summarize closed-set recall without mislabelling it as open-set accuracy."""

    positives = [row for row in results if bool(row.get("gold_answerable"))]
    if not positives:
        raise ValueError("at least one answerable query is required")
    recalls: dict[str, float] = {}
    for cutoff in cutoffs:
        if cutoff <= 0:
            raise ValueError("retrieval cutoffs must be positive")
        hit_count = 0
        for row in positives:
            relevant = {
                str(item_id) for item_id in row.get("gold_relevant_item_ids", [])
            }
            ranking = [str(item_id) for item_id in row.get("ranking_item_ids", [])]
            hit_count += bool(relevant.intersection(ranking[:cutoff]))
        recalls[f"recall_at_{cutoff}"] = hit_count / len(positives)
    reciprocal_ranks: list[float] = []
    for row in positives:
        relevant = {str(item_id) for item_id in row.get("gold_relevant_item_ids", [])}
        ranking = [str(item_id) for item_id in row.get("ranking_item_ids", [])]
        rank = next(
            (
                index
                for index, item_id in enumerate(ranking, start=1)
                if item_id in relevant
            ),
            None,
        )
        reciprocal_ranks.append(1.0 / rank if rank is not None else 0.0)
    return {
        "query_count": len(results),
        "positive_count": len(positives),
        "metric_scope": "closed_set_positive_retrieval_only",
        **recalls,
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks),
    }


def union_recall_summary(
    baseline_results: Sequence[Mapping[str, Any]],
    candidate_results: Sequence[Mapping[str, Any]],
    *,
    cutoff: int,
) -> dict[str, float | int | str]:
    """Measure complementary coverage of two top-k lists on answerable queries."""

    if cutoff <= 0:
        raise ValueError("cutoff must be positive")
    candidate_by_id = {
        str(row["query_id"]): row for row in candidate_results
    }
    positives = [row for row in baseline_results if bool(row.get("gold_answerable"))]
    hits = 0
    for row in positives:
        query_id = str(row["query_id"])
        if query_id not in candidate_by_id:
            raise ValueError(f"candidate results missing query {query_id}")
        relevant = {str(item_id) for item_id in row.get("gold_relevant_item_ids", [])}
        baseline = [str(item_id) for item_id in row.get("ranking_item_ids", [])]
        candidate = [
            str(item_id)
            for item_id in candidate_by_id[query_id].get("ranking_item_ids", [])
        ]
        hits += bool(relevant.intersection(baseline[:cutoff] + candidate[:cutoff]))
    return {
        "metric_scope": "closed_set_positive_union_retrieval_only",
        "positive_count": len(positives),
        "cutoff_per_branch": cutoff,
        "union_recall": hits / len(positives) if positives else 0.0,
    }
