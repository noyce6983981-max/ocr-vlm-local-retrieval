from __future__ import annotations

import numpy as np
import pytest

from ocr_vlm_retrieval.runtime.late_interaction import (
    atomic_save_embedding,
    exclusive_process_lock,
    load_embedding,
    positive_retrieval_summary,
    union_recall_summary,
)
from scripts.evaluate_v19_colqwen2_development import maxsim_score_matrix


def test_embedding_shard_round_trip_and_validation(tmp_path) -> None:
    path = tmp_path / "page.npy"
    atomic_save_embedding(path, np.asarray([[1.0, 2.0]], dtype=np.float32))
    loaded = load_embedding(path, expected_dim=2)
    assert loaded.dtype == np.float16
    assert loaded.tolist() == [[1.0, 2.0]]
    with pytest.raises(ValueError, match="dimension mismatch"):
        load_embedding(path, expected_dim=3)


def test_gpu_job_lock_rejects_concurrent_owner_and_recovers_stale_lock(
    tmp_path,
) -> None:
    path = tmp_path / "job.lock"
    with exclusive_process_lock(path):
        assert path.is_file()
        with (
            pytest.raises(RuntimeError, match="already owns"),
            exclusive_process_lock(path),
        ):
            pass
    assert not path.exists()
    path.write_text("99999999", encoding="ascii")
    with exclusive_process_lock(path):
        assert path.is_file()
    assert not path.exists()


def test_positive_summary_is_explicitly_closed_set() -> None:
    rows = [
        {
            "gold_answerable": True,
            "gold_relevant_item_ids": ["gold"],
            "ranking_item_ids": ["wrong", "gold"],
        },
        {
            "gold_answerable": False,
            "gold_relevant_item_ids": [],
            "ranking_item_ids": ["wrong"],
        },
    ]
    summary = positive_retrieval_summary(rows, cutoffs=(1, 3))
    assert summary["metric_scope"] == "closed_set_positive_retrieval_only"
    assert summary["positive_count"] == 1
    assert summary["recall_at_1"] == 0.0
    assert summary["recall_at_3"] == 1.0
    assert summary["mrr"] == 0.5


def test_union_summary_measures_branch_complementarity() -> None:
    baseline = [
        {
            "query_id": "q",
            "gold_answerable": True,
            "gold_relevant_item_ids": ["gold"],
            "ranking_item_ids": ["base"],
        }
    ]
    candidate = [{"query_id": "q", "ranking_item_ids": ["gold"]}]
    summary = union_recall_summary(baseline, candidate, cutoff=1)
    assert summary["union_recall"] == 1.0


def test_masked_maxsim_does_not_reward_padded_passage_tokens() -> None:
    torch = pytest.importorskip("torch")
    queries = [torch.tensor([[1.0, 0.0]])]
    passages = [
        torch.tensor([[-1.0, 0.0]]),
        torch.tensor([[-0.5, 0.0], [-0.25, 0.0]]),
    ]
    scores = maxsim_score_matrix(queries, passages)
    assert scores.tolist() == [[-1.0, -0.25]]
