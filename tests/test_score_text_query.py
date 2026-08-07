from __future__ import annotations

import unittest

import numpy as np

from scripts.score_text_query import score_text_query_payload


class FakeTextModel:
    def encode(self, queries, **kwargs):
        del queries, kwargs
        return {"dense_vecs": np.array([[1.0, 0.0]], dtype=np.float32)}


class FakeFaissIndex:
    ntotal = 2

    def search(self, query_vector, count):
        self.last_query = query_vector
        self.last_count = count
        return (
            np.array([[0.9, 0.2]], dtype=np.float32),
            np.array([[1, 0]], dtype=np.int64),
        )


class ScoreTextQueryTests(unittest.TestCase):
    def test_reusable_scorer_preserves_manifest_order(self) -> None:
        index = FakeFaissIndex()
        resources = {
            "item_ids": ["a", "b"],
            "item_columns": {"a": 0, "b": 1},
            "index_pairs": [
                (index, [{"item_id": "a"}, {"item_id": "b"}])
            ],
            "metadata_index": None,
            "metadata_rows": [],
        }
        payload = score_text_query_payload(
            "needle",
            FakeTextModel(),
            resources,
            library_revision="revision-1",
        )
        self.assertEqual(payload["item_ids"], ["a", "b"])
        self.assertAlmostEqual(payload["scores"][0], 0.2, places=6)
        self.assertAlmostEqual(payload["scores"][1], 0.9, places=6)
        self.assertEqual(payload["library_revision"], "revision-1")
        self.assertEqual(index.last_count, 2)

    def test_reusable_scorer_rejects_empty_query(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty"):
            score_text_query_payload(
                "  ",
                FakeTextModel(),
                {
                    "item_ids": [],
                    "item_columns": {},
                    "index_pairs": [],
                    "metadata_index": None,
                    "metadata_rows": [],
                },
                library_revision="revision-1",
            )


if __name__ == "__main__":
    unittest.main()
