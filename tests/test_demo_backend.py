from __future__ import annotations

import unittest

import numpy as np

from scripts.demo_backend import (
    build_method_scores,
    build_rrf_scores,
    normalize_rows,
    retrieval_metrics,
)


class DemoBackendTests(unittest.TestCase):
    def test_quality_rrf_uses_visual_for_low_confidence_page(self) -> None:
        dense = np.array([0.9, 0.8], dtype=np.float32)
        bm25 = np.array([3.0, 0.0], dtype=np.float32)
        visual = np.array([0.1, 0.9], dtype=np.float32)
        confidence = np.array([1.0, 0.0], dtype=np.float32)

        methods, branches = build_rrf_scores(
            dense,
            bm25,
            visual,
            confidence,
            k=1,
            top_n=2,
        )

        self.assertGreater(methods["quality_rrf"][1], 0.0)
        self.assertEqual(branches["bm25_rrf"][1], 0.0)

    def test_normalize_rows_handles_constant_scores(self) -> None:
        matrix = np.array([[2.0, 2.0], [1.0, 3.0]], dtype=np.float32)
        normalized = normalize_rows(matrix)
        np.testing.assert_allclose(normalized[0], [0.0, 0.0])
        np.testing.assert_allclose(normalized[1], [0.0, 1.0])

    def test_no_text_candidate_uses_visual_score(self) -> None:
        text = np.array([[0.8, 0.2]], dtype=np.float32)
        visual = np.array([[0.1, 0.9]], dtype=np.float32)
        confidences = np.array([1.0, 0.0], dtype=np.float32)
        scores, weights = build_method_scores(text, visual, confidences)
        self.assertEqual(float(weights[1]), 0.0)
        self.assertAlmostEqual(
            float(scores["adaptive"][0, 1]),
            float(scores["visual"][0, 1]),
        )

    def test_retrieval_metrics(self) -> None:
        scores = np.array(
            [[0.9, 0.1, 0.0], [0.5, 0.6, 0.55]], dtype=np.float32
        )
        metrics = retrieval_metrics(
            scores,
            expected_item_ids=["a", "c"],
            item_ids=["a", "b", "c"],
        )
        self.assertEqual(metrics["recall_at_1"], 0.5)
        self.assertEqual(metrics["recall_at_3"], 1.0)
        self.assertAlmostEqual(metrics["mrr"], 0.75)


if __name__ == "__main__":
    unittest.main()
