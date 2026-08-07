from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path

from scripts.build_text_corpus import make_chunks


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DataContractTests(unittest.TestCase):
    def test_manifest_has_unique_reviewed_items(self) -> None:
        manifest_path = PROJECT_ROOT / "data/manifest/pilot_manifest.jsonl"
        rows = [
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        self.assertEqual(len(rows), 12)
        self.assertEqual(len({row["item_id"] for row in rows}), len(rows))
        self.assertTrue(
            all(row["review_status"] == "已确认" for row in rows)
        )

    def test_expansion_manifest_has_unique_reviewed_items(self) -> None:
        manifest_path = (
            PROJECT_ROOT / "data/manifest/expansion_manifest.jsonl"
        )
        rows = [
            json.loads(line)
            for line in manifest_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]

        self.assertEqual(len(rows), 12)
        self.assertEqual(len({row["item_id"] for row in rows}), len(rows))
        self.assertTrue(
            all(row["review_status"] == "已确认" for row in rows)
        )

    def test_evaluation_answers_exist_in_manifest(self) -> None:
        manifest_path = PROJECT_ROOT / "data/manifest/pilot_manifest.jsonl"
        item_ids = {
            json.loads(line)["item_id"]
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        query_path = PROJECT_ROOT / "data/evaluation/retrieval_queries.csv"
        with query_path.open("r", encoding="utf-8-sig", newline="") as handle:
            queries = list(csv.DictReader(handle))

        self.assertEqual(len(queries), 12)
        self.assertEqual(
            len({query["query_id"] for query in queries}), len(queries)
        )
        self.assertTrue(
            all(query["expected_item_id"] in item_ids for query in queries)
        )

    def test_dataset_v1_is_fully_reviewed_and_aligned(self) -> None:
        manifest_path = (
            PROJECT_ROOT / "data/manifest/dataset_v1_manifest.jsonl"
        )
        manifest_rows = [
            json.loads(line)
            for line in manifest_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        query_path = (
            PROJECT_ROOT / "data/evaluation/dataset_v1_queries.csv"
        )
        with query_path.open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            query_rows = list(csv.DictReader(handle))

        item_ids = {row["item_id"] for row in manifest_rows}
        self.assertEqual(len(manifest_rows), 24)
        self.assertEqual(len(query_rows), 24)
        self.assertEqual(len(item_ids), 24)
        self.assertTrue(
            all(row["review_status"] == "已确认" for row in manifest_rows)
        )
        self.assertTrue(
            all(row["review_status"] == "已确认" for row in query_rows)
        )
        self.assertTrue(
            all(row["expected_item_id"] in item_ids for row in query_rows)
        )

    def test_chunking_preserves_order_and_overlap(self) -> None:
        boxes = [
            {"box_index": index, "text": text, "confidence": 0.9}
            for index, text in enumerate(["甲甲", "乙乙", "丙丙", "丁丁"])
        ]
        chunks = make_chunks(boxes, chunk_chars=5, overlap_boxes=1)

        self.assertEqual(
            [[box["box_index"] for box in chunk] for chunk in chunks],
            [[0, 1], [1, 2], [2, 3]],
        )


if __name__ == "__main__":
    unittest.main()
