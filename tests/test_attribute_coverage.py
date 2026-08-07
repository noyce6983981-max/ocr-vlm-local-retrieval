from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from ocr_vlm_retrieval.gating.attribute_coverage import (
    aggregate_candidate_evidence,
    decompose_visual_query,
    load_attribute_policy,
)
from scripts.live_search import (
    apply_attribute_coverage_guard,
    cached_visual_query_matches,
    promote_attribute_complete_evidence,
    write_json_atomic,
)

POLICY_PATH = PROJECT_ROOT / "config/v17_attribute_coverage.json"


class AttributeCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = load_attribute_policy(POLICY_PATH)

    def test_dolphin_failure_decomposes_into_mandatory_evidence(self) -> None:
        plan = decompose_visual_query("找一张黄色海豚跃过沙漠金字塔的照片", self.policy)
        values = {(row.kind, row.value) for row in plan.requirements}
        self.assertTrue(plan.compositional)
        self.assertIn(("color", "黄色"), values)
        self.assertIn(("scene", "沙漠"), values)
        self.assertIn(("object", "海豚"), values)
        self.assertIn(("object", "金字塔"), values)
        self.assertIn(("binding", "黄色海豚"), values)
        self.assertIn(("relation", "海豚跃过金字塔"), values)

    def test_high_global_similarity_cannot_hide_missing_dolphin(self) -> None:
        plan = decompose_visual_query("黄色海豚跃过沙漠金字塔", self.policy)
        evidence = {
            row.requirement_id: (
                row.threshold - 0.08
                if row.kind in {"object", "relation", "binding"} and "海豚" in row.value
                else row.full_score
            )
            for row in plan.requirements
        }
        result = aggregate_candidate_evidence(
            plan, evidence, global_score=0.72, policy=self.policy
        )
        self.assertFalse(result["accepted"])
        self.assertLess(result["coverage_ratio"], 1.0)
        self.assertTrue(result["missing_requirement_ids"])

    def test_full_attribute_coverage_accepts_candidate(self) -> None:
        plan = decompose_visual_query("黄色海豚跃过沙漠金字塔", self.policy)
        evidence = {row.requirement_id: row.full_score for row in plan.requirements}
        result = aggregate_candidate_evidence(
            plan, evidence, global_score=0.55, policy=self.policy
        )
        self.assertTrue(result["accepted"])
        self.assertEqual(result["coverage_ratio"], 1.0)
        self.assertEqual(result["missing_requirement_ids"], [])

    def test_color_object_query_preserves_attribute_binding(self) -> None:
        plan = decompose_visual_query("蓝色汽车", self.policy)
        values = {(row.kind, row.value) for row in plan.requirements}
        self.assertIn(("color", "蓝色"), values)
        self.assertIn(("object", "汽车"), values)
        self.assertIn(("binding", "蓝色汽车"), values)

    def test_quoted_text_becomes_independent_ocr_requirement(self) -> None:
        plan = decompose_visual_query("找一张牌子上写着“实验室安全”的照片", self.policy)
        values = {(row.kind, row.value) for row in plan.requirements}
        self.assertIn(("ocr", "实验室安全"), values)

    def test_single_scene_query_does_not_enable_compositional_gate(self) -> None:
        plan = decompose_visual_query("森林", self.policy)
        self.assertFalse(plan.compositional)
        result = aggregate_candidate_evidence(
            plan, {}, global_score=0.51, policy=self.policy
        )
        self.assertFalse(result["enabled"])
        self.assertTrue(result["accepted"])

    def test_english_textvqa_question_decomposes_color_and_relation(self) -> None:
        plan = decompose_visual_query(
            "What greeting is written on the left wall with a red background?",
            self.policy,
        )
        kinds = {row.kind for row in plan.requirements}
        values = {row.value for row in plan.requirements}
        self.assertTrue(plan.compositional)
        self.assertIn("color", kinds)
        self.assertIn("relation", kinds)
        self.assertIn("红色", values)

    def test_english_relation_marker_uses_word_boundaries(self) -> None:
        plan = decompose_visual_query(
            "What's the phone number on the bus?",
            self.policy,
        )
        relation = next(row for row in plan.requirements if row.kind == "relation")
        self.assertIn(" on ", relation.value)
        self.assertTrue(plan.compositional)

    def test_english_answer_slot_is_not_treated_as_an_object(self) -> None:
        plan = decompose_visual_query(
            "What letter is under his cap?",
            self.policy,
        )
        values = {(row.kind, row.value) for row in plan.requirements}
        self.assertNotIn(("object", "what letter is"), values)
        self.assertIn(("object", "visible letter"), values)
        self.assertIn(("object", "his cap"), values)
        self.assertIn(("relation", "visible letter under his cap"), values)

    def test_zone_answer_slot_becomes_a_concrete_zone_requirement(self) -> None:
        plan = decompose_visual_query(
            "What kind of zone is in front of the hitter?",
            self.policy,
        )
        values = {(row.kind, row.value) for row in plan.requirements}
        self.assertNotIn(("object", "what kind of zone is"), values)
        self.assertIn(("object", "zone"), values)
        self.assertIn(("relation", "zone in front of the hitter"), values)

    def test_logo_question_retains_holder_relation(self) -> None:
        plan = decompose_visual_query(
            "What does the logo under the shirt say?",
            self.policy,
        )
        values = {(row.kind, row.value) for row in plan.requirements}
        self.assertIn(("object", "logo text"), values)
        self.assertIn(("object", "the shirt"), values)
        self.assertIn(("relation", "logo text under the shirt"), values)

    def test_word_color_binds_to_word_not_engine(self) -> None:
        plan = decompose_visual_query(
            "What word is written in white on the engine?",
            self.policy,
        )
        values = {(row.kind, row.value) for row in plan.requirements}
        bindings = {value for kind, value in values if kind == "binding"}
        self.assertTrue(any(value.endswith("visible word") for value in bindings))
        self.assertIn(("relation", "visible word on the engine"), values)

    def test_attribute_guard_is_irreversible_and_preserves_reject(self) -> None:
        plan = decompose_visual_query("蓝色汽车", self.policy)
        complete = [
            {
                "item_id": "complete",
                "attribute_coverage_passed": True,
                "attribute_coverage_ratio": 1.0,
                "attribute_missing_requirement_ids": [],
                "attribute_weakest_requirement_id": plan.requirements[0].requirement_id,
            }
        ]
        accepted = {"accepted": True}
        apply_attribute_coverage_guard(plan, complete, accepted, "v17-test")
        self.assertTrue(accepted["accepted"])

        upstream_reject = {"accepted": False}
        apply_attribute_coverage_guard(plan, complete, upstream_reject, "v17-test")
        self.assertFalse(upstream_reject["accepted"])
        self.assertFalse(upstream_reject["upstream_gate_passed"])

    def test_complete_attribute_rows_are_promoted_before_partial_rows(self) -> None:
        rankings = {
            "quality_hybrid": [
                {"item_id": "partial", "attribute_coverage_passed": False},
                {"item_id": "complete", "attribute_coverage_passed": True},
            ]
        }
        promote_attribute_complete_evidence(rankings)
        self.assertEqual(
            [row["item_id"] for row in rankings["quality_hybrid"]],
            ["complete", "partial"],
        )

    def test_v17_visual_cache_requires_matching_plan_and_policy(self) -> None:
        plan = decompose_visual_query("蓝色汽车", self.policy)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "visual.json"
            write_json_atomic(
                path,
                {
                    "query": "蓝色汽车",
                    "encoded_query": "蓝色汽车",
                    "library_revision": "rev1",
                    "attribute_policy_sha256": "policy1",
                    "attribute_plan": {"fingerprint": plan.fingerprint},
                },
            )
            self.assertTrue(
                cached_visual_query_matches(
                    path,
                    "蓝色汽车",
                    "蓝色汽车",
                    "rev1",
                    plan.fingerprint,
                    "policy1",
                )
            )
            self.assertFalse(
                cached_visual_query_matches(
                    path,
                    "蓝色汽车",
                    "蓝色汽车",
                    "rev1",
                    plan.fingerprint,
                    "changed-policy",
                )
            )


if __name__ == "__main__":
    unittest.main()
