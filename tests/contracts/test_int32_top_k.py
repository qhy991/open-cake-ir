from __future__ import annotations

import unittest
from pathlib import Path

from open_cake_ir.compiler import Compiler

ROOT = Path(__file__).resolve().parents[2]


class Int32TopKContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")

    def test_resident_signed_values_lower_through_monotonic_uint32_keys(self) -> None:
        assessment = self.compiler.assess_file(
            ROOT / "corpus/schedules/top-k-int32-b8-smoke.json"
        )

        self.assertTrue(assessment.accepted)
        self.assertTrue(assessment.lowering_eligible)
        source = self.compiler.lower(assessment).source
        self.assertIn(
            "select_experts_score_bits = score_row.to(tl.uint32, bitcast=True)",
            source,
        )
        self.assertIn(
            "select_experts_ordered_scores = select_experts_score_bits ^ 0x80000000",
            source,
        )
        self.assertIn(
            "select_experts_ranked_score_bits = "
            "select_experts_ranked_ordered_scores ^ 0x80000000",
            source,
        )
        self.assertIn(
            "select_experts_ranked_scores = "
            "select_experts_ranked_score_bits.to(tl.int32, bitcast=True)",
            source,
        )
        self.assertIn(
            "top_values = tl.where(select_experts_ranked_valid, "
            "select_experts_ranked_scores, -2147483648)",
            source,
        )
        self.assertNotIn("select_experts_canonical_scores", source)

        values = (-2_147_483_648, -1, 0, 1, 2_147_483_647)
        keys = [((value & 0xFFFFFFFF) ^ 0x80000000) for value in values]
        self.assertEqual(sorted(values), [value for _, value in sorted(zip(keys, values))])

    def test_loop_carried_int32_state_is_an_explicit_backend_exclusion(self) -> None:
        assessment = self.compiler.assess_file(
            ROOT / "corpus/schedules/top-k-int32-streaming-b8-drift.json"
        )

        self.assertFalse(assessment.lowering_eligible)
        finding = next(
            item
            for item in assessment.findings
            if item.code == "TOP_K_INT32_ACROSS_LOOP_UNLOWERABLE"
        )
        self.assertEqual(
            finding.path,
            "operations[1].parameters.across_loop",
        )


if __name__ == "__main__":
    unittest.main()
