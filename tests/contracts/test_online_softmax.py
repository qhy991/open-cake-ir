from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.compiler import Compiler  # noqa: E402


class OnlineSoftmaxPrimitiveTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")

    def test_state_is_explicit_and_normalization_follows_the_loop(self) -> None:
        assessment = self.compiler.assess_file(
            ROOT / "corpus/schedules/online-softmax-b8-smoke.json"
        )

        self.assertTrue(assessment.accepted)
        self.assertTrue(assessment.lowering_eligible)
        source = self.compiler.lower(assessment).source
        loop = source.index("for selected_start in tl.range")
        update = source.index("# CAKE_OP:softmax_update")
        finalize = source.index("# CAKE_FINALIZE:softmax_update")
        store = source.index("# CAKE_OP:store_output")
        self.assertLess(loop, update)
        self.assertLess(update, finalize)
        self.assertLess(finalize, store)
        self.assertIn("running_max = tl.full((2,), float(\"-inf\")", source)
        self.assertIn("weighted_accumulator / running_sum", source)

    def test_unsupported_reduction_axis_fails_before_lowering(self) -> None:
        assessment = self.compiler.assess_file(
            ROOT / "corpus/schedules/online-softmax-b8-smoke-axis-drift.json"
        )

        self.assertFalse(assessment.accepted)
        self.assertFalse(assessment.lowering_eligible)
        self.assertIn(
            "ONLINE_SOFTMAX_LOGITS_SHAPE",
            [finding.code for finding in assessment.findings],
        )


if __name__ == "__main__":
    unittest.main()
