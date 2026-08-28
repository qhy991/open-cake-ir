from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.compiler import Compiler  # noqa: E402


class StreamingTopKTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")
        cls.schedule = ROOT / "corpus/schedules/top-k-streaming-b8-smoke.json"

    def test_top_k_state_is_initialized_once_and_merged_inside_the_loop(self) -> None:
        assessment = self.compiler.assess_file(self.schedule)

        self.assertTrue(assessment.accepted)
        self.assertTrue(assessment.lowering_eligible)
        source = self.compiler.lower(assessment).source
        loop = source.index("for score_start in tl.range")
        self.assertLess(source.index('top_values = tl.full((8,), float("-inf")'), loop)
        self.assertLess(source.index("top_indices = tl.full((8,), 2147483647"), loop)
        self.assertGreater(source.index("select_blocks_previous_values = top_values"), loop)
        self.assertIn(
            "select_blocks_source_positions = score_start + tl.arange(0, 8)",
            source,
        )

    def test_across_loop_without_a_loop_fails_locally(self) -> None:
        assessment = self.compiler.assess_file(
            ROOT / "corpus/schedules/top-k-b8-smoke-across-loop-drift.json"
        )

        self.assertFalse(assessment.accepted)
        self.assertIn(
            "TOP_K_LOOP_REQUIRED",
            [finding.code for finding in assessment.findings],
        )

    def test_score_state_shape_must_equal_the_loop_tile(self) -> None:
        document = json.loads(self.schedule.read_text(encoding="utf-8"))
        next(buffer for buffer in document["buffers"] if buffer["name"] == "score_tile")[
            "shape"
        ] = [4]

        assessment = self.compiler.assess(document)

        self.assertFalse(assessment.accepted)
        self.assertIn(
            "TOP_K_LOOP_TILE_MISMATCH",
            [finding.code for finding in assessment.findings],
        )


if __name__ == "__main__":
    unittest.main()
