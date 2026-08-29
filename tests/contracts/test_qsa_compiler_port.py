from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.compiler import Compiler  # noqa: E402


class QsaCompilerPortTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")

    def _source(self, name: str) -> str:
        assessment = self.compiler.assess_file(ROOT / "corpus/schedules" / name)
        self.assertTrue(assessment.accepted, name)
        self.assertTrue(assessment.lowering_eligible, name)
        return self.compiler.lower(assessment).source

    def test_every_qsa_program_node_lowers(self) -> None:
        for name in (
            "qsa-pool-t32768.json",
            "qsa-layernorm-t32768.json",
            "qsa-score-topk-t32768.json",
            "qsa-expand-t32768.json",
            "qsa-selected-attention-t32768.json",
        ):
            with self.subTest(schedule=name):
                self._source(name)

    def test_score_order_and_causal_loop_bound_are_visible(self) -> None:
        source = self._source("qsa-score-topk-t32768.json")

        relu = source.index("# CAKE_OP:relu_heads")
        head_sum = source.index("# CAKE_OP:sum_heads")
        top_k = source.index("# CAKE_OP:select_blocks")
        self.assertLess(relu, head_sum)
        self.assertLess(head_sum, top_k)
        self.assertIn("((query + 1) // 4)", source)
        self.assertIn("select_blocks_source_valid =", source)
        self.assertIn("select_blocks_ranked_keys = tl.topk(", source)
        self.assertNotIn("select_blocks_source_candidates_", source)
        self.assertLess(len(source.encode()), 10_000)
        self.assertIn("# CAKE_FINALIZE:select_blocks", source)
        self.assertIn("== 2147483647, -1", source)

    def test_attention_masks_sentinel_before_online_softmax(self) -> None:
        source = self._source("qsa-selected-attention-t32768.json")

        self.assertIn("softmax_update_effective_logits", source)
        self.assertIn("selected_ids[None, :] != -1", source)
        self.assertIn("# CAKE_FINALIZE:softmax_update", source)


if __name__ == "__main__":
    unittest.main()
