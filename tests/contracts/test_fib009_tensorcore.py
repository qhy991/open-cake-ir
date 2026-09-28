"""Bounded Cake FP16 dot mapping for the exact B300 FIB 009 Workload."""
from __future__ import annotations

import unittest
from pathlib import Path

from open_cake_ir.compiler import Compiler, Program
from open_cake_ir.compiler.frontend import parse
from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.tasks.solx_fib import gemm
from open_cake_ir.tasks.solx_fib.b300_tensorcore009 import TILES, tensorcore_source

ROOT = Path(__file__).resolve().parents[2]
TASK = "fib_gemm_n5120_k2048"


class Fib009TensorcoreTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")

    def test_bounded_structural_tiles_lower_at_three_official_batch_sizes(self) -> None:
        for rows in (128, 952, 8828):
            workload = WorkloadContract(gemm.workload_document(TASK, rows=rows))
            for block_m, block_n, block_k in sorted(TILES):
                with self.subTest(rows=rows, tile=(block_m, block_n, block_k)):
                    schedule = parse(tensorcore_source(
                        workload, block_m=block_m, block_n=block_n,
                        block_k=block_k)).document
                    dot = next(op for op in schedule["operations"]
                               if op["id"] == "dot")
                    self.assertEqual(dot["parameters"]["instruction"]["contract"],
                                     "triton.dot.fp16_fp32")
                    assessment = self.compiler.assess(schedule)
                    self.assertFalse(
                        [f for f in assessment.findings if f.blocks_lowering],
                        assessment.findings)
                    lowered = self.compiler.lower_program(
                        Program.from_schedule(schedule))
                    lowered.validate_binding()
                    leaf = lowered.lowerings[0]
                    self.assertEqual(leaf.target, "sm_103a")
                    self.assertEqual(leaf.toolchain_requirements["grid"],
                                     [(rows + block_m - 1) // block_m,
                                      (5120 + block_n - 1) // block_n, 1])
                    self.assertIn("tl.dot", leaf.source)

    def test_scope_refuses_other_target_task_short_m_and_tile(self) -> None:
        workload = WorkloadContract(gemm.workload_document(TASK, rows=128))
        with self.assertRaisesRegex(ValueError, "bounded tensor-core study"):
            tensorcore_source(workload, block_n=32)
        with self.assertRaisesRegex(ValueError, "M>=16"):
            tensorcore_source(WorkloadContract(gemm.workload_document(TASK, rows=1)))
        with self.assertRaisesRegex(ValueError, "exact B300"):
            tensorcore_source(WorkloadContract(gemm.workload_document(
                TASK, rows=128, backend="triton-b200")))
        with self.assertRaisesRegex(ValueError, "exact B300"):
            tensorcore_source(WorkloadContract(gemm.workload_document(
                "fib_gemm_n6144_k4096", rows=128)))


if __name__ == "__main__":
    unittest.main()
