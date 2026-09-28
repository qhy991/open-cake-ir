"""FIB 009 small-M scalar and column-reuse mappings on exact B300."""
from __future__ import annotations

import unittest
from pathlib import Path

from open_cake_ir.compiler import Compiler, Program
from open_cake_ir.compiler.frontend import parse
from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.tasks.solx_fib import gemm
from open_cake_ir.tasks.solx_fib.b300_gemv009 import (
    column_tiled_source, scalar_warp_program,
)

ROOT = Path(__file__).resolve().parents[2]
TASK = "fib_gemm_n5120_k2048"


class Fib009GemvTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")

    def test_scalar_warp_rewrite_preserves_each_small_workload(self) -> None:
        for rows in (1, 2, 4, 5, 6, 8):
            workload = WorkloadContract(gemm.workload_document(TASK, rows=rows))
            for warps in (4, 8):
                with self.subTest(rows=rows, warps=warps):
                    program = scalar_warp_program(
                        self.compiler, workload, num_warps=warps)
                    lowered = self.compiler.lower_program(program)
                    lowered.validate_binding()
                    leaf = lowered.lowerings[0]
                    self.assertEqual(leaf.target, "sm_103a")
                    self.assertEqual(leaf.toolchain_requirements["grid"],
                                     [rows, 5120, 1])
                    self.assertEqual(
                        leaf.toolchain_requirements["compile_options"]["num_warps"],
                        warps)

    def test_m1_column_reuse_has_complete_disjoint_output_tiles(self) -> None:
        workload = WorkloadContract(gemm.workload_document(TASK, rows=1))
        for tile in (2, 4, 8):
            for warps in (4, 8):
                with self.subTest(tile=tile, warps=warps):
                    schedule = parse(column_tiled_source(
                        workload, columns_per_cta=tile,
                        execution_groups=warps)).document
                    self.assertEqual(schedule["program_map"]["axes"][1]["tile"], tile)
                    assessment = self.compiler.assess(schedule)
                    self.assertFalse(
                        [f for f in assessment.findings if f.blocks_lowering],
                        assessment.findings)
                    lowered = self.compiler.lower_program(
                        Program.from_schedule(schedule))
                    lowered.validate_binding()
                    leaf = lowered.lowerings[0]
                    self.assertEqual(leaf.toolchain_requirements["grid"],
                                     [1, 5120 // tile, 1])
                    self.assertIn("tl.sum", leaf.source)

    def test_other_target_task_or_unbounded_shape_is_refused(self) -> None:
        workload = WorkloadContract(gemm.workload_document(TASK, rows=1))
        with self.assertRaisesRegex(ValueError, "2, 4 or 8"):
            column_tiled_source(workload, columns_per_cta=3)
        with self.assertRaisesRegex(ValueError, "official M=1"):
            column_tiled_source(WorkloadContract(gemm.workload_document(
                TASK, rows=2)))
        for other in (
                WorkloadContract(gemm.workload_document(
                    TASK, rows=1, backend="triton-b200")),
                WorkloadContract(gemm.workload_document(
                    "fib_gemm_n128_k2048", rows=1)),
                WorkloadContract(gemm.workload_document(TASK, rows=16))):
            with self.assertRaises(ValueError):
                scalar_warp_program(self.compiler, other, num_warps=8)


if __name__ == "__main__":
    unittest.main()
