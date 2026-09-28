"""Exact-target FIB 010 segmented GEMV lowering and admission."""
from __future__ import annotations

import unittest
from pathlib import Path

from open_cake_ir.compiler import Compiler, Program
from open_cake_ir.compiler.frontend import parse
from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.tasks.solx_fib import gemm
from open_cake_ir.tasks.solx_fib.b300_gemv010 import column_tiled_source

ROOT = Path(__file__).resolve().parents[2]
TASK = "fib_gemm_n6144_k4096"


class Fib010GemvTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")

    def test_segments_cover_k_and_columns_cover_n(self) -> None:
        workload = WorkloadContract(gemm.workload_document(TASK, rows=1))
        for columns, k_tile, warps in ((2, 512, 4), (2, 1024, 4),
                                        (2, 2048, 4), (4, 1024, 8)):
            with self.subTest(columns=columns, k_tile=k_tile, warps=warps):
                schedule = parse(column_tiled_source(
                    workload, columns_per_cta=columns, k_tile=k_tile,
                    execution_groups=warps)).document
                self.assertEqual(schedule["program_map"]["axes"][1]["tile"], columns)
                reductions = [op for op in schedule["operations"]
                              if op["id"].startswith("sum_k_")]
                self.assertEqual(len(reductions), 4096 // k_tile)
                self.assertTrue(all(op["parameters"]["axis"] == 1
                                    for op in reductions))
                assessment = self.compiler.assess(schedule)
                self.assertFalse([f for f in assessment.findings if f.blocks_lowering],
                                 assessment.findings)
                lowered = self.compiler.lower_program(Program.from_schedule(schedule))
                lowered.validate_binding()
                leaf = lowered.lowerings[0]
                self.assertEqual(leaf.target, "sm_103a")
                self.assertEqual(leaf.toolchain_requirements["grid"],
                                 [1, 6144 // columns, 1])
                self.assertEqual(leaf.toolchain_requirements["compile_options"]["num_warps"],
                                 warps)

    def test_wrong_task_target_shape_and_tile_are_refused(self) -> None:
        workload = WorkloadContract(gemm.workload_document(TASK, rows=1))
        for kw in ({"columns_per_cta": 3}, {"k_tile": 256},
                   {"execution_groups": 1}):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                column_tiled_source(workload, **kw)
        for other in (
            WorkloadContract(gemm.workload_document(TASK, rows=2)),
            WorkloadContract(gemm.workload_document(TASK, rows=1,
                                                   backend="triton-b200")),
            WorkloadContract(gemm.workload_document("fib_gemm_n5120_k2048", rows=1)),
        ):
            with self.subTest(other=other.workload_id), self.assertRaises(ValueError):
                column_tiled_source(other)


if __name__ == "__main__":
    unittest.main()
