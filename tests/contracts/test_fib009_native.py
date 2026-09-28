"""Exact FIB 009 native CUDA seed and its M8828 ownership boundary."""
from __future__ import annotations

import unittest
from pathlib import Path

from open_cake_ir.compiler import Compiler, Program
from open_cake_ir.compiler.frontend import parse
from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.tasks.solx_fib import gemm
from open_cake_ir.tasks.solx_fib.b300_native009 import (
    native_m8828_n128_source, native_m8828_source,
)

ROOT = Path(__file__).resolve().parents[2]


class Fib009NativeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")

    def test_complete_exact_m8828_program_lowers(self) -> None:
        workload = WorkloadContract(gemm.workload_document(
            "fib_gemm_n5120_k2048", rows=8828))
        schedule = parse(native_m8828_source(workload)).document
        self.assertEqual(schedule["lowering"]["backend"], "native_cuda")
        self.assertEqual(schedule["outputs"], ["out"])
        mma = next(op for op in schedule["operations"] if op["kind"] == "mma")
        self.assertEqual(mma["parameters"]["instruction"]["contract"],
                         "tcgen05.mma.cta_group::1.kind::f16")
        assessment = self.compiler.assess(schedule)
        self.assertFalse([f for f in assessment.findings if f.blocks_lowering],
                         assessment.findings)
        lowered = self.compiler.lower_program(Program.from_schedule(schedule))
        lowered.validate_binding()
        leaf = lowered.lowerings[0]
        self.assertEqual(leaf.target, "sm_103a")
        self.assertEqual(leaf.toolchain_requirements["grid"], [69, 80, 1])
        self.assertEqual(leaf.toolchain_requirements["argument_order"],
                         ["a", "b", "out"])
        self.assertIn("tcgen05.mma.cta_group::1.kind::f16", leaf.source)

    def test_other_shape_target_and_task_are_refused(self) -> None:
        for task, rows, backend in (
                ("fib_gemm_n5120_k2048", 952, "triton-b300"),
                ("fib_gemm_n5120_k2048", 8828, "triton-b200"),
                ("fib_gemm_n128_k2048", 8828, "triton-b300")):
            with self.subTest(task=task, rows=rows, backend=backend):
                workload = WorkloadContract(gemm.workload_document(
                    task, rows=rows, backend=backend))
                with self.assertRaises(ValueError):
                    native_m8828_source(workload)
                with self.assertRaises(ValueError):
                    native_m8828_n128_source(workload)

    def test_n128_successor_owns_half_as_many_column_ctas(self) -> None:
        workload = WorkloadContract(gemm.workload_document(
            "fib_gemm_n5120_k2048", rows=8828))
        schedule = parse(native_m8828_n128_source(workload)).document
        self.assertEqual(schedule["program_map"]["axes"][1]["tile"], 128)
        mma = next(op for op in schedule["operations"] if op["kind"] == "mma")
        self.assertEqual(mma["parameters"]["tile_shape"], [128, 128, 64])
        assessment = self.compiler.assess(schedule)
        self.assertFalse([f for f in assessment.findings if f.blocks_lowering],
                         assessment.findings)
        lowered = self.compiler.lower_program(Program.from_schedule(schedule))
        lowered.validate_binding()
        leaf = lowered.lowerings[0]
        self.assertEqual(leaf.toolchain_requirements["grid"], [69, 40, 1])
        self.assertEqual(leaf.toolchain_requirements["argument_order"],
                         ["a", "b", "out"])


if __name__ == "__main__":
    unittest.main()
