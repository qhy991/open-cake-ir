"""CPU checks for the two numerical boundaries and lease-before-allocation rule."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from prepare_l1_069 import source_for


class ScheduleContract(unittest.TestCase):
    def test_original_row_variants_keep_rounding_before_reduction_and_weight(self):
        from open_cake_ir.compiler.core import Compiler
        from open_cake_ir.compiler.frontend import parse
        from open_cake_ir.compiler.toolchain import project_triton_kernel, validate_triton_kernel

        root = Path(os.environ["CAKE_FROZEN_COMPILER_ROOT"])
        compiler = Compiler.load(root)
        self.assertEqual(compiler.commit, "5bb474c6df2948d1cc50b2d46f9ab828525be3c2")
        for rows in (8192, 131, 4096, 512, 2164, 2048, 586, 1024, 256):
            with self.subTest(rows=rows):
                assessment = compiler.assess(parse(source_for(rows, 8192, 1e-5)).document)
                self.assertTrue(assessment.lowering_eligible)
                operations = {op.op_id: op for op in assessment.typed_schedule.operations}
                self.assertEqual(operations["widen_rounded_sum"].reads, ("sum_bf16",))
                self.assertEqual(operations["square"].reads, ("values",))
                self.assertEqual(operations["widen_normalized"].reads, ("normalized_bf16",))
                self.assertEqual(operations["weighted"].reads, ("normalized_fp32", "weights"))
                lowered = compiler.lower(assessment)
                validate_triton_kernel(project_triton_kernel(lowered.source.encode(),
                                      lowered.toolchain_requirements), lowered.toolchain_requirements)
                self.assertEqual(lowered.toolchain_requirements["grid"], [rows, 1, 1])


class LeaseBoundary(unittest.TestCase):
    def test_missing_lease_fails_before_any_tensor_allocation(self):
        trace = []
        torch = ModuleType("torch")
        torch.bfloat16 = "bf16"
        torch.no_grad = lambda: lambda fn: fn
        torch.empty_like = lambda tensor: trace.append("allocation")
        driver = ModuleType("open_cake_ir.evaluation.metax_driver")
        driver.load_runtime = lambda path: trace.append("runtime_load")
        driver._call = lambda *args: trace.append("native_call")
        admission = ModuleType("open_cake_ir.evaluation.triton_metax")

        def refuse(*args, **kwargs):
            trace.append("lease_admission")
            raise ValueError("No MACA lease")

        admission.observe_local_metax = refuse
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "index.json").write_text(json.dumps({
                "status": "compiled", "target": "xcore1002",
                "compiler_commit": "5bb474c6df2948d1cc50b2d46f9ab828525be3c2",
                "runtime_library": "/unused/libmcruntime.so",
                "cases": [{"shape": [1, 131, 8192], "eps": 1e-5, "variant": "example"}],
                "variants": {},
            }))
            with patch.dict(sys.modules, {"torch": torch,
                    "open_cake_ir.evaluation.metax_driver": driver,
                    "open_cake_ir.evaluation.triton_metax": admission}), \
                    patch.dict(os.environ, {"CAKE_BENCH_ARTIFACTS": str(root)}):
                spec = importlib.util.spec_from_file_location("candidate_contract_test",
                    Path(__file__).with_name("l1_069_cake_candidate.py"))
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                device = SimpleNamespace(type="cuda", index=0)
                def tensor(shape):
                    return SimpleNamespace(shape=shape, dtype="bf16", device=device,
                                           is_contiguous=lambda: True)
                with self.assertRaisesRegex(ValueError, "No MACA lease"):
                    module.run(tensor((1, 131, 8192)), tensor((1, 131, 8192)), tensor((8192,)), 1e-5)
        self.assertEqual(trace, ["lease_admission"])


if __name__ == "__main__":
    unittest.main()
