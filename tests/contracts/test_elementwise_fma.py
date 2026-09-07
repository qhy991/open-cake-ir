"""The bounded FP32 FMA contract; optional real compilation never launches a GPU."""

from __future__ import annotations

import ast
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest

from jsonschema import Draft202012Validator

from open_cake_ir.compiler import Compiler
from open_cake_ir.compiler.performance.residency import logical_register_pressure_per_thread
from open_cake_ir.compiler.backends.triton import emit
from open_cake_ir.compiler.ir import ElementwiseOp, Schedule, ScheduleParseError
from open_cake_ir.compiler.schema import schedule_schema
from open_cake_ir.compiler.target import Target
from open_cake_ir.compiler.verifier import verify
from open_cake_ir.compiler.performance.work import work_bound

ROOT = Path(__file__).resolve().parents[2]
SCHEDULES = ROOT / "corpus/schedules"
TARGET = Target.load(ROOT / "compiler/targets/sm_100a.json")


def document(name="fma-b8-smoke.json"):
    return json.loads((SCHEDULES / name).read_text())


def operation(doc):
    return next(op for op in doc["operations"] if op["id"] == "fma")


class ElementwiseFmaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")

    def codes(self, doc):
        assessment = self.compiler.assess(doc)
        self.assertFalse(assessment.lowering_eligible)
        return {finding.code for finding in assessment.findings}

    def test_candidate_fixtures_parse_and_project_to_schema(self):
        schema = Draft202012Validator(schedule_schema())
        for path in SCHEDULES.glob("fma*.json"):
            with self.subTest(path=path.name):
                doc = json.loads(path.read_text())
                self.assertEqual(list(schema.iter_errors(doc)), [])
                self.assertEqual(Schedule.from_dict(doc).operation("fma").parameters.op,
                                 ElementwiseOp.FMA)

    def test_positive_and_nested_lowering(self):
        for name, count in [("fma-b8-smoke.json", 1), ("fma-chain-b8-smoke.json", 2)]:
            with self.subTest(name=name):
                assessment = self.compiler.assess(document(name))
                self.assertTrue(assessment.accepted, assessment.findings)
                self.assertTrue(assessment.lowering_eligible, assessment.findings)
                source = self.compiler.lower(assessment).source
                ast.parse(source)
                self.assertEqual(source.count('fma.rn.f32 $0, $1, $2, $3;'), count)
                self.assertIn('constraints="=f,f,f,f"', source)
                self.assertIn("dtype=tl.float32, is_pure=True, pack=1", source)
                self.assertNotIn("fma.rn.ftz", source)
                self.assertNotIn("fma.rn.sat", source)
                if count == 2:
                    self.assertIn("product = a_tile * b_tile", source)
                    self.assertIn("args=[inner, c_tile, product]", source)

    def test_missing_instruction_is_structural(self):
        doc = document()
        operation(doc)["parameters"].pop("instruction")
        with self.assertRaisesRegex(ScheduleParseError, "instruction is required for fma"):
            Schedule.from_dict(doc)
        self.assertTrue(list(Draft202012Validator(schedule_schema()).iter_errors(doc)))
        self.assertIn("SCHEDULE_STRUCTURE", self.codes(doc))

    def test_scalar_and_broadcast_are_not_admitted_even_as_null(self):
        for key, value in [("scalar", 1.0), ("scalar", None),
                           ("broadcast_axis", 0), ("broadcast_axis", None)]:
            with self.subTest(key=key, value=value):
                doc = document()
                operation(doc)["parameters"][key] = value
                with self.assertRaises(ScheduleParseError):
                    Schedule.from_dict(doc)
                self.assertTrue(list(Draft202012Validator(schedule_schema()).iter_errors(doc)))

    def test_wrong_input_arity(self):
        for reads in [[], ["a_tile"], ["a_tile", "b_tile"],
                      ["a_tile", "b_tile", "c_tile", "a_tile"]]:
            with self.subTest(reads=reads):
                doc = document()
                operation(doc)["reads"] = reads
                self.assertIn("ELEMENTWISE_ARITY", self.codes(doc))

    def test_one_result_is_required(self):
        for writes in [[], ["y_tile", "c_tile"]]:
            with self.subTest(writes=writes):
                doc = document()
                operation(doc)["writes"] = writes
                self.assertIn("ELEMENTWISE_FMA_RESULT_COUNT", self.codes(doc))

    def test_fp32_is_not_implicit_mixed_precision(self):
        for dtype in ["fp16", "bf16", "int32"]:
            with self.subTest(dtype=dtype):
                doc = document()
                next(b for b in doc["buffers"] if b["name"] == "a_tile")["dtype"] = dtype
                self.assertIn("ELEMENTWISE_INSTRUCTION_DTYPE_DIFFERS", self.codes(doc))

    def test_unknown_operand_is_a_finding_not_a_crash(self):
        doc = document()
        operation(doc)["reads"][2] = "absent"
        self.assertFalse(self.compiler.assess(doc).accepted)

    def test_same_shape_is_required(self):
        doc = document()
        next(b for b in doc["buffers"] if b["name"] == "c_tile")["shape"] = [64]
        self.assertIn("ELEMENTWISE_SHAPE_MISMATCH", self.codes(doc))

    def test_global_operand_is_refused(self):
        doc = document()
        operation(doc)["reads"][2] = "c"
        self.assertIn("ELEMENTWISE_FMA_SPACE", self.codes(doc))

    def test_contract_cannot_be_borrowed_from_tanh(self):
        self.assertIn("ELEMENTWISE_INSTRUCTION_KIND_DIFFERS",
                      self.codes(document("fma-b8-smoke-instruction-drift.json")))
        doc = document("swiglu-b8-smoke.json")
        tanh = next(op for op in doc["operations"] if op["id"] == "tanh_gate")
        tanh["parameters"]["instruction"]["contract"] = "ptx.fma.rn.f32"
        self.assertIn("ELEMENTWISE_INSTRUCTION_KIND_DIFFERS", self.codes(doc))

    def test_target_must_admit_the_exact_contract(self):
        target = replace(TARGET, instruction_contracts=frozenset(
            c for c in TARGET.instruction_contracts if c != "ptx.fma.rn.f32"))
        codes = {f.code for f in verify(Schedule.from_dict(document()), target)}
        self.assertIn("TARGET_INSTRUCTION_UNSUPPORTED", codes)

    def test_unsupported_backends_refuse_before_emission(self):
        for backend in ["cutlass_cute_dsl", "checked_cuda_asset"]:
            with self.subTest(backend=backend):
                doc = document()
                doc["lowering"]["backend"] = backend
                self.assertFalse(self.compiler.assess(doc).lowering_eligible)

    def test_two_flops_per_element_and_unchanged_global_traffic(self):
        for name, flops in [("fma-b8-smoke.json", 2048),
                            ("fma-chain-b8-smoke.json", 5120)]:
            with self.subTest(name=name):
                bound = work_bound(Schedule.from_dict(document(name)))
                self.assertEqual(bound.flops, flops)
                self.assertTrue(bound.flops_exact)
                self.assertEqual(bound.compulsory_bytes, 4 * 8 * 128 * 4)

    def test_pressure_accounts_for_all_three_distinct_inputs(self):
        schedule = Schedule.from_dict(document())
        # Three 128-element FP32 inputs remain live at the FMA. Its output may
        # reuse one input in the existing logical proxy; this is not a register bound.
        self.assertEqual(logical_register_pressure_per_thread(schedule, TARGET), 3)

    @unittest.skipUnless(importlib.util.find_spec("triton") and
                         importlib.util.find_spec("torch"),
                         "real GPU-free compilation requires the Triton/Torch toolchain")
    def test_real_sm100_compile_preserves_fma_and_rounded_producer(self):
        import triton
        from triton.backends.compiler import GPUTarget
        from triton.compiler import ASTSource

        for name, minimum_fmas in [("fma-b8-smoke.json", 1),
                                   ("fma-chain-b8-smoke.json", 2)]:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                schedule = Schedule.from_dict(document(name))
                emission = emit(schedule, TARGET)
                path = Path(directory) / "candidate.py"
                path.write_text(emission.source)
                module_name = "_cake_fma_compile_test"
                spec = importlib.util.spec_from_file_location(module_name, path)
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                try:
                    spec.loader.exec_module(module)
                    toolchain = emission.toolchain
                    kernel = getattr(module, toolchain["kernel_entry_point"])
                    compiled = triton.compile(ASTSource(
                        kernel, signature=toolchain["signature"],
                        constexprs=toolchain["compile_constants"], attrs={}),
                        target=GPUTarget("cuda", 100, 32),
                        options=toolchain["compile_options"])
                    ptx = compiled.asm["ptx"]
                    fmas = len(re.findall(r"(?m)^\s*fma\.rn\.f32\b", ptx))
                    self.assertGreaterEqual(fmas, minimum_fmas)
                    self.assertNotRegex(ptx, r"fma\.[^;\n]*(ftz|sat)")
                    if minimum_fmas == 2:
                        self.assertRegex(ptx, r"mul(?:\.rn)?\.f32\b")
                    self.assertTrue(compiled.asm["cubin"])
                    print(json.dumps({
                        "case": name, "triton": triton.__version__,
                        "target": "cuda:100", "fma_rn_f32_instructions": fmas,
                        "cubin_bytes": len(compiled.asm["cubin"]),
                        "gpu_executed": False, "performance_measured": False,
                    }, sort_keys=True))
                finally:
                    sys.modules.pop(module_name, None)


if __name__ == "__main__":
    unittest.main()
