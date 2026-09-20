"""Existing MoE/QSA selection emissions must reach the isolated native compiler."""
import json
from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler
from open_cake_ir.compiler.toolchain import project_triton_kernel, validate_triton_kernel
from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.tasks.solx_fib import moe

ROOT = Path(__file__).resolve().parents[2]


class SelectionSourceAdmission(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")

    def test_original_moe_routing_stages_reach_the_native_source_boundary(self):
        plan = moe.launch_plan(WorkloadContract(moe.workload_document()))
        for stage in plan.stages[:4]:
            with self.subTest(stage=stage.name):
                assessment = self.compiler.assess(json.loads(stage.schedule_bytes))
                self.assertTrue(assessment.lowering_eligible, assessment.findings)
                lowered = self.compiler.lower(assessment)
                kernel = project_triton_kernel(lowered.source.encode(), lowered.toolchain_requirements)
                validate_triton_kernel(kernel, lowered.toolchain_requirements)

    def test_existing_qsa_streaming_and_sorted_half_selection_project(self):
        for merge in (1, 2):
            with self.subTest(source_tiles_per_merge=merge):
                document = json.loads((ROOT / "corpus/schedules/qsa-score-topk-t32768.json").read_text())
                op = next(op for op in document["operations"] if op["kind"] == "top_k")
                op["parameters"]["source_tiles_per_merge"] = merge
                assessment = self.compiler.assess(document)
                self.assertTrue(assessment.lowering_eligible, assessment.findings)
                lowered = self.compiler.lower(assessment)
                kernel = project_triton_kernel(lowered.source.encode(), lowered.toolchain_requirements)
                self.assertIn(b"tl.topk(" if merge == 1 else b"tl.bitonic_merge(", kernel)
                validate_triton_kernel(kernel, lowered.toolchain_requirements)

    def test_selection_admission_keeps_calls_closed_and_module_attributes_direct(self):
        source = b'''import triton
import triton.language as tl
@triton.jit
def choose(x, out):
    offsets = tl.arange(0, 32)
    values = tl.load(x + offsets)
    chosen = tl.topk(values, 2)
    tl.store(out + tl.arange(0, 2), chosen)
'''
        requirements = {"kernel_entry_point": "choose", "signature": {"x": "*fp32", "out": "*fp32"},
                        "compile_constants": {}}
        validate_triton_kernel(source, requirements)
        for expression in (b"tl.topk.__call__(values, 2)", b"tl.topk_unchecked(values, 2)",
                           b"tl.topk(values, 2, **options)", b"eval('values')"):
            with self.subTest(expression=expression), self.assertRaises(ValueError):
                validate_triton_kernel(source.replace(b"tl.topk(values, 2)", expression), requirements)
