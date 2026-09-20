"""Measured C550 routing operations retain the unqualified top-k boundaries."""
import json
from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler, frontend
from open_cake_ir.compiler.backends import triton
from open_cake_ir.compiler.backends.common import EmitError
from open_cake_ir.compiler.ir import Schedule
from open_cake_ir.compiler.target import declared_target
from open_cake_ir.compiler.toolchain import project_triton_kernel
from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.tasks.solx_fib import attention, moe

ROOT = Path(__file__).resolve().parents[2]


class MetaxRoutingPrimitives(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")

    def test_observed_routing_shapes_use_the_public_compiler_and_native_projection(self):
        for suffix in ("group-scores", "expert-selection"):
            with self.subTest(stage=suffix):
                result = self.compiler.assess_file(ROOT / f"corpus/schedules/xcore1002-routing-{suffix}.py")
                self.assertTrue(result.lowering_eligible, result.findings)
                lowered = self.compiler.lower(result)
                self.assertEqual(lowered.toolchain_requirements["code_object"], "mcfatbin")
                self.assertEqual(lowered.toolchain_requirements["warp_size"], 64)
                self.assertIn(b"tl.topk(", project_triton_kernel(lowered.source.encode(), lowered.toolchain_requirements))

    def test_unqualified_top_k_forms_have_their_own_backend_refusal(self):
        for suffix in ("integer-topk-unqualified", "loop-topk-unqualified"):
            with self.subTest(form=suffix):
                path = ROOT / f"corpus/schedules/xcore1002-routing-{suffix}.py"
                document = frontend.read_schedule(path).document
                result = self.compiler.assess(document)
                self.assertTrue(result.accepted, result.findings)
                self.assertFalse(result.lowering_eligible)
                self.assertEqual([f.code for f in result.findings if f.blocks_lowering], ["MACA_TOP_K_UNQUALIFIED"])
                with self.assertRaises(EmitError):
                    triton.emit(Schedule.from_dict(document), declared_target("xcore1002"))
                document["target"] = "sm_103a"
                self.assertTrue(self.compiler.assess(document).lowering_eligible)

    def test_original_attention_and_moe_stage_structures_have_no_missing_target_operations(self):
        # This is stage-level portability, not a successor Workload: the original
        # frozen Workload documents keep their B300 identity and are never edited.
        for owner, tasks in ((attention, attention.TASKS), (moe, (moe.TASK,))):
            for task in tasks:
                workload = WorkloadContract(owner.workload_document(task))
                plan = owner.launch_plan(workload)
                for stage in plan.stages:
                    with self.subTest(task=task, stage=stage.name):
                        document = json.loads(stage.schedule_bytes)
                        document["target"] = "xcore1002"
                        result = self.compiler.assess(document)
                        self.assertTrue(result.lowering_eligible, result.findings)
                        self.compiler.lower(result)
                self.assertEqual(workload.target, "sm_103a")
