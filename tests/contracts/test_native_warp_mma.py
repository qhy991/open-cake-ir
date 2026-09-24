"""The B300 native route emits the admitted TMA and warp-partitioned MMA contract."""

import json
from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler
from open_cake_ir.compiler.backends import native_cuda
from open_cake_ir.compiler.ir import Schedule
from open_cake_ir.compiler.target import Target


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = Path(__file__).parent / "fixtures/native_tinygemm_partitioned_b1.json"
TARGET = Target.load(ROOT / "compiler/targets/sm_103a.json")


def document():
    return json.loads(FIXTURE.read_text())


class NativeWarpMma(unittest.TestCase):
    def test_short_k_fixture_emits_tma_warp_mma_with_exact_public_abi(self):
        compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")
        for batch, features, depth in ((1, 128, 720), (16, 1024, 1024)):
            with self.subTest(shape=(batch, features, depth)):
                source = document()
                source["schedule_id"] = f"native-warp-b{batch}-n{features}-k{depth}"
                shape = {
                    "input": [batch, depth], "weight": [features, depth],
                    "bias": [features], "output": [batch, features],
                }
                for buffer in source["buffers"]:
                    if buffer["name"] in shape:
                        buffer["shape"] = shape[buffer["name"]]
                assessment = compiler.assess(source)
                self.assertTrue(assessment.accepted, assessment.findings)
                self.assertTrue(assessment.lowering_eligible, assessment.findings)
                emission = compiler.lower(assessment)
                self.assertIn("cp.async.bulk.tensor.2d", emission.source)
                self.assertIn("CUtensorMap activation_map,", emission.source)
                self.assertIn("CUtensorMap weight_map,", emission.source)
                self.assertIn("tma_load(dst, &activation_map", emission.source)
                self.assertIn("tma_load(dst, &weight_map", emission.source)
                self.assertIn("handle->a_map, handle->b_map", emission.source)
                self.assertIn("ldmatrix.sync.aligned.m8n8.x4", emission.source)
                self.assertIn("mma.sync.aligned.m16n8k16", emission.source)
                self.assertIn("barrier.sync 2, 128", emission.source)
                self.assertIn("prefetch.tensormap", emission.source)
                self.assertLess(emission.source.index("// CAKE_OP:load_bias"),
                                emission.source.index("barrier_wait(base + kBarrierOffset + warp * 8)"))
                self.assertLess(emission.source.index("bias0 = feature0"),
                                emission.source.index("barrier_wait(base + kBarrierOffset + warp * 8)"))
                self.assertEqual(emission.toolchain_requirements["argument_order"],
                                 ["input", "weight", "bias", "output"])
                self.assertEqual(emission.toolchain_requirements["grid"],
                                 [1, features // 8, 1])
                self.assertEqual(emission.toolchain_requirements["block"], [384, 1, 1])

    def test_route_refuses_unqualified_shapes_and_broken_hardware_commitments(self):
        changes = (
            ("shape", "NATIVE_WARP_SHAPE_EVIDENCE"),
            ("partitions", "NATIVE_WARP_MMA"),
            ("descriptor", "NATIVE_WARP_TMA"),
            ("roles", "NATIVE_WARP_ROLES"),
            ("store", "NATIVE_WARP_EPILOGUE"),
        )
        for change, expected in changes:
            with self.subTest(change=change):
                source = document()
                if change == "shape":
                    next(b for b in source["buffers"] if b["name"] == "weight")["shape"] = [256, 720]
                elif change == "partitions":
                    mma = next(o for o in source["operations"] if o["kind"] == "mma")
                    mma["parameters"]["k_partitions"] = [
                        [0, 255], [255, 512], [512, 768], [768, 1024]
                    ]
                elif change == "descriptor":
                    next(o for o in source["operations"] if o["id"] == "load_activation")[
                        "parameters"]["descriptor_box"] = [16, 128]
                elif change == "roles":
                    source["roles"][0]["execution_groups"] = [0, 1]
                else:
                    next(o for o in source["operations"] if o["kind"] == "store")[
                        "parameters"]["coalesced"] = True
                schedule = Schedule.from_dict(source)
                codes = {finding.code for finding in native_cuda.preflight(schedule, TARGET)}
                self.assertIn(expected, codes)
