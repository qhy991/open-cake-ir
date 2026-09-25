"""Bounded native CUDA MN-major B route proven by one exact-B300 PTX probe.

The external probe used BF16 A[128,128] in TMEM, physical B[128,32] in
swizzle-64B shared memory, and accumulated FP32 C[128,32]. This test pins the
typed emission and its intended refusal guards; device evidence is retained
under F-2026-09-24-003 rather than inferred from this source test.
"""
from __future__ import annotations

import json
from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler, Schedule, Target
from open_cake_ir.compiler.backends.native_cuda import preflight
from open_cake_ir.compiler.verifier import verify

ROOT = Path(__file__).resolve().parents[2]


def mn_document() -> dict:
    d = json.loads((ROOT / "tests/fixtures/tmem-state-mma-sm103a.json").read_text())
    d["schedule_id"] = "tmem-a-mn-major-b-k128-witness"
    d["lowering"]["entry_point"] = "cake_tmem_a_mn_major_b_k128_witness"
    for item in d["buffers"]:
        if item["name"] in ("a", "a_reg", "a_tmem"):
            item["shape"] = [128, 128]
        elif item["name"] in ("b", "b_stage"):
            item["shape"] = [128, 32]
        elif item["name"] in ("c", "acc", "dot"):
            item["shape"] = [128, 32]
        if item["name"] == "b_stage":
            item["swizzle"] = "swizzle_64b"
        if item["name"] == "acc":
            item["byte_offset"] = 32768
    stage = next(op for op in d["operations"] if op["id"] == "load_b")
    stage["parameters"]["descriptor_box"] = [128, 32]
    mma = next(op for op in d["operations"] if op["id"] == "mma")
    mma["parameters"]["tile_shape"] = [128, 32, 128]
    mma["parameters"]["instruction"]["shape"] = [128, 32, 16]
    mma["parameters"]["instruction"]["operand_major"] = ["k", "mn"]
    return d


class NativeMnMajorB(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")
        cls.target = Target.load(ROOT / "compiler/targets/sm_103a.json")

    def test_exact_proven_route_emits_b_major_and_k_step(self):
        d = mn_document()
        schedule = Schedule.from_dict(d)
        self.assertFalse([f for f in verify(schedule, self.target) if f.blocks_lowering])
        self.assertEqual(preflight(schedule, self.target), ())
        assessment = self.compiler.assess(d)
        self.assertTrue(assessment.lowering_eligible, [
            (f.code, f.path) for f in assessment.findings if f.blocks_lowering
        ])
        lowering = self.compiler.lower(assessment)
        source = lowering.source
        self.assertIn("for (int atom=0; atom<8; ++atom)", source)
        self.assertIn("atom*1024, 512, 4), 134808720u", source)
        self.assertIn("cuuint64_t dims[2] = {32, 128}", source)
        self.assertIn("CU_TENSOR_MAP_SWIZZLE_64B", source)
        self.assertEqual(lowering.toolchain_requirements["argument_order"],
                         ["a", "b", "c"])

    def test_wrong_b_orientation_is_owned_by_the_shape_rule(self):
        d = mn_document()
        next(b for b in d["buffers"] if b["name"] == "b_stage")["shape"] = [32, 128]
        findings = verify(Schedule.from_dict(d), self.target)
        self.assertIn("MMA_INPUT_TILE_DOMAIN", {f.code for f in findings})
        self.assertIn("NATIVE_MMA_TILE_DOMAIN",
                      {f.code for f in preflight(Schedule.from_dict(d), self.target)})

    def test_mn_route_refuses_unmeasured_swizzle_and_geometry(self):
        d = mn_document()
        next(b for b in d["buffers"] if b["name"] == "b_stage")["swizzle"] = "swizzle_128b"
        self.assertIn("NATIVE_MN_MAJOR_B_UNQUALIFIED",
                      {f.code for f in preflight(Schedule.from_dict(d), self.target)})
        d = mn_document()
        next(op for op in d["operations"] if op["id"] == "mma")["parameters"]["tile_shape"] = [128, 64, 128]
        self.assertIn("NATIVE_MN_MAJOR_B_UNQUALIFIED",
                      {f.code for f in preflight(Schedule.from_dict(d), self.target)})


if __name__ == "__main__":
    unittest.main()
