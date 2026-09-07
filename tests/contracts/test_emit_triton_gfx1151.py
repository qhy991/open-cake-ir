"""Exact gfx1151 projection of the target-neutral Triton lowering mechanism."""

from __future__ import annotations

import unittest
from pathlib import Path

from open_cake_ir.compiler.emit_triton import emit, preflight
from open_cake_ir.compiler.ir import Schedule
from open_cake_ir.compiler.target import Target


ROOT = Path(__file__).resolve().parents[2]
TARGET_PATH = ROOT / "compiler/targets/gfx1151.json"
SWIGLU_PATH = ROOT / "corpus/schedules/swiglu-b8-smoke-gfx1151.json"
RMSNORM_BASELINE_PATH = (
    ROOT / "corpus/schedules/llama-rmsnorm-mul-b8-gfx1151-r64-w4.json"
)
RMSNORM_ONE_ROW_PATH = (
    ROOT / "corpus/schedules/llama-rmsnorm-mul-b8-gfx1151-r1-w8.json"
)


class Gfx1151TritonProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.target = Target.load(TARGET_PATH)

    def test_swiglu_uses_one_triton_route_with_exact_hip_runtime_custody(self) -> None:
        schedule = Schedule.load(SWIGLU_PATH)

        self.assertEqual(preflight(schedule, self.target), ())
        emission = emit(schedule, self.target)

        self.assertEqual(schedule.lowering.backend.value, "triton")
        self.assertIn("if torch.version.hip is None:", emission.source)
        self.assertIn('!= "gfx1151"', emission.source)
        self.assertIn("libdevice.tanh", emission.source)
        self.assertEqual(
            emission.toolchain["triton_target"],
            {"backend": "hip", "arch": "gfx1151", "warp_size": 32},
        )
        self.assertEqual(emission.toolchain["binary_role"], "hsaco")
        self.assertEqual(emission.toolchain["assembly_role"], "amdgcn")

    def test_true_one_row_rmsnorm_derives_4096_workgroups_and_eight_waves(self) -> None:
        schedule = Schedule.load(RMSNORM_ONE_ROW_PATH)

        self.assertEqual(preflight(schedule, self.target), ())
        emission = emit(schedule, self.target)

        self.assertEqual(emission.toolchain["grid"], [512, 8, 1])
        self.assertEqual(emission.toolchain["compile_options"], {"num_warps": 8})
        self.assertEqual(
            emission.toolchain["compile_constants"]["BLOCK_ROW_BLOCK"], 1
        )
        self.assertIn("tl.arange(0, BLOCK_ROW_BLOCK)", emission.source)
        self.assertIn("tl.sum(sq.to(tl.float32), axis=1)", emission.source)

    def test_fixed_baseline_remains_a_distinct_64_row_four_wave_schedule(self) -> None:
        emission = emit(Schedule.load(RMSNORM_BASELINE_PATH), self.target)

        self.assertEqual(emission.toolchain["grid"], [8, 8, 1])
        self.assertEqual(emission.toolchain["compile_options"], {"num_warps": 4})
        self.assertEqual(
            emission.toolchain["compile_constants"]["BLOCK_ROW_BLOCK"], 64
        )


if __name__ == "__main__":
    unittest.main()
