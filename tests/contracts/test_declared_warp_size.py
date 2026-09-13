"""Role-slot width is declared by the Target that owns it, not by a shared constant.

F-2026-09-13-006 step 1. The readers were already correct -- every one of them reads
`target.warp_size` -- so what these check is that the value now comes from the document,
that an undeclared width is refused rather than substituted, and that a declared width
actually reaches the shared computations.
"""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from open_cake_ir.compiler.ir import Schedule
from open_cake_ir.compiler.performance.residency import residency_upper_bound
from open_cake_ir.compiler.target import Target, TargetParseError

ROOT = Path(__file__).resolve().parents[2]
TARGETS = sorted((ROOT / "compiler/targets").glob("*.json"))


def document(name: str) -> dict:
    return json.loads((ROOT / "compiler/targets" / f"{name}.json").read_text(encoding="utf-8"))


class DeclaredWarpSize(unittest.TestCase):
    def test_every_released_target_declares_the_width_it_is_read_with(self):
        self.assertTrue(TARGETS)
        for path in TARGETS:
            with self.subTest(target=path.stem):
                value = json.loads(path.read_text(encoding="utf-8"))
                self.assertIn("warp_size", value, "the Target must state its own role-slot width")
                self.assertEqual(Target.load(path).warp_size, value["warp_size"])

    def test_an_undeclared_width_is_refused_and_never_substituted(self):
        for name in ("sm_100a", "apple_gpu_family7"):
            value = document(name)
            del value["warp_size"]
            with self.subTest(target=name), self.assertRaisesRegex(TargetParseError, "target.warp_size"):
                Target.from_dict(value)

    def test_a_width_its_own_limits_cannot_hold_is_refused_naming_both(self):
        value = document("sm_100a")
        # 32 slots of 64 threads cannot fit in the 1024 threads the same document declares.
        value["warp_size"] = 64
        with self.assertRaisesRegex(TargetParseError, "warp_size disagrees with"):
            Target.from_dict(value)

    def test_a_declared_width_reaches_the_shared_residency_computation(self):
        schedule = Schedule.from_dict(json.loads(
            (ROOT / "corpus/schedules/atomic-reservation-b8-smoke.json").read_text(encoding="utf-8")))
        narrow = Target.from_dict(document("sm_100a"))
        wide = document("sm_100a")
        wide["warp_size"] = 64
        wide["resource_limits"] = dict(wide["resource_limits"], maximum_warps_per_cta=16)
        wide = Target.from_dict(wide)
        threads = {bound.resource: bound for bound in residency_upper_bound(schedule, narrow).bounds}
        doubled = {bound.resource: bound for bound in residency_upper_bound(schedule, wide).bounds}
        self.assertEqual(threads["threads"].per_cta * 2, doubled["threads"].per_cta)
        self.assertEqual(threads["threads"].ctas, doubled["threads"].ctas * 2)

    def test_the_width_is_not_a_shared_constant_any_more(self):
        source = (ROOT / "src/open_cake_ir/compiler/target.py").read_text(encoding="utf-8")
        self.assertNotIn("def warp_size", source, "warp_size must be a declared field, not a property")


if __name__ == "__main__":
    unittest.main()
