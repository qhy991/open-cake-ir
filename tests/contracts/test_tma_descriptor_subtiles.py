"""A TMA box may repeatedly fill the final axis; each emitter owns that route."""

import json
from pathlib import Path
import unittest

from open_cake_ir.compiler.backends import cutedsl, native_cuda
from open_cake_ir.compiler.ir import Schedule
from open_cake_ir.compiler.target import Target
from open_cake_ir.compiler.verifier import verify


ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / "corpus/schedules/native-gemm-bias-sm103a.json"
CUTE = ROOT / "corpus/schedules/flash-kmeans-assignment-full.json"


def document(path, operation, box):
    result = json.loads(path.read_text())
    next(op for op in result["operations"] if op["id"] == operation)["parameters"]["descriptor_box"] = box
    return Schedule.from_dict(result)


class TmaDescriptorSubtiles(unittest.TestCase):
    def test_even_k_subtile_is_structural_but_existing_native_route_refuses_it(self):
        target = Target.load(ROOT / "compiler/targets/sm_103a.json")
        schedule = document(NATIVE, "load_a", [128, 32])
        self.assertNotIn("TMA_DESCRIPTOR_MISMATCH", {f.code for f in verify(schedule, target)})
        self.assertIn("NATIVE_TMA_DESCRIPTOR", {f.code for f in native_cuda.preflight(schedule, target)})

    def test_cute_refuses_the_subtile_it_does_not_emit(self):
        target = Target.load(ROOT / "compiler/targets/sm_100a.json")
        schedule = document(CUTE, "load_tokens", [128, 32])
        self.assertNotIn("TMA_DESCRIPTOR_MISMATCH", {f.code for f in verify(schedule, target)})
        self.assertIn("CUTE_LOAD_DESCRIPTOR_TILING_UNSUPPORTED",
                      {f.code for f in cutedsl.preflight(schedule, target)})

    def test_row_split_and_nondividing_k_box_are_hardware_findings(self):
        target = Target.load(ROOT / "compiler/targets/sm_103a.json")
        for box in ([64, 64], [128, 48]):
            with self.subTest(box=box):
                schedule = document(NATIVE, "load_a", box)
                self.assertIn("TMA_DESCRIPTOR_MISMATCH", {f.code for f in verify(schedule, target)})
