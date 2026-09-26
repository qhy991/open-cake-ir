"""Bounded B300 row-solve emission with an explicit P stage handoff."""
from __future__ import annotations

import copy
import unittest

from open_cake_ir.compiler import Schedule
from open_cake_ir.compiler.backends import native_cuda
from open_cake_ir.compiler.verifier import verify
from tests.contracts.test_forward_substitute_ir import document, target


def native_document() -> dict:
    value = document()
    value["schedule_id"] = "native-forward-substitute-bf16-c32"
    value["lowering"] = {"backend": "native_cuda", "entry_point": "cake_forward_substitute_bf16_c32"}
    value["roles"].append({"name": "copy", "execution_groups": [4]})
    value["allocations"].append({
        "name": "p_smem", "space": "shared", "size_bytes": 2048,
    })
    p_tile = next(item for item in value["buffers"] if item["name"] == "p_tile")
    p_tile.update(space="shared", allocation="p_smem", swizzle="swizzle_64b")
    value["barriers"].append({
        "name": "p_ready", "count": 1, "producers": ["copy"],
        "consumers": ["compute"], "mechanism": "mbarrier",
    })
    load = next(item for item in value["operations"] if item["id"] == "load_p")
    load.update(role="copy", signals=["p_ready"])
    solve = next(item for item in value["operations"] if item["id"] == "solve")
    solve["waits"] = ["p_ready"]
    return value


class NativeForwardSubstitute(unittest.TestCase):
    def test_typed_stage_and_solve_emit_explicit_order(self):
        schedule = Schedule.from_dict(native_document())
        self.assertEqual([f for f in verify(schedule, target())
                          if f.blocks_lowering], [])
        self.assertEqual(native_cuda.preflight(schedule, target()), ())
        source = native_cuda.emit(schedule, target()).source
        self.assertIn("cake_init(bar0, 1)", source)
        self.assertIn("cake_arrive(bar0)", source)
        self.assertIn("cake_wait(bar0, 0)", source)
        self.assertIn("float u0 = b4[0];", source)
        self.assertIn("float u31 = b4[31];", source)
        self.assertEqual(source.count("= __fmaf_rn(__bfloat162float("), 496)
        self.assertNotIn("for (int prior=", source)
        self.assertIn("cake_inval(bar0)", source)

    def test_wrong_stage_or_wait_is_refused_by_owner(self):
        value = native_document()
        next(item for item in value["operations"] if item["id"] == "solve")["waits"] = []
        self.assertIn("NATIVE_FORWARD_SOLVE_OWNER",
                      {f.code for f in native_cuda.preflight(Schedule.from_dict(value), target())})
        value = native_document()
        next(item for item in value["buffers"] if item["name"] == "p_tile")["swizzle"] = "swizzle_128b"
        codes = {f.code for f in native_cuda.preflight(Schedule.from_dict(value), target())}
        self.assertIn("NATIVE_FORWARD_SOLVE_DOMAIN", codes)
        value = native_document()
        next(item for item in value["barriers"] if item["name"] == "p_ready")["count"] = 2
        self.assertIn("NATIVE_SOLVE_P_STAGE",
                      {f.code for f in native_cuda.preflight(Schedule.from_dict(value), target())})

    def test_unrelated_shared_load_is_not_admitted_as_a_solve_stage(self):
        value = native_document()
        extra = copy.deepcopy(next(item for item in value["operations"] if item["id"] == "load_p"))
        extra["id"] = "extra_load"
        extra["writes"] = ["extra_stage"]
        value["operations"].insert(1, extra)
        stage = copy.deepcopy(next(item for item in value["buffers"] if item["name"] == "p_tile"))
        stage.update(name="extra_stage", allocation="extra_smem")
        value["buffers"].append(stage)
        value["allocations"].append({"name": "extra_smem", "space": "shared", "size_bytes": 2048})
        self.assertIn("NATIVE_SHARED_LOAD_SCOPE",
                      {f.code for f in native_cuda.preflight(Schedule.from_dict(value), target())})


if __name__ == "__main__":
    unittest.main()
