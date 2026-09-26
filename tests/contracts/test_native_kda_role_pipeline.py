"""One producer loop, ordered MMA loop and compute loop for H64 carried state."""
from __future__ import annotations

import unittest

from open_cake_ir.compiler import Schedule
from open_cake_ir.compiler.backends import native_cuda
from open_cake_ir.compiler.backends.native_cuda import _role_carried_domain
from open_cake_ir.compiler.verifier import verify
from tests.contracts.test_native_kda_two_stage_prefetch import document as prefetched_document
from tests.contracts.test_native_two_phase_k128 import target


def document() -> dict:
    value = prefetched_document()
    value["schedule_id"] = "native-k128-v-beta-h64-role-pipeline"
    value["lowering"]["entry_point"] = "cake_k128_v_beta_h64_role_pipeline"
    next(op for op in value["operations"] if op["id"] == "load_p")["role"] = "mma"
    next(b for b in value["barriers"] if b["name"] == "p_ready")["producers"] = ["mma"]
    return value


class NativeKdaRolePipeline(unittest.TestCase):
    def test_roles_advance_in_separate_chunk_loops(self):
        schedule = Schedule.from_dict(document())
        self.assertEqual([f for f in verify(schedule, target()) if f.blocks_lowering], [])
        self.assertEqual(native_cuda.preflight(schedule, target()), ())
        self.assertTrue(_role_carried_domain(schedule, target(), schedule.tile_loops[0]))
        source = native_cuda.emit(schedule, target()).source
        self.assertIn("CAKE_NATIVE_ROLE_CARRIED_PIPELINE", source)
        self.assertIn("CAKE_ROLE_COPY_LOOP", source)
        self.assertIn("CAKE_ROLE_MMA_LOOP", source)
        self.assertIn("CAKE_ROLE_COMPUTE_LOOP", source)
        self.assertIn("cake_wait(bar0, (it0&1))", source)
        self.assertIn("cake_wait(bar2, (it0&1))", source)
        self.assertIn("CAKE_NATIVE_TERMINAL_STATE_STORE", source)
        self.assertNotIn("__syncthreads()", source[
            source.index("CAKE_ROLE_COPY_LOOP"):source.index("// CAKE_OP: store_final")])

    def test_copy_owned_p_stage_keeps_the_qualified_two_slot_control(self):
        schedule = Schedule.from_dict(prefetched_document())
        self.assertFalse(_role_carried_domain(schedule, target(), schedule.tile_loops[0]))
        source = native_cuda.emit(schedule, target()).source
        self.assertNotIn("CAKE_NATIVE_ROLE_CARRIED_PIPELINE", source)

    def test_extra_p_reader_disables_the_role_rewrite(self):
        value = document()
        next(op for op in value["operations"] if op["id"] == "cast_v")[
            "reads"] = ["p_stage"]
        schedule = Schedule.from_dict(value)
        self.assertFalse(_role_carried_domain(schedule, target(), schedule.tile_loops[0]))
        self.assertIn("NATIVE_ROLE_PIPELINE_DOMAIN",
                      {f.code for f in native_cuda.preflight(schedule, target())})


if __name__ == "__main__":
    unittest.main()
