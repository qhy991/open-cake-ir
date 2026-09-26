"""The MMA-owned P tile needs a distinct slot while the next chunk runs."""
from __future__ import annotations

import unittest

from open_cake_ir.compiler import Schedule
from open_cake_ir.compiler.backends import native_cuda
from open_cake_ir.compiler.backends.native_cuda import _role_carried_domain
from open_cake_ir.compiler.verifier import verify
from tests.contracts.test_native_kda_role_pipeline import document as role_document
from tests.contracts.test_native_two_phase_k128 import target


def document() -> dict:
    value = role_document()
    value["schedule_id"] = "native-k128-v-beta-h64-p-double-slot"
    value["lowering"]["entry_point"] = "cake_k128_v_beta_h64_p_double_slot"
    next(b for b in value["buffers"] if b["name"] == "p_stage")["stages"] = 2
    next(a for a in value["allocations"] if a["name"] == "p_smem")["size_bytes"] = 4096
    return value


class NativeKdaPDoubleSlot(unittest.TestCase):
    def test_producer_and_solve_use_the_same_parity_slot(self):
        schedule = Schedule.from_dict(document())
        self.assertEqual([f for f in verify(schedule, target()) if f.blocks_lowering], [])
        self.assertEqual(native_cuda.preflight(schedule, target()), ())
        self.assertTrue(_role_carried_domain(schedule, target(), schedule.tile_loops[0]))
        source = native_cuda.emit(schedule, target()).source
        load = source[source.index("// CAKE_OP: load_p"):source.index("// CAKE_OP: mma_correction")]
        solve = source[source.index("// CAKE_OP: solve"):source.index("// CAKE_OP: round_updates")]
        self.assertIn("CAKE_NATIVE_VECTOR_P_STAGE", load)
        self.assertIn("+ ((it0&1))*2048", load)
        self.assertIn("+ ((it0&1))*2048", solve)
        self.assertIn("cake_wait(bar3, (it0&1))", solve)

    def test_three_slots_are_refused_by_the_role_mapping(self):
        value = document()
        next(b for b in value["buffers"] if b["name"] == "p_stage")["stages"] = 3
        next(a for a in value["allocations"] if a["name"] == "p_smem")["size_bytes"] = 6144
        schedule = Schedule.from_dict(value)
        self.assertFalse(_role_carried_domain(schedule, target(), schedule.tile_loops[0]))
        self.assertIn("NATIVE_ROLE_PIPELINE_DOMAIN",
                      {f.code for f in native_cuda.preflight(schedule, target())})

    def test_short_allocation_is_rejected_by_storage_analysis(self):
        value = document()
        next(a for a in value["allocations"] if a["name"] == "p_smem")["size_bytes"] = 2048
        schedule = Schedule.from_dict(value)
        self.assertIn("BUFFER_ALLOCATION_OVERFLOW",
                      {f.code for f in verify(schedule, target()) if f.blocks_lowering})


if __name__ == "__main__":
    unittest.main()
