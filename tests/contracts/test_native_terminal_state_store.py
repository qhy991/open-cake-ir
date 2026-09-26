"""Only the final observable H64 state overwrite may skip earlier loop stores."""
from __future__ import annotations

import unittest

from open_cake_ir.compiler import Schedule
from open_cake_ir.compiler.backends import native_cuda
from open_cake_ir.compiler.backends.native_cuda import _last_chunk_terminal_store
from open_cake_ir.compiler.verifier import verify
from tests.contracts.test_native_kda_chunk256 import document as full_document
from tests.contracts.test_native_kda_head_address import document as two_chunk_document
from tests.contracts.test_native_two_phase_k128 import target


class NativeTerminalStateStore(unittest.TestCase):
    def test_only_last_of_256_full_state_overwrites_is_emitted(self):
        schedule = Schedule.from_dict(full_document())
        self.assertEqual([f for f in verify(schedule, target()) if f.blocks_lowering], [])
        self.assertEqual(native_cuda.preflight(schedule, target()), ())
        self.assertTrue(_last_chunk_terminal_store(
            schedule, target(), schedule.operation("store_final")))
        source = native_cuda.emit(schedule, target()).source
        marker = source.index("CAKE_NATIVE_TERMINAL_STATE_STORE")
        self.assertIn("if (it0 == 255)", source[marker:marker+175])
        self.assertEqual(source.count("CAKE_NATIVE_TERMINAL_STATE_STORE"), 1)

    def test_two_chunk_control_keeps_every_iteration_store(self):
        schedule = Schedule.from_dict(two_chunk_document())
        self.assertFalse(_last_chunk_terminal_store(
            schedule, target(), schedule.operation("store_final")))
        self.assertNotIn("CAKE_NATIVE_TERMINAL_STATE_STORE",
                         native_cuda.emit(schedule, target()).source)

    def test_an_in_kernel_reader_prevents_elision(self):
        value = full_document()
        next(op for op in value["operations"] if op["id"] == "load_initial")[
            "reads"] = ["final_state"]
        schedule = Schedule.from_dict(value)
        self.assertFalse(_last_chunk_terminal_store(
            schedule, target(), schedule.operation("store_final")))

    def test_a_loop_or_nonowned_address_prevents_elision(self):
        value = full_document()
        access = next(a for a in value["access_maps"]
                      if a["operation"] == "store_final")
        access["indices"][0] = {"source": "dimension", "dimension": 0}
        schedule = Schedule.from_dict(value)
        self.assertFalse(_last_chunk_terminal_store(
            schedule, target(), schedule.operation("store_final")))


if __name__ == "__main__":
    unittest.main()
