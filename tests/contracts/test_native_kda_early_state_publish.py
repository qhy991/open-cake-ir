"""Publish carried state after TMEM output reads, before token-major output store."""
from __future__ import annotations

import unittest

from open_cake_ir.compiler import Schedule
from open_cake_ir.compiler.backends import native_cuda
from open_cake_ir.compiler.verifier import verify
from tests.contracts.test_native_kda_role_pipeline import document as role_document
from tests.contracts.test_native_two_phase_k128 import target


def document() -> dict:
    value = role_document()
    value["schedule_id"] = "native-k128-v-beta-h64-early-state"
    value["lowering"]["entry_point"] = "cake_k128_v_beta_h64_early_state"
    moved = ("transpose_public_output", "store_output")
    operations = value["operations"]
    selected = [op for op in operations if op["id"] in moved]
    operations[:] = [op for op in operations if op["id"] not in moved]
    index = next(i for i, op in enumerate(operations)
                 if op["id"] == "publish_state") + 1
    operations[index:index] = selected
    body = value["tile_loops"][0]["body"]
    body[:] = [name for name in body if name not in moved]
    index = body.index("publish_state") + 1
    body[index:index] = moved
    return value


class NativeKdaEarlyStatePublish(unittest.TestCase):
    def test_state_publish_can_precede_completed_output_writeback(self):
        schedule = Schedule.from_dict(document())
        self.assertEqual([f for f in verify(schedule, target()) if f.blocks_lowering], [])
        self.assertEqual(native_cuda.preflight(schedule, target()), ())
        source = native_cuda.emit(schedule, target()).source
        self.assertLess(source.index("// CAKE_OP: read_query"),
                        source.index("// CAKE_OP: publish_state"))
        self.assertLess(source.index("// CAKE_OP: read_output"),
                        source.index("// CAKE_OP: publish_state"))
        self.assertLess(source.index("// CAKE_OP: publish_state"),
                        source.index("// CAKE_OP: store_output"))
        self.assertIn("CAKE_NATIVE_TERMINAL_STATE_STORE", source)

    def test_state_cannot_be_published_before_query_tmem_readout(self):
        value = document()
        moved = ("read_output", "read_query", "combine_output", "scale_output",
                 "round_output", "transpose_public_output", "store_output")
        operations = value["operations"]
        selected = [op for op in operations if op["id"] in moved]
        operations[:] = [op for op in operations if op["id"] not in moved]
        index = next(i for i, op in enumerate(operations)
                     if op["id"] == "publish_state") + 1
        operations[index:index] = selected
        body = value["tile_loops"][0]["body"]
        body[:] = [name for name in body if name not in moved]
        index = body.index("publish_state") + 1
        body[index:index] = moved
        schedule = Schedule.from_dict(value)
        codes = {f.code for f in native_cuda.preflight(schedule, target())}
        self.assertIn("NATIVE_CARRIED_OUTPUT_READ_BEFORE_PUBLICATION", codes)


if __name__ == "__main__":
    unittest.main()
