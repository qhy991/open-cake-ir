"""Consume the qualified KDA preparation beta view without a host transpose."""
from __future__ import annotations

import unittest

from open_cake_ir.compiler import Schedule
from open_cake_ir.compiler.backends import native_cuda
from open_cake_ir.compiler.verifier import verify
from tests.contracts.test_native_kda_p_double_slot import document as two_slot_document
from tests.contracts.test_native_two_phase_k128 import target


def document() -> dict:
    value = two_slot_document()
    value["schedule_id"] = "native-k128-v-beta-h64-prepared-bridge"
    value["lowering"]["entry_point"] = "cake_k128_v_beta_h64_prepared_bridge"
    next(b for b in value["buffers"] if b["name"] == "beta_gate")["shape"] = [256, 64, 32]
    access = next(a for a in value["access_maps"]
                  if a["operation"] == "load_beta" and a["buffer"] == "beta_gate")
    access["indices"] = [
        {"source": "loop", "name": "chunk_index"},
        {"source": "program", "name": "head"},
        {"source": "dimension", "dimension": 2},
    ]
    return value


class NativeKdaPreparedBetaBridge(unittest.TestCase):
    def test_prepared_beta_is_read_in_head_then_token_order(self):
        schedule = Schedule.from_dict(document())
        self.assertEqual([f for f in verify(schedule, target()) if f.blocks_lowering], [])
        self.assertEqual(native_cuda.preflight(schedule, target()), ())
        source = native_cuda.emit(schedule, target()).source
        beta_load = source[source.index("// CAKE_OP: load_beta"):
                           source.index("// CAKE_OP: apply_beta")]
        self.assertIn("(it0) * 2048 + (int(blockIdx.x)) * 32 + ((0 + col)) * 1",
                      beta_load)
        self.assertIn("+ ((it0&1))*2048", source[source.index("// CAKE_OP: load_p"):
                                                     source.index("// CAKE_OP: mma_correction")])

    def test_prepared_access_rejects_the_old_token_head_shape(self):
        value = document()
        next(b for b in value["buffers"] if b["name"] == "beta_gate")["shape"] = [256, 32, 64]
        schedule = Schedule.from_dict(value)
        codes = {f.code for f in verify(schedule, target()) if f.blocks_lowering}
        self.assertIn("ACCESS_PROGRAM_EXTENT_MISMATCH", codes)
        self.assertIn("ACCESS_TILE_MISMATCH", codes)


if __name__ == "__main__":
    unittest.main()
