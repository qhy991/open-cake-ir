"""A bounded H64/T8192 chunk loop with explicit public V and output axes."""
from __future__ import annotations

import unittest

from open_cake_ir.compiler import Schedule
from open_cake_ir.compiler.backends import native_cuda
from open_cake_ir.compiler.verifier import verify
from tests.contracts.test_native_kda_head_address import document as h64_document
from tests.contracts.test_native_two_phase_k128 import target


def document(chunks: int = 256) -> dict:
    value = h64_document()
    value["schedule_id"] = f"native-k128-v-beta-h64-chunk{chunks}"
    value["lowering"]["entry_point"] = f"cake_k128_v_beta_h64_chunk{chunks}"
    for buffer in value["buffers"]:
        if buffer["space"] == "global" and buffer["name"] not in {
                "initial_state", "final_state"}:
            buffer["shape"][0] = chunks
    return value


class NativeKdaChunk256(unittest.TestCase):
    def test_full_fixed_chunk_count_is_explicit_and_emittable(self):
        schedule = Schedule.from_dict(document())
        self.assertEqual([f for f in verify(schedule, target()) if f.blocks_lowering], [])
        self.assertEqual(native_cuda.preflight(schedule, target()), ())
        source = native_cuda.emit(schedule, target()).source
        self.assertIn("it0<256", source)
        self.assertIn("dim3(64,1,1)", source)
        self.assertIn("cake_tma4(", source)

    def test_an_unqualified_chunk_count_is_refused(self):
        schedule = Schedule.from_dict(document(3))
        codes = {finding.code for finding in native_cuda.preflight(schedule, target())}
        self.assertIn("NATIVE_TRANSPOSE_LOAD_DOMAIN", codes)
        self.assertIn("NATIVE_TRANSPOSE_STORE_DOMAIN", codes)


if __name__ == "__main__":
    unittest.main()
