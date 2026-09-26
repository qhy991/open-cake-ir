"""BF16 carried-state TMEM read after a typed two-phase K128 chunk."""
from __future__ import annotations

import copy
import unittest

from open_cake_ir.compiler import Schedule
from open_cake_ir.compiler.backends import native_cuda
from open_cake_ir.compiler.verifier import verify
from tests.contracts.test_native_two_phase_k128 import document as k128_document, target


def document() -> dict:
    value = k128_document()
    value["schedule_id"] = "native-two-phase-k128-bf16-state-read"
    value["lowering"]["entry_point"] = "cake_two_phase_k128_bf16_state_read"
    value["buffers"].append({
        "name": "state_copy", "space": "register", "dtype": "bf16",
        "shape": [128, 128], "mode": "scratch",
    })
    read = {
        "id": "read_state", "kind": "load", "role": "compute",
        "reads": ["state_tmem"], "writes": ["state_copy"],
        "waits": ["state_ready"], "depends_on": ["read_correction"],
        "parameters": {
            "movement": "tmem",
            "source_atom": {"op": "tcgen05.Ld32x32b", "repetition": 16},
        },
    }
    value["operations"].insert(
        next(i for i, op in enumerate(value["operations"]) if op["id"] == "round_state"),
        read,
    )
    body = value["tile_loops"][0]["body"]
    body.insert(body.index("round_state"), "read_state")
    next(b for b in value["barriers"] if b["name"] == "state_ready")[
        "consumers"
    ].append("compute")
    return value


class NativeBf16TmemRead(unittest.TestCase):
    def test_carried_state_reads_packed_words_at_the_current_phase(self):
        schedule = Schedule.from_dict(document())
        self.assertEqual([f for f in verify(schedule, target()) if f.blocks_lowering], [])
        self.assertEqual(native_cuda.preflight(schedule, target()), ())
        source = native_cuda.emit(schedule, target()).source
        read = source[source.index("// CAKE_OP: read_state"):]
        self.assertIn("cake_wait(bar0, (it0&1));", read)
        self.assertIn("tcgen05.ld.sync.aligned.32x32b.x16.b32", read)
        self.assertIn('"=r"(word0)', read)
        self.assertIn("col/2", read)
        self.assertIn("__ushort_as_bfloat16(uint16_t(word0 >> 16))", read)

    def test_missing_phase_and_wider_than_state_atom_are_refused(self):
        value = document()
        next(op for op in value["operations"] if op["id"] == "read_state")[
            "waits"
        ] = []
        schedule = Schedule.from_dict(value)
        self.assertIn("CARRIED_TMEM_BARRIER",
                      {f.code for f in verify(schedule, target())})
        self.assertIn("NATIVE_TMEM_BF16_STATE",
                      {f.code for f in native_cuda.preflight(schedule, target())})

        value = document()
        next(op for op in value["operations"] if op["id"] == "read_state")[
            "parameters"
        ]["source_atom"]["repetition"] = 128
        schedule = Schedule.from_dict(value)
        self.assertIn("TMEM_LOAD_BF16_PACKING",
                      {f.code for f in verify(schedule, target())})
        self.assertIn("NATIVE_TMEM_BF16_STATE",
                      {f.code for f in native_cuda.preflight(schedule, target())})

    def test_unrelated_tmem_store_is_not_a_carried_state_read(self):
        value = copy.deepcopy(document())
        read = next(op for op in value["operations"] if op["id"] == "read_state")
        read["reads"] = ["updates_tmem"]
        read["waits"] = ["u_ready"]
        schedule = Schedule.from_dict(value)
        self.assertIn("NATIVE_TMEM_BF16_STATE",
                      {f.code for f in native_cuda.preflight(schedule, target())})


if __name__ == "__main__":
    unittest.main()
