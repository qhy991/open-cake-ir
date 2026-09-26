"""Compose a decayed K128 prior state with the carried correction result."""
from __future__ import annotations

import unittest

from open_cake_ir.compiler import Schedule
from open_cake_ir.compiler.backends import native_cuda
from open_cake_ir.compiler.performance.work import work_bound
from open_cake_ir.compiler.verifier import verify
from tests.contracts.test_native_bf16_tmem_read import document as bf16_document
from tests.contracts.test_native_two_phase_k128 import target


def document() -> dict:
    value = bf16_document()
    value["schedule_id"] = "native-two-phase-k128-decayed-state"
    value["lowering"]["entry_point"] = "cake_two_phase_k128_decayed_state"
    value["buffers"].extend([
        {"name": "prefix", "space": "global", "dtype": "fp32",
         "shape": [2, 128], "mode": "input"},
        {"name": "prefix_reg", "space": "register", "dtype": "fp32",
         "shape": [128], "mode": "scratch"},
        {"name": "state_fp", "space": "register", "dtype": "fp32",
         "shape": [128, 128], "mode": "scratch"},
        {"name": "decayed_state", "space": "register", "dtype": "fp32",
         "shape": [128, 128], "mode": "scratch"},
        {"name": "state_sum", "space": "register", "dtype": "fp32",
         "shape": [128, 128], "mode": "scratch"},
    ])
    added = [
        {"id": "load_prefix", "kind": "load", "role": "compute",
         "reads": ["prefix"], "writes": ["prefix_reg"],
         "depends_on": ["read_state"], "parameters": {"movement": "global"}},
        {"id": "cast_state", "kind": "cast", "role": "compute",
         "reads": ["state_copy"], "writes": ["state_fp"],
         "depends_on": ["read_state"], "parameters": {"to": "fp32"}},
        {"id": "scale_state", "kind": "elementwise", "role": "compute",
         "reads": ["state_fp", "prefix_reg"], "writes": ["decayed_state"],
         "depends_on": ["cast_state", "load_prefix"],
         "parameters": {"op": "mul", "broadcast_axis": 1}},
        {"id": "combine_state", "kind": "elementwise", "role": "compute",
         "reads": ["decayed_state", "correction"], "writes": ["state_sum"],
         "depends_on": ["scale_state", "read_correction"],
         "parameters": {"op": "add"}},
    ]
    operations = value["operations"]
    insert = next(i for i, op in enumerate(operations) if op["id"] == "round_state")
    operations[insert:insert] = added
    body = value["tile_loops"][0]["body"]
    insert = body.index("round_state")
    body[insert:insert] = [op["id"] for op in added]
    round_state = next(op for op in operations if op["id"] == "round_state")
    round_state["reads"] = ["state_sum"]
    round_state["depends_on"] = ["combine_state"]
    value["access_maps"].append({
        "operation": "load_prefix", "buffer": "prefix",
        "indices": [{"source": "loop", "name": "chunk_index"},
                    {"source": "dimension", "dimension": 1}],
        "boundary": "mask_tiled_axes",
    })
    return value


class NativeKdaDecayedState(unittest.TestCase):
    def test_explicit_prior_state_decay_precedes_bf16_publication(self):
        schedule = Schedule.from_dict(document())
        self.assertEqual([f for f in verify(schedule, target()) if f.blocks_lowering], [])
        self.assertEqual(native_cuda.preflight(schedule, target()), ())
        source = native_cuda.emit(schedule, target()).source
        for left, right in (("// CAKE_OP: read_state", "// CAKE_OP: scale_state"),
                            ("// CAKE_OP: scale_state", "// CAKE_OP: combine_state"),
                            ("// CAKE_OP: combine_state", "// CAKE_OP: publish_state")):
            self.assertLess(source.index(left), source.index(right))
        self.assertIn("cake_wait(bar0, (it0&1));", source)
        self.assertIsNotNone(work_bound(schedule))

    def test_missing_decay_input_has_a_data_edge_refusal(self):
        value = document()
        value["operations"] = [op for op in value["operations"]
                               if op["id"] != "load_prefix"]
        self.assertIn("BUFFER_UNPRODUCED",
                      {f.code for f in verify(Schedule.from_dict(value), target())})


if __name__ == "__main__":
    unittest.main()
