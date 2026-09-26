"""Two independent B producer rings ahead of ordered H64 carried state."""
from __future__ import annotations

import unittest

from open_cake_ir.compiler import Schedule
from open_cake_ir.compiler.backends import native_cuda
from open_cake_ir.compiler.verifier import verify
from tests.contracts.test_native_kda_chunk256 import document as single_stage_document
from tests.contracts.test_native_two_phase_k128 import target


_STAGED = {"base_stage", "query_stage", "correction_stage", "output_stage"}
_ALLOCATIONS = {"base_smem", "query_smem", "correction_smem", "output_smem"}


def document() -> dict:
    value = single_stage_document()
    value["schedule_id"] = "native-k128-v-beta-h64-prefetch2"
    value["lowering"]["entry_point"] = "cake_k128_v_beta_h64_prefetch2"
    for pipeline in value["pipelines"]:
        pipeline["stages"] = 2
    for buffer in value["buffers"]:
        if buffer["name"] in _STAGED:
            buffer["stages"] = 2
    for allocation in value["allocations"]:
        if allocation["name"] in _ALLOCATIONS:
            allocation["size_bytes"] *= 2
    value["tile_loops"][0]["range_options"]["num_stages"] = 2
    return value


class NativeKdaTwoStagePrefetch(unittest.TestCase):
    def test_two_slot_ring_keeps_chunk_state_order(self):
        schedule = Schedule.from_dict(document())
        self.assertEqual([f for f in verify(schedule, target()) if f.blocks_lowering], [])
        self.assertEqual(native_cuda.preflight(schedule, target()), ())
        source = native_cuda.emit(schedule, target()).source
        self.assertIn("CAKE_NATIVE_PREFETCH_2STAGE", source)
        self.assertIn("it0<256", source)
        self.assertIn("(it0+1)<256", source)
        self.assertIn("cake_wait(bar0, (it0&1))", source)
        self.assertIn("cake_wait(bar2, (it0&1))", source)
        self.assertIn("cake_wait(bar6, (it0&1))", source)
        self.assertIn("cake_wait(bar1+(it0&1), ((it0/2)&1))", source)

    def test_unmatched_stage_counts_are_refused_by_the_stage_owner(self):
        value = document()
        value["pipelines"][1]["stages"] = 1
        schedule = Schedule.from_dict(value)
        self.assertIn("NATIVE_CARRIED_PIPELINE_STAGES",
                      {f.code for f in native_cuda.preflight(schedule, target())})

    def test_three_slots_have_no_qualification(self):
        value = document()
        for pipeline in value["pipelines"]:
            pipeline["stages"] = 3
        for buffer in value["buffers"]:
            if buffer["name"] in _STAGED:
                buffer["stages"] = 3
        for allocation in value["allocations"]:
            if allocation["name"] in _ALLOCATIONS:
                allocation["size_bytes"] = allocation["size_bytes"] // 2 * 3
        value["tile_loops"][0]["range_options"]["num_stages"] = 3
        schedule = Schedule.from_dict(value)
        self.assertIn("NATIVE_CARRIED_PIPELINE_STAGES",
                      {f.code for f in native_cuda.preflight(schedule, target())})


if __name__ == "__main__":
    unittest.main()
