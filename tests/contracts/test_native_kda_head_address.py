"""One head per CTA for bounded KDA chunk/state address lowering."""
from __future__ import annotations

import unittest

from open_cake_ir.compiler import Schedule
from open_cake_ir.compiler.backends import native_cuda
from open_cake_ir.compiler.verifier import verify
from tests.contracts.test_native_kda_v_beta_input import document as one_head_document
from tests.contracts.test_native_two_phase_k128 import target


def document(heads: int = 64) -> dict:
    value = one_head_document()
    value["schedule_id"] = "native-k128-v-beta-head-address"
    value["lowering"]["entry_point"] = "cake_k128_v_beta_head_address"
    value.pop("grid", None)
    value["program_map"] = {"axes": [
        {"name": "head", "axis": 0, "buffer": "b_base",
         "dimension": 1, "tile": 1},
    ]}
    for buffer in value["buffers"]:
        if buffer["space"] != "global":
            continue
        name = buffer["name"]
        if name in {"initial_state", "final_state"}:
            buffer["shape"].insert(0, heads)
        elif name in {"v_public", "chunk_output", "beta_gate"}:
            buffer["shape"].insert(2, heads)
        else:
            buffer["shape"].insert(1, heads)
    for access in value["access_maps"]:
        # Upstream fixtures reuse axis dictionaries across accesses; detach before
        # assigning the physical axis numbers of each head-aware buffer.
        access["indices"] = [dict(component) for component in access["indices"]]
        name = access["buffer"]
        if name in {"initial_state", "final_state"}:
            position = 0
        elif name in {"v_public", "chunk_output", "beta_gate"}:
            position = 2
        else:
            position = 1
        access["indices"].insert(position, {"source": "program", "name": "head"})
        for dimension, component in enumerate(access["indices"]):
            if component["source"] == "dimension":
                component["dimension"] = dimension
    return value


class NativeKdaHeadAddress(unittest.TestCase):
    def test_rank_four_head_mapping_is_explicit_and_emittable(self):
        schedule = Schedule.from_dict(document())
        self.assertEqual([f for f in verify(schedule, target()) if f.blocks_lowering], [])
        self.assertEqual(native_cuda.preflight(schedule, target()), ())
        source = native_cuda.emit(schedule, target()).source
        self.assertIn("cake_tma4(", source)
        self.assertIn("dim3(64,1,1)", source)
        self.assertIn("int(blockIdx.x)", source)
        self.assertIn("cuuint64_t dims[4] = {32, 128, 64, 2}", source)
        self.assertIn("cuuint32_t box[4] = {32, 128, 1, 1}", source)

    def test_wrong_head_selector_is_refused(self):
        value = document()
        access = next(a for a in value["access_maps"] if a["operation"] == "load_base")
        access["indices"][1] = {"source": "dimension", "dimension": 1}
        self.assertIn("NATIVE_CARRIED_MMA_DOMAIN", {
            finding.code for finding in native_cuda.preflight(Schedule.from_dict(value), target())
        })

    def test_rank_five_descriptor_is_refused(self):
        value = document()
        buffer = next(b for b in value["buffers"] if b["name"] == "b_base")
        buffer["shape"].insert(1, 1)
        access = next(a for a in value["access_maps"] if a["operation"] == "load_base")
        access["indices"].insert(1, {"source": "dimension", "dimension": 1})
        self.assertIn("NATIVE_TMA_DESCRIPTOR", {
            finding.code for finding in native_cuda.preflight(Schedule.from_dict(value), target())
        })

    def test_other_head_extent_is_not_inferred_from_h64(self):
        value = document(32)
        schedule = Schedule.from_dict(value)
        self.assertEqual([f for f in verify(schedule, target()) if f.blocks_lowering], [])
        codes = {f.code for f in native_cuda.preflight(schedule, target())}
        self.assertIn("NATIVE_CARRIED_MMA_DOMAIN", codes)
        self.assertIn("NATIVE_TMA_DESCRIPTOR", codes)

    def test_output_must_name_the_head_owner(self):
        value = document()
        access = next(a for a in value["access_maps"] if a["operation"] == "store_output")
        access["indices"][2] = {"source": "dimension", "dimension": 2}
        codes = {f.code for f in native_cuda.preflight(Schedule.from_dict(value), target())}
        self.assertIn("NATIVE_TRANSPOSE_STORE_DOMAIN", codes)
        self.assertIn("NATIVE_STORE_PROGRAM_OWNERSHIP", codes)


if __name__ == "__main__":
    unittest.main()
