"""Exact BF16 M64 TMEM atoms and their carried-state analysis boundary."""
from __future__ import annotations

from copy import deepcopy
import unittest

from jsonschema import Draft202012Validator

from open_cake_ir.compiler import Schedule
from open_cake_ir.compiler.backends import native_cuda
from open_cake_ir.compiler.schema import schedule_schema
from open_cake_ir.compiler.verifier import verify
from tests.contracts.test_native_kda_prepared_beta_bridge import document as base_document
from tests.contracts.test_native_two_phase_k128 import target


VALUE_ROWS = {
    "initial_reg", "state_tmem", "base_acc", "base_pred", "rhs_adjusted",
    "updates", "updates_bf", "updates_tmem", "correction_acc", "correction",
    "next_state", "state_copy", "state_fp", "decayed_state", "state_sum",
    "query_acc", "output_acc", "query_reg", "output_reg", "output_sum",
    "output_scaled", "output_bf16", "v_row_major", "v_fp32", "rhs_beta",
}


def m64_document() -> dict:
    value = base_document()
    value["schedule_id"] = "kda-m64-tmem-copy-atom-contract"
    value["lowering"]["entry_point"] = "cake_kda_m64_tmem_copy_atom_contract"
    value["program_map"]["axes"].append({
        "name": "value_slice", "axis": 1, "buffer": "initial_state",
        "dimension": 1, "tile": 64,
    })
    for buffer in value["buffers"]:
        if buffer["name"] in VALUE_ROWS:
            buffer["shape"][0] = 64
        if buffer["name"] in ("public_output_tile", "v_token_major"):
            buffer["shape"][1] = 64
    for operation in value["operations"]:
        if operation["kind"] == "mma":
            operation["parameters"]["tile_shape"][0] = 64
            operation["parameters"]["instruction"]["shape"][0] = 64
        if operation["kind"] == "tmem_store":
            atom = operation["parameters"]["destination_atom"]
            atom["op"] = "tcgen05.St16x256b"
            atom["repetition"] = 2 if operation["id"] == "publish_updates" else 8
        if operation["id"] == "read_state":
            operation["parameters"]["source_atom"] = {
                "op": "tcgen05.Ld16x256b", "repetition": 8,
            }
    for access in value["access_maps"]:
        if access["operation"] in ("load_initial", "store_final"):
            access["indices"][1] = {"source": "program_tile", "name": "value_slice"}
        if access["operation"] in ("load_v", "store_output"):
            access["indices"][3] = {"source": "program_tile", "name": "value_slice"}
    return value


def blocked_codes(value: dict) -> set[str]:
    return {finding.code for finding in verify(Schedule.from_dict(value), target())
            if finding.blocks_lowering}


class KdaM64TmemCopyAtoms(unittest.TestCase):
    def test_m64_four_warp_copy_atoms_complete_shared_analysis(self):
        value = m64_document()
        Draft202012Validator(schedule_schema()).validate(value)
        self.assertEqual(blocked_codes(value), set())
        # The NVIDIA emitter still owns M64 lane mapping and has not admitted it.
        self.assertIn("NATIVE_TMEM_STORE_CONTRACT",
                      {finding.code for finding in native_cuda.preflight(
                          Schedule.from_dict(value), target())})

    def test_old_atom_cannot_claim_a_64_row_state_tile(self):
        value = m64_document()
        next(op for op in value["operations"] if op["id"] == "store_initial")[
            "parameters"]["destination_atom"]["op"] = "tcgen05.St32x32b"
        self.assertIn("TMEM_STORE_CONTRACT", blocked_codes(value))

    def test_update_tile_uses_x2_not_the_full_state_x8(self):
        value = m64_document()
        next(op for op in value["operations"] if op["id"] == "publish_updates")[
            "parameters"]["destination_atom"]["repetition"] = 8
        self.assertIn("TMEM_STORE_CONTRACT", blocked_codes(value))

    def test_state_read_requires_the_exact_64_by_128_x8_atom(self):
        value = m64_document()
        next(op for op in value["operations"] if op["id"] == "read_state")[
            "parameters"]["source_atom"]["repetition"] = 2
        self.assertIn("TMEM_LOAD_ATOM", blocked_codes(value))

    def test_carried_state_still_requires_four_arrivals(self):
        value = m64_document()
        next(b for b in value["barriers"] if b["name"] == "state_ready")["count"] = 2
        self.assertIn("CARRIED_TMEM_BARRIER", blocked_codes(value))

    def test_unlisted_repetition_is_refused_by_authoring_schema(self):
        value = m64_document()
        next(op for op in value["operations"] if op["id"] == "store_initial")[
            "parameters"]["destination_atom"]["repetition"] = 4
        self.assertTrue(list(Draft202012Validator(schedule_schema()).iter_errors(value)))

    def test_m128_control_remains_valid(self):
        self.assertEqual(blocked_codes(deepcopy(base_document())), set())


if __name__ == "__main__":
    unittest.main()
