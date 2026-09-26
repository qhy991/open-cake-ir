"""Read public token-major V and FP32 beta into V-row-owned RHS registers."""
from __future__ import annotations

import unittest

from open_cake_ir.compiler import Schedule
from open_cake_ir.compiler.backends import native_cuda
from open_cake_ir.compiler.verifier import verify
from tests.contracts.test_native_kda_state_output import document as state_output_document
from tests.contracts.test_native_two_phase_k128 import target


def document() -> dict:
    value = state_output_document()
    value["schedule_id"] = "native-k128-v-beta-token-major"
    value["lowering"]["entry_point"] = "cake_k128_v_beta_token_major"
    buffers = value["buffers"]
    buffers[:] = [b for b in buffers if b["name"] not in {"rhs", "rhs_reg"}]
    buffers.extend([
        {"name": "public_output_tile", "space": "register", "dtype": "bf16",
         "shape": [32, 128], "mode": "scratch"},
        {"name": "v_public", "space": "global", "dtype": "bf16",
         "shape": [2, 32, 128], "mode": "input"},
        {"name": "v_token_major", "space": "register", "dtype": "bf16",
         "shape": [32, 128], "mode": "scratch"},
        {"name": "v_row_major", "space": "register", "dtype": "bf16",
         "shape": [128, 32], "mode": "scratch"},
        {"name": "v_fp32", "space": "register", "dtype": "fp32",
         "shape": [128, 32], "mode": "scratch"},
        {"name": "beta_gate", "space": "global", "dtype": "fp32",
         "shape": [2, 32], "mode": "input"},
        {"name": "beta_reg", "space": "register", "dtype": "fp32",
         "shape": [32], "mode": "scratch"},
        {"name": "rhs_beta", "space": "register", "dtype": "fp32",
         "shape": [128, 32], "mode": "scratch"},
    ])
    next(b for b in buffers if b["name"] == "chunk_output")["shape"] = [2, 32, 128]
    operations = value["operations"]
    body = value["tile_loops"][0]["body"]
    operations[:] = [op for op in operations if op["id"] != "load_rhs"]
    body.remove("load_rhs")

    def insert_before(anchor: str, added: list[dict]) -> None:
        index = next(i for i, op in enumerate(operations) if op["id"] == anchor)
        operations[index:index] = added
        index = body.index(anchor)
        body[index:index] = [op["id"] for op in added]

    load_v = {"id": "load_v", "kind": "load", "role": "compute",
              "reads": ["v_public"], "writes": ["v_token_major"],
              "parameters": {"movement": "global"}}
    transpose_v = {"id": "transpose_v", "kind": "transpose", "role": "compute",
                   "reads": ["v_token_major"], "writes": ["v_row_major"],
                   "depends_on": ["load_v"], "parameters": {}}
    cast_v = {"id": "cast_v", "kind": "cast", "role": "compute",
              "reads": ["v_row_major"], "writes": ["v_fp32"],
              "depends_on": ["transpose_v"], "parameters": {"to": "fp32"}}
    insert_before("subtract_base", [load_v, transpose_v, cast_v])
    subtract = next(op for op in operations if op["id"] == "subtract_base")
    subtract["reads"] = ["v_fp32", "base_pred"]
    subtract["depends_on"] = ["cast_v", "read_base"]
    load_beta = {"id": "load_beta", "kind": "load", "role": "compute",
                 "reads": ["beta_gate"], "writes": ["beta_reg"],
                 "parameters": {"movement": "global"}}
    apply_beta = {"id": "apply_beta", "kind": "elementwise", "role": "compute",
                  "reads": ["rhs_adjusted", "beta_reg"], "writes": ["rhs_beta"],
                  "depends_on": ["subtract_base", "load_beta"],
                  "parameters": {"op": "mul", "broadcast_axis": 1}}
    insert_before("load_p", [load_beta, apply_beta])
    solve = next(op for op in operations if op["id"] == "solve")
    solve["reads"] = ["p_stage", "rhs_beta"]
    solve["depends_on"] = ["load_p", "apply_beta"]

    transpose_output = {"id": "transpose_public_output", "kind": "transpose",
                        "role": "compute", "reads": ["output_bf16"],
                        "writes": ["public_output_tile"],
                        "depends_on": ["round_output"], "parameters": {}}
    insert_before("store_output", [transpose_output])
    store = next(op for op in operations if op["id"] == "store_output")
    store["reads"] = ["public_output_tile"]
    store["depends_on"] = ["transpose_public_output"]

    value["access_maps"][:] = [a for a in value["access_maps"]
                               if a["operation"] != "load_rhs"]
    chunk = {"source": "loop", "name": "chunk_index"}
    dim1 = {"source": "dimension", "dimension": 1}
    dim2 = {"source": "dimension", "dimension": 2}
    value["access_maps"].extend([
        {"operation": "load_v", "buffer": "v_public",
         "indices": [chunk, dim1, dim2], "boundary": "mask_tiled_axes"},
        {"operation": "load_beta", "buffer": "beta_gate",
         "indices": [chunk, dim1], "boundary": "mask_tiled_axes"},
    ])
    return value


class NativeKdaVBetaInput(unittest.TestCase):
    def test_public_v_and_beta_form_row_owned_fp32_rhs(self):
        schedule = Schedule.from_dict(document())
        self.assertEqual([f for f in verify(schedule, target()) if f.blocks_lowering], [])
        self.assertEqual(native_cuda.preflight(schedule, target()), ())
        source = native_cuda.emit(schedule, target()).source
        load = source[source.index("// CAKE_OP: load_v"):]
        self.assertIn("((0 + col)) * 128", load)
        self.assertIn("((0 + (int(threadIdx.x) - 0))) * 1", load)
        self.assertIn("// CAKE_OP: transpose_v", load)
        self.assertIn("// CAKE_OP: apply_beta", source)
        self.assertIn("__fmul_rn(", source)

    def test_nonexclusive_input_view_refuses_the_fused_load(self):
        value = document()
        value["buffers"].append({
            "name": "extra_v_copy", "space": "register", "dtype": "bf16",
            "shape": [32, 128], "mode": "scratch",
        })
        value["operations"].append({
            "id": "extra_v_reader", "kind": "elementwise", "role": "compute",
            "reads": ["v_token_major"], "writes": ["extra_v_copy"],
            "depends_on": ["load_v"], "parameters": {"op": "relu"},
        })
        value["tile_loops"][0]["body"].append("extra_v_reader")
        self.assertIn("NATIVE_TRANSPOSE_LOAD_DOMAIN",
                      {f.code for f in native_cuda.preflight(
                          Schedule.from_dict(value), target())})


if __name__ == "__main__":
    unittest.main()
