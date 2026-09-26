"""Four MMA phases produce a bounded K128 state and C32 output tile."""
from __future__ import annotations

import copy
import math
import unittest

from open_cake_ir.compiler import Schedule
from open_cake_ir.compiler.backends import native_cuda
from open_cake_ir.compiler.verifier import verify
from tests.contracts.test_native_kda_decayed_state import document as state_document
from tests.contracts.test_native_two_phase_k128 import target


def document() -> dict:
    value = state_document()
    value["schedule_id"] = "native-k128-state-and-output"
    value["lowering"]["entry_point"] = "cake_k128_state_and_output"
    value["buffers"].extend([
        {"name": "query_b", "space": "global", "dtype": "bf16",
         "shape": [2, 128, 32], "mode": "input"},
        {"name": "output_b", "space": "global", "dtype": "bf16",
         "shape": [2, 32, 32], "mode": "input"},
        {"name": "chunk_output", "space": "global", "dtype": "bf16",
         "shape": [2, 128, 32], "mode": "output"},
        {"name": "query_stage", "space": "shared", "dtype": "bf16",
         "shape": [128, 32], "mode": "scratch", "allocation": "query_smem",
         "swizzle": "swizzle_64b"},
        {"name": "output_stage", "space": "shared", "dtype": "bf16",
         "shape": [32, 32], "mode": "scratch", "allocation": "output_smem",
         "swizzle": "swizzle_64b"},
        {"name": "query_acc", "space": "tensor", "dtype": "fp32",
         "shape": [128, 32], "mode": "scratch", "allocation": "tensor",
         "byte_offset": 122880},
        {"name": "output_acc", "space": "tensor", "dtype": "fp32",
         "shape": [128, 32], "mode": "scratch", "allocation": "tensor",
         "byte_offset": 139264},
        {"name": "query_reg", "space": "register", "dtype": "fp32",
         "shape": [128, 32], "mode": "scratch"},
        {"name": "output_reg", "space": "register", "dtype": "fp32",
         "shape": [128, 32], "mode": "scratch"},
        {"name": "output_sum", "space": "register", "dtype": "fp32",
         "shape": [128, 32], "mode": "scratch"},
        {"name": "output_scaled", "space": "register", "dtype": "fp32",
         "shape": [128, 32], "mode": "scratch"},
        {"name": "output_bf16", "space": "register", "dtype": "bf16",
         "shape": [128, 32], "mode": "scratch"},
    ])
    value["allocations"].extend([
        {"name": "query_smem", "space": "shared", "size_bytes": 8192},
        {"name": "output_smem", "space": "shared", "size_bytes": 2048},
    ])
    tensor = next(a for a in value["allocations"] if a["name"] == "tensor")
    tensor["size_bytes"] = 262144
    tensor["tensor_columns"] = 512
    for name in ("base_ready", "correction_ready"):
        next(b for b in value["barriers"] if b["name"] == name)["count"] = 2
    value["barriers"].extend([
        {"name": "query_done", "count": 1, "producers": ["mma"],
         "consumers": ["compute"], "mechanism": "mbarrier"},
        {"name": "output_done", "count": 1, "producers": ["mma"],
         "consumers": ["compute"], "mechanism": "mbarrier"},
    ])
    ops = {op["id"]: op for op in value["operations"]}

    def tma(source: str, stage: str, op_id: str, pipeline: str,
            barrier: str, shape: list[int]) -> dict:
        return {"id": op_id, "kind": "load", "role": "copy",
                "reads": [source], "writes": [stage], "signals": [barrier],
                "pipeline": pipeline,
                "parameters": {"movement": "tma", "descriptor_box": shape}}

    load_query = tma("query_b", "query_stage", "load_query", "base",
                     "base_ready", [128, 32])
    query_mma = copy.deepcopy(ops["mma_base"])
    query_mma.update(id="mma_query", reads=["state_tmem", "query_stage"],
                     writes=["query_acc"], signals=["query_done"],
                     depends_on=["store_initial", "load_query"])
    query_read = copy.deepcopy(ops["read_base"])
    query_read.update(id="read_query", reads=["query_acc"],
                      writes=["query_reg"], waits=["query_done"],
                      depends_on=["mma_query"])
    load_output = tma("output_b", "output_stage", "load_output", "correction",
                      "correction_ready", [32, 32])
    output_mma = copy.deepcopy(ops["mma_correction"])
    output_mma.update(id="mma_output", reads=["updates_tmem", "output_stage"],
                      writes=["output_acc"], signals=["output_done"],
                      depends_on=["publish_updates", "load_output"])
    output_mma["parameters"]["tile_shape"] = [128, 32, 32]
    output_mma["parameters"]["instruction"]["shape"] = [128, 32, 16]
    output_read = copy.deepcopy(ops["read_correction"])
    output_read.update(id="read_output", reads=["output_acc"],
                       writes=["output_reg"], waits=["output_done"],
                       depends_on=["mma_output"])
    combine = {"id": "combine_output", "kind": "elementwise", "role": "compute",
               "reads": ["query_reg", "output_reg"], "writes": ["output_sum"],
               "depends_on": ["read_query", "read_output"],
               "parameters": {"op": "add"}}
    scale = {"id": "scale_output", "kind": "elementwise", "role": "compute",
             "reads": ["output_sum"], "writes": ["output_scaled"],
             "depends_on": ["combine_output"],
             "parameters": {"op": "mul", "scalar": 1 / math.sqrt(128)}}
    rounded = {"id": "round_output", "kind": "cast", "role": "compute",
               "reads": ["output_scaled"], "writes": ["output_bf16"],
               "depends_on": ["scale_output"], "parameters": {"to": "bf16"}}
    stored = {"id": "store_output", "kind": "store", "role": "compute",
              "reads": ["output_bf16"], "writes": ["chunk_output"],
              "depends_on": ["round_output"],
              "parameters": {"coalesced": False}}
    operations = value["operations"]
    body = value["tile_loops"][0]["body"]

    def insert_before(anchor: str, new: list[dict]) -> None:
        index = next(i for i, op in enumerate(operations) if op["id"] == anchor)
        operations[index:index] = new
        index = body.index(anchor)
        body[index:index] = [op["id"] for op in new]

    def insert_after(anchor: str, new: list[dict]) -> None:
        index = next(i for i, op in enumerate(operations) if op["id"] == anchor) + 1
        operations[index:index] = new
        index = body.index(anchor) + 1
        body[index:index] = [op["id"] for op in new]

    insert_before("mma_base", [load_query])
    insert_after("mma_base", [query_mma])
    insert_before("mma_correction", [load_output])
    insert_after("mma_correction", [output_mma])
    # The query accumulator stays in its own TMEM columns while the row solve
    # runs. Reading it just before the output epilogue avoids keeping 32 FP32
    # values live per row across all 496 strict-lower FMAs.
    insert_after("read_correction", [output_read, query_read,
                                     combine, scale, rounded, stored])

    chunk = {"source": "loop", "name": "chunk_index"}
    dim1 = {"source": "dimension", "dimension": 1}
    dim2 = {"source": "dimension", "dimension": 2}
    for op_id, source in (("load_query", "query_b"),
                          ("load_output", "output_b"),
                          ("store_output", "chunk_output")):
        value["access_maps"].append({
            "operation": op_id, "buffer": source,
            "indices": [chunk, dim1, dim2],
            "boundary": "mask_tiled_axes",
        })
    value["outputs"].append("chunk_output")
    return value


class NativeKdaStateOutput(unittest.TestCase):
    def test_four_mma_witness_has_explicit_output_and_state(self):
        schedule = Schedule.from_dict(document())
        self.assertEqual([f for f in verify(schedule, target()) if f.blocks_lowering], [])
        self.assertEqual(native_cuda.preflight(schedule, target()), ())
        source = native_cuda.emit(schedule, target()).source
        for op in ("mma_base", "mma_query", "solve", "mma_correction",
                   "mma_output", "store_output", "publish_state"):
            self.assertIn("// CAKE_OP: " + op, source)
        self.assertLess(source.index("// CAKE_OP: mma_base"),
                        source.index("// CAKE_OP: mma_query"))
        self.assertLess(source.index("// CAKE_OP: mma_correction"),
                        source.index("// CAKE_OP: mma_output"))
        self.assertLess(source.index("// CAKE_OP: solve"),
                        source.index("// CAKE_OP: read_query"))
        self.assertEqual(source.count("cake_tma3((smem"), 4)

    def test_both_projection_and_correction_groups_keep_tensor_ownership(self):
        value = document()
        query = next(op for op in value["operations"] if op["id"] == "mma_query")
        query["reads"][0] = "updates_tmem"
        self.assertIn("NATIVE_TWO_PHASE_ORDER",
                      {f.code for f in native_cuda.preflight(
                          Schedule.from_dict(value), target())})
        value = document()
        output = next(op for op in value["operations"] if op["id"] == "mma_output")
        output["reads"][0] = "state_tmem"
        self.assertIn("NATIVE_TWO_PHASE_ORDER",
                      {f.code for f in native_cuda.preflight(
                          Schedule.from_dict(value), target())})


if __name__ == "__main__":
    unittest.main()
