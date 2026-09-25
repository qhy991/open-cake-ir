"""Physically orient KDA preparation outputs for the V-row TMEM-A state path."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler, Schedule, Target
from open_cake_ir.compiler.backends import native_cuda
from open_cake_ir.compiler.performance.work import work_bound
from open_cake_ir.compiler.verifier import verify

ROOT = Path(__file__).resolve().parents[2]


def native_b_document() -> dict:
    path = ROOT / "tests/contracts/test_kda_state_preparation.py"
    spec = importlib.util.spec_from_file_location("kda_state_prep_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    d = module.state_preparation_document()
    d["schedule_id"] = "kda-b300-h64-fused-state-preparation-native-b"
    d["lowering"]["entry_point"] = "cake_kda_b300_fused_state_preparation_native_b"

    def buffer(name):
        return next(item for item in d["buffers"] if item["name"] == name)

    def operation(op_id):
        return next(item for item in d["operations"] if item["id"] == op_id)

    def rename_output(old, new):
        buffer(old)["name"] = new
        d["outputs"] = [new if name == old else name for name in d["outputs"]]
        for op in d["operations"]:
            op["reads"] = [new if name == old else name for name in op["reads"]]
            op["writes"] = [new if name == old else name for name in op["writes"]]
        for access in d["access_maps"]:
            if access["buffer"] == old:
                access["buffer"] = new

    rename_output("final_key_out", "final_key_mma_out")

    C, K = 32, 128
    buffer("final_key_mma_out")["shape"] = [256, 64, K, C]

    def transpose_before_store(store_id: str, name: str, shape: tuple[int, int]):
        store = operation(store_id)
        source = store["reads"][0]
        producer = next(item["id"] for item in d["operations"] if source in item["writes"])
        tile = name + "_tile"
        d["buffers"].append({
            "name": tile, "space": "register", "dtype": "bf16",
            "shape": list(shape), "mode": "scratch",
        })
        transpose = {
            "id": name, "kind": "transpose", "role": "compute",
            "reads": [source], "writes": [tile], "depends_on": [producer],
            "parameters": {},
        }
        d["operations"].insert(d["operations"].index(store), transpose)
        store["reads"] = [tile]
        store["depends_on"] = [name]

    # Native CUDA's tcgen05 B operand is physically [N,K] and the operation
    # computes A[M,K] @ B[N,K].T. The existing base-key/query, B and inverse
    # outputs already have this physical orientation. State correction alone
    # needs final_key[K,C] as B for U[V,C] @ final_key[C,K].
    transpose_before_store("prep_store_final_key", "prep_transpose_final_key", (K, C))
    return d


class KdaNativeBPreparation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")

    def test_all_seven_outputs_have_explicit_mma_orientation(self):
        value = native_b_document()
        assessment = self.compiler.assess(value)
        self.assertTrue(assessment.lowering_eligible, [
            (item.code, item.path) for item in assessment.findings if item.blocks_lowering
        ])
        source = self.compiler.lower(assessment).source
        self.assertEqual(source.count("tl.dot("), 21)
        self.assertIn("# CAKE_OP:prep_transpose_final_key", source)
        self.assertIn("tl.trans(prep_final_key_tile)", source)
        self.assertIn("inv_inverse_tile = inv_x5.to(tl.bfloat16)", source)
        shapes = {item["name"]: item["shape"] for item in value["buffers"]}
        self.assertEqual(shapes["base_key_out"], [256, 64, 32, 128])
        self.assertEqual(shapes["base_query_out"], [256, 64, 32, 128])
        self.assertEqual(shapes["final_key_mma_out"], [256, 64, 128, 32])
        self.assertEqual(value["outputs"], [
            "b_out", "inverse_out", "base_key_out",
            "base_query_out", "final_key_mma_out", "beta_gate_out", "prefix_end_out",
        ])
        bound = work_bound(Schedule.from_dict(value))
        self.assertEqual(bound.compulsory_read_bytes, 403749120)
        self.assertEqual(bound.compulsory_written_bytes, 480247808)

    def test_missing_transpose_is_refused_by_its_data_edge(self):
        value = native_b_document()
        value["operations"] = [
            item for item in value["operations"] if item["id"] != "prep_transpose_final_key"
        ]
        findings = verify(
            Schedule.from_dict(value), Target.load(ROOT / "compiler/targets/sm_103a.json")
        )
        self.assertIn("BUFFER_UNPRODUCED", {item.code for item in findings})

    def test_k128_base_projection_needs_a_new_native_staging_route(self):
        """Current K-major shared B rows stop at 128 bytes, short of BF16 K128."""
        witness = json.loads((ROOT / "tests/fixtures/tmem-state-mma-sm103a.json").read_text())
        for item in witness["buffers"]:
            if item["name"] in ("a", "a_reg", "a_tmem"):
                item["shape"] = [128, 128]
            elif item["name"] in ("b", "b_stage"):
                item["shape"] = [32, 128]
            elif item["name"] in ("c", "acc", "dot"):
                item["shape"] = [128, 32]
        witness["operations"][2]["parameters"]["descriptor_box"] = [32, 128]
        witness["operations"][3]["parameters"]["tile_shape"] = [128, 32, 128]
        witness["operations"][3]["parameters"]["instruction"]["shape"] = [128, 32, 16]
        errors = native_cuda.preflight(
            Schedule.from_dict(witness), Target.load(ROOT / "compiler/targets/sm_103a.json")
        )
        self.assertIn("NATIVE_SMEM_LAYOUT", {item.code for item in errors})


if __name__ == "__main__":
    unittest.main()
