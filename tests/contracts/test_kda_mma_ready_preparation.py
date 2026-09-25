"""Physically orient KDA preparation outputs for the V-row TMEM-A state path."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler, Schedule, Target
from open_cake_ir.compiler.performance.work import work_bound
from open_cake_ir.compiler.verifier import verify

ROOT = Path(__file__).resolve().parents[2]


def mma_ready_document() -> dict:
    path = ROOT / "tests/contracts/test_kda_state_preparation.py"
    spec = importlib.util.spec_from_file_location("kda_state_prep_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    d = module.state_preparation_document()
    d["schedule_id"] = "kda-b300-h64-fused-state-preparation-mma-ready"
    d["lowering"]["entry_point"] = "cake_kda_b300_fused_state_preparation_mma_ready"

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

    for old, new in (
        ("b_out", "b_mma_out"),
        ("inverse_out", "inverse_mma_out"),
        ("base_key_out", "base_key_mma_out"),
        ("base_query_out", "base_query_mma_out"),
    ):
        rename_output(old, new)

    C, K = 32, 128
    for name in ("base_key_mma_out", "base_query_mma_out"):
        buffer(name)["shape"] = [256, 64, K, C]

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

    transpose_before_store("cpl_store_b", "prep_transpose_b", (C, C))
    transpose_before_store("prep_store_base_key", "prep_transpose_base_key", (K, C))
    transpose_before_store("prep_store_base_query", "prep_transpose_base_query", (K, C))

    # The dual-view five-step inverse already has X^T; round that explicit value
    # rather than transposing the separately accumulated X after the fact.
    inverse_round = operation("inv_round_output_inverse")
    assert inverse_round["reads"] == ["inv_x5"]
    inverse_round["reads"] = ["inv_xt5"]
    inverse_round["depends_on"] = ["inv_update_inverse_t_4"]
    return d


class KdaMmaReadyPreparation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")

    def test_all_seven_outputs_have_explicit_mma_orientation(self):
        value = mma_ready_document()
        assessment = self.compiler.assess(value)
        self.assertTrue(assessment.lowering_eligible, [
            (item.code, item.path) for item in assessment.findings if item.blocks_lowering
        ])
        source = self.compiler.lower(assessment).source
        self.assertEqual(source.count("tl.dot("), 21)
        for name, operand in (
            ("prep_transpose_b", "cpl_b_tile"),
            ("prep_transpose_base_key", "prep_base_key_tile"),
            ("prep_transpose_base_query", "prep_base_query_tile"),
        ):
            self.assertIn(f"# CAKE_OP:{name}", source)
            self.assertIn(f"tl.trans({operand})", source)
        self.assertIn("inv_inverse_tile = inv_xt5.to(tl.bfloat16)", source)
        shapes = {item["name"]: item["shape"] for item in value["buffers"]}
        self.assertEqual(shapes["base_key_mma_out"], [256, 64, 128, 32])
        self.assertEqual(shapes["base_query_mma_out"], [256, 64, 128, 32])
        self.assertEqual(value["outputs"], [
            "b_mma_out", "inverse_mma_out", "base_key_mma_out",
            "base_query_mma_out", "final_key_out", "beta_gate_out", "prefix_end_out",
        ])
        bound = work_bound(Schedule.from_dict(value))
        self.assertEqual(bound.compulsory_read_bytes, 403749120)
        self.assertEqual(bound.compulsory_written_bytes, 480247808)

    def test_missing_transpose_is_refused_by_its_data_edge(self):
        value = mma_ready_document()
        value["operations"] = [
            item for item in value["operations"] if item["id"] != "prep_transpose_base_key"
        ]
        findings = verify(
            Schedule.from_dict(value), Target.load(ROOT / "compiler/targets/sm_103a.json")
        )
        self.assertIn("BUFFER_UNPRODUCED", {item.code for item in findings})


if __name__ == "__main__":
    unittest.main()
