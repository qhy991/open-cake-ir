"""Fused KDA preparation with MN-major base-key/query B operands."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler, Schedule, Target
from open_cake_ir.compiler.performance.work import work_bound
from open_cake_ir.compiler.verifier import verify

ROOT = Path(__file__).resolve().parents[2]


def mn_ready_document() -> dict:
    path = ROOT / "tests/contracts/test_kda_native_b_preparation.py"
    spec = importlib.util.spec_from_file_location("kda_native_b_prep_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    d = module.native_b_document()
    d["schedule_id"] = "kda-b300-h64-fused-state-preparation-mn-b"
    d["lowering"]["entry_point"] = "cake_kda_b300_fused_state_preparation_mn_b"

    def buffer(name):
        return next(item for item in d["buffers"] if item["name"] == name)

    def operation(op_id):
        return next(item for item in d["operations"] if item["id"] == op_id)

    def turn_base(name):
        output = name + "_out"
        new_output = name + "_mn_out"
        buffer(output)["name"] = new_output
        buffer(new_output)["shape"] = [256, 64, 128, 32]
        d["outputs"] = [new_output if item == output else item for item in d["outputs"]]
        for op in d["operations"]:
            op["reads"] = [new_output if item == output else item for item in op["reads"]]
            op["writes"] = [new_output if item == output else item for item in op["writes"]]
        for access in d["access_maps"]:
            if access["buffer"] == output:
                access["buffer"] = new_output
        store = operation("prep_store_" + name)
        source = store["reads"][0]
        producer = next(op["id"] for op in d["operations"] if source in op["writes"])
        tile = name + "_mn_tile"
        d["buffers"].append({
            "name": tile, "space": "register", "dtype": "bf16",
            "shape": [128, 32], "mode": "scratch",
        })
        turn = "prep_transpose_" + name + "_mn"
        d["operations"].insert(d["operations"].index(store), {
            "id": turn, "kind": "transpose", "role": "compute",
            "reads": [source], "writes": [tile], "depends_on": [producer],
            "parameters": {},
        })
        store["reads"] = [tile]
        store["depends_on"] = [turn]

    turn_base("base_key")
    turn_base("base_query")
    return d


class KdaMnReadyPreparation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")
        cls.target = Target.load(ROOT / "compiler/targets/sm_103a.json")

    def test_fused_preparation_exposes_two_mn_b_operands(self):
        d = mn_ready_document()
        assessment = self.compiler.assess(d)
        self.assertTrue(assessment.lowering_eligible, [
            (f.code, f.path) for f in assessment.findings if f.blocks_lowering
        ])
        source = self.compiler.lower(assessment).source
        self.assertEqual(source.count("tl.dot("), 21)
        self.assertIn("base_key_mn_tile = tl.trans(prep_base_key_tile)", source)
        self.assertIn("base_query_mn_tile = tl.trans(prep_base_query_tile)", source)
        self.assertIn("final_key_mma_out", d["outputs"])
        outputs = {b["name"]: b["shape"] for b in d["buffers"] if b["mode"] == "output"}
        self.assertEqual(outputs["base_key_mn_out"], [256, 64, 128, 32])
        self.assertEqual(outputs["base_query_mn_out"], [256, 64, 128, 32])
        self.assertEqual(outputs["final_key_mma_out"], [256, 64, 128, 32])
        bound = work_bound(Schedule.from_dict(d))
        self.assertEqual(bound.compulsory_written_bytes, 480247808)

    def test_missing_mn_transpose_is_refused_by_its_data_edge(self):
        d = mn_ready_document()
        d["operations"] = [op for op in d["operations"]
                           if op["id"] != "prep_transpose_base_key_mn"]
        findings = verify(Schedule.from_dict(d), self.target)
        self.assertIn("BUFFER_UNPRODUCED", {f.code for f in findings})


if __name__ == "__main__":
    unittest.main()
