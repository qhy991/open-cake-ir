"""KDA preparation for state-dependent forward substitution without inverse MMAs."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler, Schedule, Target
from open_cake_ir.compiler.performance.work import work_bound
from open_cake_ir.compiler.verifier import verify

ROOT = Path(__file__).resolve().parents[2]


def serial_preparation_document() -> dict:
    path = ROOT / "tests/contracts/test_kda_mn_ready_preparation.py"
    spec = importlib.util.spec_from_file_location("kda_mn_ready_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    d = module.mn_ready_document()
    d["schedule_id"] = "kda-b300-h64-serial-solve-preparation"
    d["lowering"]["entry_point"] = "cake_kda_b300_serial_solve_preparation"
    unused_pt = {
        "cpl_load_upper", "cpl_prediction_coupling_t", "cpl_mask_prediction_t",
        "cpl_beta_column", "cpl_negate_pt", "cpl_round_pt",
    }
    unused_pt_buffers = {
        "mask_strict_upper", "cpl_upper_tile", "cpl_raw_prediction_t",
        "cpl_masked_prediction_t", "cpl_beta_prediction_t",
        "cpl_negative_prediction_t", "cpl_pt_tile",
    }
    d["operations"] = [op for op in d["operations"]
                       if not op["id"].startswith("inv_")
                       and op["id"] not in unused_pt]
    d["buffers"] = [b for b in d["buffers"]
                    if not b["name"].startswith("inv_")
                    and b["name"] not in ("identity", "inverse_out")
                    and b["name"] not in unused_pt_buffers]
    d["access_maps"] = [a for a in d["access_maps"]
                        if not a["operation"].startswith("inv_")
                        and a["operation"] not in unused_pt
                        and a["buffer"] not in ("identity", "inverse_out")
                        and a["buffer"] not in unused_pt_buffers]
    d["outputs"].remove("inverse_out")
    d["buffers"].append({
        "name": "p_out", "space": "global", "dtype": "bf16",
        "shape": [256, 64, 32, 32], "mode": "output",
    })
    store = {
        "id": "prep_store_p", "kind": "store", "role": "compute",
        "reads": ["cpl_p_tile"], "writes": ["p_out"],
        "depends_on": ["cpl_round_p"], "parameters": {"coalesced": True},
    }
    insert = next(i for i, op in enumerate(d["operations"])
                  if op["id"] == "cpl_store_b")
    d["operations"].insert(insert, store)
    d["access_maps"].append({
        "operation": "prep_store_p", "buffer": "p_out",
        "indices": [
            {"source": "program", "name": "chunk"},
            {"source": "program", "name": "head"},
            {"source": "dimension", "dimension": 2},
            {"source": "dimension", "dimension": 3},
        ],
        "boundary": "mask_tiled_axes",
    })
    d["outputs"].insert(0, "p_out")
    return d


class KdaSerialPreparation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")
        cls.target = Target.load(ROOT / "compiler/targets/sm_103a.json")

    def test_preparation_exports_p_and_omits_inverse_chain(self):
        d = serial_preparation_document()
        assessment = self.compiler.assess(d)
        self.assertTrue(assessment.lowering_eligible, [
            (f.code, f.path) for f in assessment.findings if f.blocks_lowering
        ])
        source = self.compiler.lower(assessment).source
        self.assertEqual(source.count("tl.dot("), 2)
        self.assertNotIn("# CAKE_OP:inv_", source)
        self.assertIn("# CAKE_OP:prep_store_p", source)
        self.assertEqual(d["outputs"], [
            "p_out", "b_out", "base_key_mn_out", "base_query_mn_out",
            "final_key_mma_out", "beta_gate_out", "prefix_end_out",
        ])
        bound = work_bound(Schedule.from_dict(d))
        self.assertEqual(bound.compulsory_written_bytes, 480247808)

    def test_p_output_must_have_a_producer(self):
        d = serial_preparation_document()
        d["operations"] = [op for op in d["operations"]
                           if op["id"] != "cpl_round_p"]
        findings = verify(Schedule.from_dict(d), self.target)
        self.assertIn("BUFFER_UNPRODUCED", {f.code for f in findings})


if __name__ == "__main__":
    unittest.main()
