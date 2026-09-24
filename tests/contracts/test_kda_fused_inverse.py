"""Fuse B300 KDA coupling and BF16 inverse without global P/PT tensors."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler, Schedule, Target
from open_cake_ir.compiler.performance.work import work_bound
from open_cake_ir.compiler.verifier import verify

ROOT = Path(__file__).resolve().parents[2]


def _document(filename: str, factory: str) -> dict:
    path = ROOT / "tests/contracts" / filename
    spec = importlib.util.spec_from_file_location(factory, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, factory)()


def fused_inverse_document() -> dict:
    f = _document("test_kda_fused_preprocessing.py", "fused_document")
    i = _document("test_kda_inverse_stage.py", "inverse_document")
    d = copy.deepcopy(f)
    d["schedule_id"] = "kda-b300-h64-fused-prep-inverse"
    d["lowering"]["entry_point"] = "cake_kda_b300_fused_prep_inverse"
    d["buffers"] = [b for b in d["buffers"] if b["name"] not in ("p_out", "pt_out")]
    d["operations"] = [
        op for op in d["operations"] if op["id"] not in ("cpl_store_p", "cpl_store_pt")
    ]
    d["access_maps"] = [
        a
        for a in d["access_maps"]
        if a["operation"] not in ("cpl_store_p", "cpl_store_pt")
    ]
    d["outputs"] = ["b_out", "inverse_out"]
    for b in i["buffers"]:
        if b["space"] == "global":
            if b["name"] in ("identity", "inverse_out"):
                d["buffers"].append(copy.deepcopy(b))
        elif b["name"] not in ("p0", "pt0"):
            item = copy.deepcopy(b)
            item["name"] = "inv_" + item["name"]
            d["buffers"].append(item)

    def name(value):
        if value == "p0":
            return "cpl_p_tile"
        if value == "pt0":
            return "cpl_pt_tile"
        if value in ("identity", "inverse_out"):
            return value
        return "inv_" + value

    producer = {value: op["id"] for op in d["operations"] for value in op["writes"]}
    for op in i["operations"]:
        if op["id"] in ("load_p", "load_pt"):
            continue
        item = copy.deepcopy(op)
        item["id"] = "inv_" + item["id"]
        item["reads"] = [name(value) for value in item["reads"]]
        item["writes"] = [name(value) for value in item["writes"]]
        deps = list(
            dict.fromkeys(
                producer[value] for value in item["reads"] if value in producer
            )
        )
        if deps:
            item["depends_on"] = deps
        else:
            item.pop("depends_on", None)
        d["operations"].append(item)
        for value in item["writes"]:
            producer[value] = item["id"]
    for a in i["access_maps"]:
        if a["operation"] in ("load_p", "load_pt"):
            continue
        item = copy.deepcopy(a)
        item["operation"] = "inv_" + item["operation"]
        d["access_maps"].append(item)
    return d


class KdaFusedInverse(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")

    def test_coupling_flows_to_21_mmas_without_p_pt_global_spill(self):
        value = fused_inverse_document()
        assessment = self.compiler.assess(value)
        self.assertTrue(
            assessment.lowering_eligible,
            [
                (finding.code, finding.path)
                for finding in assessment.findings
                if finding.blocks_lowering
            ],
        )
        source = self.compiler.lower(assessment).source
        self.assertEqual(value["outputs"], ["b_out", "inverse_out"])
        self.assertEqual(source.count("tl.dot("), 21)
        self.assertLess(
            source.index("# CAKE_OP:cpl_round_p"),
            source.index("# CAKE_OP:inv_correction_0"),
        )
        self.assertIn("inv_c0 = tl.dot(cpl_p_tile, tl.trans(inv_i0))", source)
        globals_ = {b["name"] for b in value["buffers"] if b["space"] == "global"}
        self.assertFalse(
            globals_
            & {
                "p_out",
                "pt_out",
                "p_global",
                "pt_global",
                "q_norm",
                "k_norm",
                "log_decay",
                "key_forward",
                "key_backward",
                "query_forward",
            }
        )
        bound = work_bound(Schedule.from_dict(value))
        self.assertEqual(bound.compulsory_read_bytes, 403749120)
        self.assertEqual(bound.compulsory_written_bytes, 67108864)
        self.assertEqual(bound.mma_flops, 32212254720)

    def test_missing_coupling_cast_refuses_inverse_operand(self):
        value = fused_inverse_document()
        value["operations"] = [
            op for op in value["operations"] if op["id"] != "cpl_round_p"
        ]
        findings = verify(
            Schedule.from_dict(value),
            Target.load(ROOT / "compiler/targets/sm_103a.json"),
        )
        self.assertIn("BUFFER_UNPRODUCED", {finding.code for finding in findings})


if __name__ == "__main__":
    unittest.main()
