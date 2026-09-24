"""Fuse KDA preparation components without global intermediate reloads."""

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


def fused_document() -> dict:
    u = _document("test_kda_upstream_stage.py", "upstream_document")
    f = _document("test_kda_factor_stage.py", "factor_document")
    c = _document("test_kda_coupling_stage.py", "coupling_document")
    # The consumer's local tiles replace the producer's global stores and reloads.
    f_alias = {
        "k_tile": "k_norm_tile",
        "q_tile": "q_norm_tile",
        "log_tile": "log_decay_tile",
    }
    f_shared = {"key_forward_tile", "key_backward_tile", "query_forward_tile"}

    def f_name(name):
        return f_alias.get(name, name if name in f_shared else "fac_" + name)

    c_alias = {
        "key_forward_tile": "key_forward_tile",
        "key_backward_tile": "key_backward_tile",
        "query_forward_tile": "query_forward_tile",
        "beta_tile": "beta_sigmoid_tile",
    }
    c_global = {
        "mask_strict_lower",
        "mask_strict_upper",
        "mask_lower_equal",
        "p_out",
        "pt_out",
        "b_out",
    }

    def c_name(name):
        return c_alias.get(name, name if name in c_global else "cpl_" + name)

    remove_u = {"store_q_norm", "store_k_norm", "store_log_decay", "store_beta_gate"}
    remove_f = {
        "load_k",
        "load_q",
        "load_log",
        "store_key_forward",
        "store_key_backward",
        "store_query_forward",
    }
    remove_c = {
        "load_key_forward",
        "load_key_backward",
        "load_query_forward",
        "load_beta",
    }

    buffers = []
    for b in u["buffers"]:
        if b["space"] == "global" and b["mode"] == "output":
            continue
        buffers.append(copy.deepcopy(b))
    for b in f["buffers"]:
        if b["space"] == "global" or b["name"] in f_alias:
            continue
        item = copy.deepcopy(b)
        item["name"] = f_name(b["name"])
        buffers.append(item)
    for b in c["buffers"]:
        if b["space"] == "global":
            if b["mode"] in ("input", "output") and b["name"] not in (
                "key_forward",
                "key_backward",
                "query_forward",
                "beta_gate",
            ):
                buffers.append(copy.deepcopy(b))
            continue
        if b["name"] in c_alias:
            continue
        item = copy.deepcopy(b)
        item["name"] = c_name(b["name"])
        buffers.append(item)

    operations = []
    producer = {}

    def append(source, prefix, drop, map_buffer):
        for op in source["operations"]:
            if op["id"] in drop:
                continue
            item = copy.deepcopy(op)
            item["id"] = prefix + op["id"]
            item["reads"] = [map_buffer(name) for name in op["reads"]]
            item["writes"] = [map_buffer(name) for name in op["writes"]]
            deps = list(
                dict.fromkeys(
                    producer[name] for name in item["reads"] if name in producer
                )
            )
            if deps:
                item["depends_on"] = deps
            else:
                item.pop("depends_on", None)
            operations.append(item)
            for name in item["writes"]:
                producer[name] = item["id"]

    append(u, "", remove_u, lambda x: x)
    append(f, "fac_", remove_f, f_name)
    append(c, "cpl_", remove_c, c_name)

    access_maps = []
    for a in u["access_maps"]:
        if a["operation"] not in remove_u:
            access_maps.append(copy.deepcopy(a))
    for a in c["access_maps"]:
        if a["operation"] in remove_c:
            continue
        item = copy.deepcopy(a)
        item["operation"] = "cpl_" + item["operation"]
        access_maps.append(item)

    d = {
        "schema_version": 2,
        "schedule_id": "kda-b300-h64-fused-upstream-factor-coupling",
        "target": "sm_103a",
        "lowering": {
            "backend": "triton",
            "entry_point": "cake_kda_b300_fused_upstream_factor_coupling",
        },
        "program_map": copy.deepcopy(u["program_map"]),
        "roles": copy.deepcopy(u["roles"]),
        "allocations": [],
        "pipelines": [],
        "barriers": [],
        "tile_loops": [],
        "buffers": buffers,
        "operations": operations,
        "access_maps": access_maps,
        "outputs": copy.deepcopy(c["outputs"]),
        "metadata": {},
    }
    return d


class KdaFusedPreprocessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")

    def test_fusion_keeps_three_mmas_and_removes_global_intermediates(self):
        value = fused_document()
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
        self.assertEqual(value["target"], "sm_103a")
        self.assertEqual(value["outputs"], ["p_out", "pt_out", "b_out"])
        self.assertEqual(source.count("tl.dot("), 3)
        self.assertIn("tl.cumsum(log_decay_tile.to(tl.float32), axis=0", source)
        self.assertLess(
            source.index("# CAKE_OP:round_k"), source.index("# CAKE_OP:fac_cast_k")
        )
        self.assertLess(
            source.index("# CAKE_OP:fac_cast_key_forward"),
            source.index("# CAKE_OP:cpl_prediction_coupling"),
        )
        globals_ = {b["name"] for b in value["buffers"] if b["space"] == "global"}
        self.assertFalse(
            globals_
            & {
                "q_norm",
                "k_norm",
                "log_decay",
                "beta_gate",
                "key_forward",
                "key_backward",
                "query_forward",
            }
        )
        bound = work_bound(Schedule.from_dict(value))
        self.assertEqual(bound.compulsory_read_bytes, 403747072)
        self.assertEqual(bound.compulsory_written_bytes, 100663296)

    def test_missing_upstream_producer_blocks_fused_factor(self):
        value = fused_document()
        value["operations"] = [
            op for op in value["operations"] if op["id"] != "round_k"
        ]
        findings = verify(
            Schedule.from_dict(value),
            Target.load(ROOT / "compiler/targets/sm_103a.json"),
        )
        self.assertIn("BUFFER_UNPRODUCED", {finding.code for finding in findings})


if __name__ == "__main__":
    unittest.main()
