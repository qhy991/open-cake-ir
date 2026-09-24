"""State-kernel input commitments for the fused KDA preparation variant."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler, Schedule, Target
from open_cake_ir.compiler.performance.work import work_bound
from open_cake_ir.compiler.verifier import verify

ROOT = Path(__file__).resolve().parents[2]


def fused_inverse_document() -> dict:
    path = ROOT / "tests/contracts/test_kda_fused_inverse.py"
    spec = importlib.util.spec_from_file_location("kda_fused_inverse_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.fused_inverse_document()


def state_preparation_document() -> dict:
    d = fused_inverse_document()
    d["schedule_id"] = "kda-b300-h64-fused-state-preparation"
    d["lowering"]["entry_point"] = "cake_kda_b300_fused_state_preparation"
    TILES, HEADS, C, K = 256, 64, 32, 128
    for name, dtype, shape in (
        ("base_key_out", "bf16", (TILES, HEADS, C, K)),
        ("base_query_out", "bf16", (TILES, HEADS, C, K)),
        ("final_key_out", "bf16", (TILES, HEADS, C, K)),
        ("beta_gate_out", "fp32", (TILES, HEADS, C)),
        ("prefix_end_out", "fp32", (TILES, HEADS, K)),
    ):
        d["buffers"].append(
            {
                "name": name,
                "space": "global",
                "dtype": dtype,
                "shape": list(shape),
                "mode": "output",
            }
        )
        d["outputs"].append(name)
    new = []
    insert = next(
        i for i, op in enumerate(d["operations"]) if op["id"] == "cpl_load_lower"
    )
    producer = {
        buffer: op["id"] for op in d["operations"][:insert] for buffer in op["writes"]
    }

    def local(name, dtype, shape):
        d["buffers"].append(
            {
                "name": name,
                "space": "register",
                "dtype": dtype,
                "shape": list(shape),
                "mode": "scratch",
            }
        )
        return name

    def op(name, kind, reads, writes, params):
        deps = list(dict.fromkeys(producer[x] for x in reads if x in producer))
        item = {
            "id": name,
            "kind": kind,
            "role": "compute",
            "reads": list(reads),
            "writes": list(writes),
            "parameters": params,
        }
        if deps:
            item["depends_on"] = deps
        new.append(item)
        for x in writes:
            producer[x] = name
        return name

    def unary(name, src, dst, kind, shape):
        local(dst, "fp32", shape)
        op(name, "elementwise", (src,), (dst,), {"op": kind})
        return dst

    def binary(name, a, b, dst, kind, shape, axis=None):
        local(dst, "fp32", shape)
        params = {"op": kind}
        if axis is not None:
            params["broadcast_axis"] = axis
        op(name, "elementwise", (a, b), (dst,), params)
        return dst

    def cast(name, src, dst):
        local(dst, "bf16", (C, K))
        op(name, "cast", (src,), (dst,), {"to": "bf16"})
        return dst

    def store(name, src, dst):
        return op(name, "store", (src,), (dst,), {"coalesced": True})

    # fac_log_prefix and fac_last_log exist before the coupling/ inverse body.
    unary(
        "prep_exp_prefix", "fac_log_prefix", "prep_prefix_factor", "exp", (C, K)
    )
    base_key = binary(
        "prep_base_key",
        "fac_k_fp",
        "prep_prefix_factor",
        "prep_base_key_fp",
        "mul",
        (C, K),
    )
    base_query = binary(
        "prep_base_query",
        "fac_q_fp",
        "prep_prefix_factor",
        "prep_base_query_fp",
        "mul",
        (C, K),
    )
    binary(
        "prep_final_delta",
        "fac_last_log",
        "fac_log_prefix",
        "prep_final_delta_tile",
        "sub",
        (C, K),
        axis=1,
    )
    unary(
        "prep_exp_final_delta",
        "prep_final_delta_tile",
        "prep_final_factor",
        "exp",
        (C, K),
    )
    final_key = binary(
        "prep_final_key",
        "fac_k_fp",
        "prep_final_factor",
        "prep_final_key_fp",
        "mul",
        (C, K),
    )
    for label, src in (
        ("base_key", base_key),
        ("base_query", base_query),
        ("final_key", final_key),
    ):
        tile = local("prep_" + label + "_tile", "bf16", (C, K))
        op("prep_round_" + label, "cast", (src,), (tile,), {"to": "bf16"})
        store("prep_store_" + label, tile, label + "_out")
    store("prep_store_beta", "beta_sigmoid_tile", "beta_gate_out")
    unary(
        "prep_exp_prefix_end", "fac_last_log", "prep_prefix_end", "exp", (K,)
    )
    store("prep_store_prefix_end", "prep_prefix_end", "prefix_end_out")
    d["operations"][insert:insert] = new
    chunk = {"source": "program", "name": "chunk"}
    head = {"source": "program", "name": "head"}
    token = {"source": "dimension", "dimension": 2}
    key = {"source": "dimension", "dimension": 3}
    for name in ("base_key", "base_query", "final_key"):
        d["access_maps"].append(
            {
                "operation": "prep_store_" + name,
                "buffer": name + "_out",
                "indices": [chunk, head, token, key],
                "boundary": "mask_tiled_axes",
            }
        )
    for name, buffer in (
        ("prep_store_beta", "beta_gate_out"),
        ("prep_store_prefix_end", "prefix_end_out"),
    ):
        d["access_maps"].append(
            {
                "operation": name,
                "buffer": buffer,
                "indices": [chunk, head, token],
                "boundary": "mask_tiled_axes",
            }
        )
    return d


class KdaStatePreparation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")

    def test_state_kernel_operands_are_explicit_outputs(self):
        value = state_preparation_document()
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
        self.assertEqual(source.count("tl.dot("), 21)
        self.assertEqual(
            value["outputs"],
            [
                "b_out",
                "inverse_out",
                "base_key_out",
                "base_query_out",
                "final_key_out",
                "beta_gate_out",
                "prefix_end_out",
            ],
        )
        self.assertIn("prep_base_key_fp = fac_k_fp * prep_prefix_factor", source)
        self.assertIn(
            "prep_final_delta_tile = fac_last_log[None, :] - fac_log_prefix", source
        )
        self.assertLess(
            source.index("# CAKE_OP:prep_store_final_key"),
            source.index("# CAKE_OP:cpl_prediction_coupling"),
        )
        bound = work_bound(Schedule.from_dict(value))
        self.assertEqual(bound.compulsory_read_bytes, 403749120)
        self.assertEqual(bound.compulsory_written_bytes, 480247808)

    def test_missing_final_factor_refuses_its_downstream_store(self):
        value = state_preparation_document()
        value["operations"] = [
            op for op in value["operations"] if op["id"] != "prep_exp_final_delta"
        ]
        findings = verify(
            Schedule.from_dict(value),
            Target.load(ROOT / "compiler/targets/sm_103a.json"),
        )
        self.assertIn("BUFFER_UNPRODUCED", {finding.code for finding in findings})


if __name__ == "__main__":
    unittest.main()
