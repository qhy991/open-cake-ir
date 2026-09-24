"""Bounded midpoint-scaled KDA coupling stage; factors are upstream-owned."""

from __future__ import annotations

from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler, Schedule, Target
from open_cake_ir.compiler.performance.work import work_bound
from open_cake_ir.compiler.verifier import verify

ROOT = Path(__file__).resolve().parents[2]


def coupling_document() -> dict:
    """Build three tensor-core couplings with explicit triangular masks."""
    TILES, HEADS, C, K = 256, 64, 32, 128
    buffers = []
    operations = []
    access_maps = []
    producer = {}

    def buffer(name, dtype, shape, space="register", mode="scratch"):
        buffers.append(
            {
                "name": name,
                "space": space,
                "dtype": dtype,
                "shape": list(shape),
                "mode": mode,
            }
        )
        return name

    def operation(name, kind, reads, writes, parameters):
        deps = list(dict.fromkeys(producer[r] for r in reads if r in producer))
        row = {
            "id": name,
            "kind": kind,
            "role": "compute",
            "reads": list(reads),
            "writes": list(writes),
            "parameters": parameters,
        }
        if deps:
            row["depends_on"] = deps
        operations.append(row)
        for w in writes:
            producer[w] = name
        return name

    def mma(name, a, b, out):
        buffer(out, "fp32", (C, C))
        operation(
            name,
            "mma",
            (a, b),
            (out,),
            {
                "accumulator": "fp32",
                "tile_shape": [C, C, K],
                "instruction": {"contract": "triton.dot.bf16_fp32"},
            },
        )
        return out

    def select(name, mask, raw, out):
        buffer(out, "fp32", (C, C))
        operation(name, "select", (mask, raw), (out,), {"false_value": 0.0})
        return out

    def mul(name, a, b, out, axis):
        buffer(out, "fp32", (C, C))
        operation(
            name, "elementwise", (a, b), (out,), {"op": "mul", "broadcast_axis": axis}
        )
        return out

    def cast(name, src, out):
        buffer(out, "bf16", (C, C))
        operation(name, "cast", (src,), (out,), {"to": "bf16"})
        return out

    for name in ("key_forward", "key_backward", "query_forward"):
        buffer(name, "bf16", (TILES, HEADS, C, K), "global", "input")
    buffer("beta_gate", "fp32", (TILES, HEADS, C), "global", "input")
    for name in ("mask_strict_lower", "mask_strict_upper", "mask_lower_equal"):
        buffer(name, "int32", (C, C), "global", "input")
    for name in ("p_out", "pt_out", "b_out"):
        buffer(name, "bf16", (TILES, HEADS, C, C), "global", "output")
    for name in ("key_forward_tile", "key_backward_tile", "query_forward_tile"):
        buffer(name, "bf16", (C, K))
    buffer("beta_tile", "fp32", (C,))
    for name in ("lower_tile", "upper_tile", "lower_equal_tile"):
        buffer(name, "int32", (C, C))
    for op_name, src, dst in (
        ("load_key_forward", "key_forward", "key_forward_tile"),
        ("load_key_backward", "key_backward", "key_backward_tile"),
        ("load_query_forward", "query_forward", "query_forward_tile"),
        ("load_beta", "beta_gate", "beta_tile"),
        ("load_lower", "mask_strict_lower", "lower_tile"),
        ("load_upper", "mask_strict_upper", "upper_tile"),
        ("load_lower_equal", "mask_lower_equal", "lower_equal_tile"),
    ):
        operation(op_name, "load", (src,), (dst,), {"movement": "global"})
    a = mma(
        "prediction_coupling", "key_forward_tile", "key_backward_tile", "raw_prediction"
    )
    at = mma(
        "prediction_coupling_t",
        "key_backward_tile",
        "key_forward_tile",
        "raw_prediction_t",
    )
    b = mma("output_coupling", "query_forward_tile", "key_backward_tile", "raw_output")
    a = select("mask_prediction", "lower_tile", a, "masked_prediction")
    at = select("mask_prediction_t", "upper_tile", at, "masked_prediction_t")
    b = select("mask_output", "lower_equal_tile", b, "masked_output")
    p = mul("beta_row", "masked_prediction", "beta_tile", "beta_prediction", 0)
    pt = mul("beta_column", "masked_prediction_t", "beta_tile", "beta_prediction_t", 1)
    # P = -diag(beta) A. The output coupling B has no beta factor.
    for name, src, dst in (
        ("negate_p", p, "negative_prediction"),
        ("negate_pt", pt, "negative_prediction_t"),
    ):
        buffer(dst, "fp32", (C, C))
        operation(name, "elementwise", (src,), (dst,), {"op": "mul", "scalar": -1.0})
    p = cast("round_p", "negative_prediction", "p_tile")
    pt = cast("round_pt", "negative_prediction_t", "pt_tile")
    b = cast("round_b", "masked_output", "b_tile")
    for name, src, dst in (
        ("store_p", p, "p_out"),
        ("store_pt", pt, "pt_out"),
        ("store_b", b, "b_out"),
    ):
        operation(name, "store", (src,), (dst,), {"coalesced": True})
    chunk = {"source": "program", "name": "chunk"}
    head = {"source": "program", "name": "head"}
    i = {"source": "dimension", "dimension": 2}
    j = {"source": "dimension", "dimension": 3}
    for name in ("key_forward", "key_backward", "query_forward"):
        access_maps.append(
            {
                "operation": "load_" + name,
                "buffer": name,
                "indices": [chunk, head, i, j],
                "boundary": "mask_tiled_axes",
            }
        )
    access_maps.append(
        {
            "operation": "load_beta",
            "buffer": "beta_gate",
            "indices": [chunk, head, i],
            "boundary": "mask_tiled_axes",
        }
    )
    for op_name, buf in (
        ("load_lower", "mask_strict_lower"),
        ("load_upper", "mask_strict_upper"),
        ("load_lower_equal", "mask_lower_equal"),
    ):
        access_maps.append(
            {
                "operation": op_name,
                "buffer": buf,
                "indices": [
                    {"source": "dimension", "dimension": 0},
                    {"source": "dimension", "dimension": 1},
                ],
                "boundary": "mask_tiled_axes",
            }
        )
    for op_name, buf in (
        ("store_p", "p_out"),
        ("store_pt", "pt_out"),
        ("store_b", "b_out"),
    ):
        access_maps.append(
            {
                "operation": op_name,
                "buffer": buf,
                "indices": [chunk, head, i, j],
                "boundary": "mask_tiled_axes",
            }
        )
    d = {
        "schema_version": 2,
        "schedule_id": "kda-b300-h64-midpoint-coupling-bf16",
        "target": "sm_103a",
        "lowering": {
            "backend": "triton",
            "entry_point": "cake_kda_b300_midpoint_coupling_bf16",
        },
        "program_map": {
            "axes": [
                {
                    "name": "chunk",
                    "axis": 0,
                    "buffer": "key_forward",
                    "dimension": 0,
                    "tile": 1,
                },
                {
                    "name": "head",
                    "axis": 1,
                    "buffer": "key_forward",
                    "dimension": 1,
                    "tile": 1,
                },
            ]
        },
        "roles": [{"name": "compute", "execution_groups": [0, 1, 2, 3]}],
        "allocations": [],
        "pipelines": [],
        "barriers": [],
        "tile_loops": [],
        "buffers": buffers,
        "operations": operations,
        "access_maps": access_maps,
        "outputs": ["p_out", "pt_out", "b_out"],
        "metadata": {},
    }
    return d


class KdaCouplingStage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")

    def test_three_mmas_and_beta_axes_are_explicit(self):
        value = coupling_document()
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
        self.assertIn(
            "raw_prediction = tl.dot(key_forward_tile, tl.trans(key_backward_tile))",
            source,
        )
        self.assertIn(
            "raw_prediction_t = tl.dot(key_backward_tile, tl.trans(key_forward_tile))",
            source,
        )
        self.assertIn("masked_prediction = tl.where(lower_tile != 0", source)
        self.assertIn("masked_prediction_t = tl.where(upper_tile != 0", source)
        self.assertIn(
            "beta_prediction = masked_prediction * beta_tile[:, None]", source
        )
        self.assertIn(
            "beta_prediction_t = masked_prediction_t * beta_tile[None, :]", source
        )
        self.assertEqual(work_bound(Schedule.from_dict(value)).mma_flops, 12884901888)

    def test_mma_tile_domain_refuses_wrong_factor_width(self):
        value = coupling_document()
        next(
            buffer
            for buffer in value["buffers"]
            if buffer["name"] == "key_backward_tile"
        )["shape"] = [32, 64]
        findings = verify(
            Schedule.from_dict(value),
            Target.load(ROOT / "compiler/targets/sm_103a.json"),
        )
        self.assertIn("MMA_INPUT_TILE_DOMAIN", {finding.code for finding in findings})


if __name__ == "__main__":
    unittest.main()
