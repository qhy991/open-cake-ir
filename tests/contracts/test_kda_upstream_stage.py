"""Fixed B300 KDA Q/K normalization and gate preprocessing component."""

from __future__ import annotations

from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler, Schedule, Target
from open_cake_ir.compiler.performance.work import work_bound
from open_cake_ir.compiler.verifier import verify

ROOT = Path(__file__).resolve().parents[2]


def upstream_document() -> dict:
    """Compose full-width Q/K normalization, log-decay and beta gate."""
    T, H, D, C = 8192, 64, 128, 32
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

    def cast(name, src, out, dtype, shape):
        buffer(out, dtype, shape)
        operation(name, "cast", (src,), (out,), {"to": dtype})
        return out

    def unary(name, src, out, kind, shape, scalar=None):
        buffer(out, "fp32", shape)
        params = {"op": kind}
        if scalar is not None:
            params["scalar"] = scalar
        operation(name, "elementwise", (src,), (out,), params)
        return out

    def binary(name, a, b, out, kind, shape, axis=None):
        buffer(out, "fp32", shape)
        params = {"op": kind}
        if axis is not None:
            params["broadcast_axis"] = axis
        operation(name, "elementwise", (a, b), (out,), params)
        return out

    def load(name, src, dst, dtype, shape):
        buffer(dst, dtype, shape)
        operation(name, "load", (src,), (dst,), {"movement": "global"})
        return dst

    for name in ("q", "k", "g"):
        buffer(name, "bf16", (T, H, D), "global", "input")
    buffer("beta", "bf16", (T, H), "global", "input")
    buffer("A_log", "fp32", (H,), "global", "input")
    buffer("dt_bias", "fp32", (H, D), "global", "input")
    for name in ("q_norm", "k_norm"):
        buffer(name, "bf16", (T // C, H, C, D), "global", "output")
    buffer("log_decay", "fp32", (T // C, H, C, D), "global", "output")
    buffer("beta_gate", "fp32", (T // C, H, C), "global", "output")
    for stem in ("q", "k"):
        tile = load("load_" + stem, stem, stem + "_tile", "bf16", (C, D))
        fp = cast("cast_" + stem, tile, stem + "_fp", "fp32", (C, D))
        square = unary("square_" + stem, fp, stem + "_square", "square", (C, D))
        buffer(stem + "_sumsq", "fp32", (C,))
        operation(
            "reduce_" + stem,
            "reduce",
            (square,),
            (stem + "_sumsq",),
            {"op": "sum", "axis": 1, "scope": "cta", "across_loop": False},
        )
        stabilized = unary(
            "stabilize_" + stem,
            stem + "_sumsq",
            stem + "_stabilized",
            "add",
            (C,),
            scalar=1e-24,
        )
        inverse = unary("rsqrt_" + stem, stabilized, stem + "_invnorm", "rsqrt", (C,))
        normalized = binary(
            "normalize_" + stem,
            fp,
            inverse,
            stem + "_normalized",
            "mul",
            (C, D),
            axis=0,
        )
        result = cast("round_" + stem, normalized, stem + "_norm_tile", "bf16", (C, D))
        operation(
            "store_" + stem + "_norm",
            "store",
            (result,),
            (stem + "_norm",),
            {"coalesced": True},
        )
    g_tile = load("load_g", "g", "g_tile", "bf16", (C, D))
    g_fp = cast("cast_g", g_tile, "g_fp", "fp32", (C, D))
    rate_log = load("load_rate_log", "A_log", "rate_log", "fp32", (1,))
    rate = unary("rate_exp", rate_log, "rate", "exp", (1,))
    bias = load("load_bias", "dt_bias", "bias_tile", "fp32", (D,))
    shifted = binary("gate_bias", g_fp, bias, "shifted", "add", (C, D), axis=1)
    scaled = binary("gate_rate", shifted, rate, "scaled", "mul", (C, D))
    negated = unary("negate_gate", scaled, "negative_gate", "mul", (C, D), scalar=-1.0)
    expneg = unary("exp_negative_gate", negated, "exp_negative", "exp", (C, D))
    denom = unary("gate_denom", expneg, "gate_denominator", "add", (C, D), scalar=1.0)
    sigmoid = unary("gate_sigmoid", denom, "gate_sigmoid_tile", "reciprocal", (C, D))
    logd = unary(
        "scale_log_decay", sigmoid, "log_decay_tile", "mul", (C, D), scalar=-5.0
    )
    operation("store_log_decay", "store", (logd,), ("log_decay",), {"coalesced": True})
    beta_tile = load("load_beta", "beta", "beta_tile", "bf16", (C,))
    beta_fp = cast("cast_beta", beta_tile, "beta_fp", "fp32", (C,))
    beta_neg = unary("negate_beta", beta_fp, "negative_beta", "mul", (C,), scalar=-1.0)
    beta_exp = unary("exp_negative_beta", beta_neg, "exp_beta", "exp", (C,))
    beta_denom = unary(
        "beta_denom", beta_exp, "beta_denominator", "add", (C,), scalar=1.0
    )
    beta_sig = unary(
        "beta_sigmoid", beta_denom, "beta_sigmoid_tile", "reciprocal", (C,)
    )
    operation(
        "store_beta_gate", "store", (beta_sig,), ("beta_gate",), {"coalesced": True}
    )
    chunk_tile = {"source": "program_tile", "name": "chunk"}
    chunk_scalar = {"source": "program", "name": "chunk"}
    head = {"source": "program", "name": "head"}
    key_input = {"source": "dimension", "dimension": 2}
    token_output = {"source": "dimension", "dimension": 2}
    key_output = {"source": "dimension", "dimension": 3}
    for name in ("q", "k", "g"):
        access_maps.append(
            {
                "operation": "load_" + name,
                "buffer": name,
                "indices": [chunk_tile, head, key_input],
                "boundary": "mask_tiled_axes",
            }
        )
    access_maps.append(
        {
            "operation": "load_beta",
            "buffer": "beta",
            "indices": [chunk_tile, head],
            "boundary": "mask_tiled_axes",
        }
    )
    access_maps.append(
        {
            "operation": "load_rate_log",
            "buffer": "A_log",
            "indices": [head],
            "boundary": "mask_tiled_axes",
        }
    )
    access_maps.append(
        {
            "operation": "load_bias",
            "buffer": "dt_bias",
            "indices": [head, {"source": "dimension", "dimension": 1}],
            "boundary": "mask_tiled_axes",
        }
    )
    for op_name, buf in (
        ("store_q_norm", "q_norm"),
        ("store_k_norm", "k_norm"),
        ("store_log_decay", "log_decay"),
    ):
        access_maps.append(
            {
                "operation": op_name,
                "buffer": buf,
                "indices": [chunk_scalar, head, token_output, key_output],
                "boundary": "mask_tiled_axes",
            }
        )
    access_maps.append(
        {
            "operation": "store_beta_gate",
            "buffer": "beta_gate",
            "indices": [chunk_scalar, head, token_output],
            "boundary": "mask_tiled_axes",
        }
    )
    d = {
        "schema_version": 2,
        "schedule_id": "kda-b300-h64-upstream-qk-gate",
        "target": "sm_103a",
        "lowering": {
            "backend": "triton",
            "entry_point": "cake_kda_b300_upstream_qk_gate",
        },
        "program_map": {
            "axes": [
                {"name": "chunk", "axis": 0, "buffer": "q", "dimension": 0, "tile": C},
                {"name": "head", "axis": 1, "buffer": "q", "dimension": 1, "tile": 1},
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
        "outputs": ["q_norm", "k_norm", "log_decay", "beta_gate"],
        "metadata": {},
    }
    return d


class KdaUpstreamStage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")

    def test_complete_component_has_explicit_four_outputs(self):
        value = upstream_document()
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
        self.assertEqual(
            value["outputs"], ["q_norm", "k_norm", "log_decay", "beta_gate"]
        )
        self.assertIn("q_sumsq = tl.sum(q_square.to(tl.float32), axis=1)", source)
        self.assertIn("k_sumsq = tl.sum(k_square.to(tl.float32), axis=1)", source)
        self.assertIn("q_invnorm = tl.rsqrt(q_stabilized)", source)
        self.assertIn("q_normalized = q_fp * q_invnorm[:, None]", source)
        self.assertIn("log_decay_tile = gate_sigmoid_tile * -5.0", source)
        self.assertIn("_cake_kda_b300_upstream_qk_gate_kernel[(256, 64, 1)]", source)
        bound = work_bound(Schedule.from_dict(value))
        self.assertGreater(bound.compulsory_written_bytes, 500_000_000)
        self.assertIn("rate_exp", bound.uncounted_arithmetic)

    def test_wrong_q_reduction_axis_is_rejected_by_shape_rule(self):
        value = upstream_document()
        next(op for op in value["operations"] if op["id"] == "reduce_q")["parameters"][
            "axis"
        ] = 0
        findings = verify(
            Schedule.from_dict(value),
            Target.load(ROOT / "compiler/targets/sm_103a.json"),
        )
        self.assertIn("REDUCE_SHAPE_MISMATCH", {finding.code for finding in findings})


if __name__ == "__main__":
    unittest.main()
