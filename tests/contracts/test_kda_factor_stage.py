"""KDA midpoint factor component; upstream log-decays and normalized Q/K are required."""

from __future__ import annotations

from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler, Schedule, Target
from open_cake_ir.compiler.performance.work import work_bound
from open_cake_ir.compiler.verifier import verify

ROOT = Path(__file__).resolve().parents[2]


def factor_document() -> dict:
    """Compose log-prefix midpoint extraction and BF16 factor commitments."""
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

    def cast(name, src, out, dtype, shape):
        buffer(out, dtype, shape)
        operation(name, "cast", (src,), (out,), {"to": dtype})
        return out

    def elementwise(name, srcs, out, kind, shape, axis=None, scalar=None):
        buffer(out, "fp32", shape)
        params = {"op": kind}
        if axis is not None:
            params["broadcast_axis"] = axis
        if scalar is not None:
            params["scalar"] = scalar
        operation(name, "elementwise", srcs, (out,), params)
        return out

    for name, dtype in [("k_norm", "bf16"), ("q_norm", "bf16"), ("log_decay", "fp32")]:
        buffer(name, dtype, (TILES, HEADS, C, K), "global", "input")
    for name in ("key_forward", "key_backward", "query_forward"):
        buffer(name, "bf16", (TILES, HEADS, C, K), "global", "output")
    for name, dtype in [("k_tile", "bf16"), ("q_tile", "bf16"), ("log_tile", "fp32")]:
        buffer(name, dtype, (C, K))
    for op_name, src, dst in (
        ("load_k", "k_norm", "k_tile"),
        ("load_q", "q_norm", "q_tile"),
        ("load_log", "log_decay", "log_tile"),
    ):
        operation(op_name, "load", (src,), (dst,), {"movement": "global"})
    k = cast("cast_k", "k_tile", "k_fp", "fp32", (C, K))
    q = cast("cast_q", "q_tile", "q_fp", "fp32", (C, K))
    buffer("log_prefix", "fp32", (C, K))
    operation(
        "scan_log",
        "scan",
        ("log_tile",),
        ("log_prefix",),
        {"op": "sum", "axis": 0, "direction": "forward"},
    )
    buffer("token_index", "int32", (C,))
    operation(
        "token_range",
        "coordinate",
        (),
        ("token_index",),
        {"source": "range", "start": 0, "extent": C},
    )
    for name, point in (("first", 0), ("last", C - 1)):
        buffer(f"{name}_predicate", "int32", (C,))
        operation(
            f"compare_{name}",
            "compare",
            ("token_index",),
            (f"{name}_predicate",),
            {"op": "eq", "scalar": point},
        )
        cast(f"cast_{name}_mask", f"{name}_predicate", f"{name}_mask", "fp32", (C,))
        elementwise(
            f"mask_{name}_log",
            ("log_prefix", f"{name}_mask"),
            f"{name}_selected",
            "mul",
            (C, K),
            axis=0,
        )
        buffer(f"{name}_log", "fp32", (K,))
        operation(
            f"reduce_{name}_log",
            "reduce",
            (f"{name}_selected",),
            (f"{name}_log",),
            {"op": "sum", "axis": 0, "scope": "cta", "across_loop": False},
        )
    mid = elementwise("add_log_ends", ("first_log", "last_log"), "mid_sum", "add", (K,))
    mid = elementwise("midpoint", ("mid_sum",), "midpoint_log", "mul", (K,), scalar=0.5)
    forward = elementwise(
        "forward_log", ("log_prefix", mid), "forward_delta", "sub", (C, K), axis=1
    )
    backward = elementwise(
        "backward_log", (mid, "log_prefix"), "backward_delta", "sub", (C, K), axis=1
    )
    forward = elementwise("exp_forward", (forward,), "forward_factor", "exp", (C, K))
    backward = elementwise(
        "exp_backward", (backward,), "backward_factor", "exp", (C, K)
    )
    kf = elementwise(
        "multiply_key_forward", (k, forward), "key_forward_fp", "mul", (C, K)
    )
    kb = elementwise(
        "multiply_key_backward", (k, backward), "key_backward_fp", "mul", (C, K)
    )
    qf = elementwise(
        "multiply_query_forward", (q, forward), "query_forward_fp", "mul", (C, K)
    )
    for name, source in (
        ("key_forward", kf),
        ("key_backward", kb),
        ("query_forward", qf),
    ):
        local = cast("cast_" + name, source, name + "_tile", "bf16", (C, K))
        operation("store_" + name, "store", (local,), (name,), {"coalesced": True})
    chunk = {"source": "program", "name": "chunk"}
    head = {"source": "program", "name": "head"}
    i = {"source": "dimension", "dimension": 2}
    j = {"source": "dimension", "dimension": 3}
    for op_name, buf in (
        ("load_k", "k_norm"),
        ("load_q", "q_norm"),
        ("load_log", "log_decay"),
        ("store_key_forward", "key_forward"),
        ("store_key_backward", "key_backward"),
        ("store_query_forward", "query_forward"),
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
        "schedule_id": "kda-b300-h64-midpoint-factors-bf16",
        "target": "sm_103a",
        "lowering": {
            "backend": "triton",
            "entry_point": "cake_kda_b300_midpoint_factors_bf16",
        },
        "program_map": {
            "axes": [
                {
                    "name": "chunk",
                    "axis": 0,
                    "buffer": "k_norm",
                    "dimension": 0,
                    "tile": 1,
                },
                {
                    "name": "head",
                    "axis": 1,
                    "buffer": "k_norm",
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
        "outputs": ["key_forward", "key_backward", "query_forward"],
        "metadata": {},
    }
    return d


class KdaMidpointFactors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")

    def test_chunk_log_scan_and_midpoint_factors_lower_together(self):
        value = factor_document()
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
            value["outputs"], ["key_forward", "key_backward", "query_forward"]
        )
        self.assertIn(
            "tl.cumsum(log_tile.to(tl.float32), axis=0, reverse=False)", source
        )
        self.assertIn("first_predicate = (token_index == 0).to(tl.int32)", source)
        self.assertIn("last_predicate = (token_index == 31).to(tl.int32)", source)
        self.assertIn("forward_delta = log_prefix - midpoint_log[None, :]", source)
        self.assertIn("backward_delta = midpoint_log[None, :] - log_prefix", source)
        self.assertIn("key_forward_tile = key_forward_fp.to(tl.bfloat16)", source)
        self.assertIn(
            "scan_log", work_bound(Schedule.from_dict(value)).uncounted_arithmetic
        )

    def test_wrong_scan_output_shape_is_localized(self):
        value = factor_document()
        next(buffer for buffer in value["buffers"] if buffer["name"] == "log_prefix")[
            "shape"
        ] = [16, 128]
        findings = verify(
            Schedule.from_dict(value),
            Target.load(ROOT / "compiler/targets/sm_103a.json"),
        )
        self.assertIn("SCAN_SHAPE_MISMATCH", {finding.code for finding in findings})


if __name__ == "__main__":
    unittest.main()
