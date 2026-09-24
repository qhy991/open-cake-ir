"""Bounded dual-view inverse stage; its input transpose relation is upstream-owned."""

from __future__ import annotations

from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler, Schedule, Target
from open_cake_ir.compiler.performance.work import work_bound
from open_cake_ir.compiler.verifier import verify

ROOT = Path(__file__).resolve().parents[2]


def inverse_document() -> dict:
    """Build the five-step dual-view schedule without duplicating 18 MMA nodes."""
    TILES, HEADS, C = 256, 64, 32
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
        dependencies = list(dict.fromkeys(producer[r] for r in reads if r in producer))
        row = {
            "id": name,
            "kind": kind,
            "role": "compute",
            "reads": list(reads),
            "writes": list(writes),
            "parameters": parameters,
        }
        if dependencies:
            row["depends_on"] = dependencies
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
                "tile_shape": [C, C, C],
                "instruction": {"contract": "triton.dot.bf16_fp32"},
            },
        )
        return out

    def cast(name, src, out, dtype):
        buffer(out, dtype, (C, C))
        operation(name, "cast", (src,), (out,), {"to": dtype})
        return out

    def add(name, a, b, out):
        buffer(out, "fp32", (C, C))
        operation(name, "elementwise", (a, b), (out,), {"op": "add"})
        return out

    buffer("p_global", "bf16", (TILES, HEADS, C, C), "global", "input")
    buffer("pt_global", "bf16", (TILES, HEADS, C, C), "global", "input")
    buffer("identity", "bf16", (C, C), "global", "input")
    buffer("inverse_out", "bf16", (TILES, HEADS, C, C), "global", "output")
    for name in ("p0", "pt0", "i0"):
        buffer(name, "bf16", (C, C))
    operation("load_p", "load", ("p_global",), ("p0",), {"movement": "global"})
    operation("load_pt", "load", ("pt_global",), ("pt0",), {"movement": "global"})
    operation("load_identity", "load", ("identity",), ("i0",), {"movement": "global"})
    cast("cast_identity", "i0", "x0", "fp32")
    p = "p0"
    pt = "pt0"
    x = "x0"
    xt = "x0"
    xtb = "i0"
    for step in range(5):
        c = mma(f"correction_{step}", p, xtb, f"c{step}")
        ct = mma(f"correction_t_{step}", xtb, p, f"ct{step}")
        x_next = add(f"update_inverse_{step}", x, c, f"x{step + 1}")
        xt_next = add(f"update_inverse_t_{step}", xt, ct, f"xt{step + 1}")
        if step < 4:
            pn = mma(f"square_power_{step}", p, pt, f"p_fp32_{step + 1}")
            ptn = mma(f"square_power_t_{step}", pt, p, f"pt_fp32_{step + 1}")
            p = cast(f"round_power_{step}", pn, f"p{step + 1}", "bf16")
            pt = cast(f"round_power_t_{step}", ptn, f"pt{step + 1}", "bf16")
            xtb = cast(f"round_inverse_t_{step}", xt_next, f"xtb{step + 1}", "bf16")
        x = x_next
        xt = xt_next
    cast("round_output_inverse", x, "inverse_tile", "bf16")
    operation(
        "store_inverse",
        "store",
        ("inverse_tile",),
        ("inverse_out",),
        {"coalesced": True},
    )
    chunk = {"source": "program", "name": "chunk"}
    head = {"source": "program", "name": "head"}
    i = {"source": "dimension", "dimension": 2}
    j = {"source": "dimension", "dimension": 3}
    for op_name, buf_name in [
        ("load_p", "p_global"),
        ("load_pt", "pt_global"),
        ("store_inverse", "inverse_out"),
    ]:
        access_maps.append(
            {
                "operation": op_name,
                "buffer": buf_name,
                "indices": [chunk, head, i, j],
                "boundary": "mask_tiled_axes",
            }
        )
    access_maps.append(
        {
            "operation": "load_identity",
            "buffer": "identity",
            "indices": [
                {"source": "dimension", "dimension": 0},
                {"source": "dimension", "dimension": 1},
            ],
            "boundary": "mask_tiled_axes",
        }
    )
    d = {
        "schema_version": 2,
        "schedule_id": "kda-b300-h64-inverse-dual-bf16",
        "target": "sm_103a",
        "lowering": {
            "backend": "triton",
            "entry_point": "cake_kda_b300_inverse_dual_bf16",
        },
        "program_map": {
            "axes": [
                {
                    "name": "chunk",
                    "axis": 0,
                    "buffer": "p_global",
                    "dimension": 0,
                    "tile": 1,
                },
                {
                    "name": "head",
                    "axis": 1,
                    "buffer": "p_global",
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
        "outputs": ["inverse_out"],
        "metadata": {},
    }
    return d


class KdaInverseStage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")

    def test_five_doubling_steps_have_explicit_mma_and_rounding_edges(self):
        value = inverse_document()
        assessment = self.compiler.assess(value)
        self.assertTrue(
            assessment.lowering_eligible,
            [
                (finding.code, finding.path)
                for finding in assessment.findings
                if finding.blocks_lowering
            ],
        )
        lowering = self.compiler.lower(assessment)
        source = lowering.source
        self.assertEqual(value["target"], "sm_103a")
        self.assertEqual(sum(op["kind"] == "mma" for op in value["operations"]), 18)
        self.assertEqual(source.count("tl.dot("), 18)
        self.assertIn("c0 = tl.dot(p0, tl.trans(i0))", source)
        self.assertIn("p_fp32_1 = tl.dot(p0, tl.trans(pt0))", source)
        self.assertIn("inverse_tile = x5.to(tl.bfloat16)", source)
        self.assertLess(
            source.index("# CAKE_OP:correction_0"),
            source.index("# CAKE_OP:correction_4"),
        )
        self.assertIn("torch.empty((256, 64, 32, 32), dtype=torch.bfloat16", source)
        self.assertEqual(work_bound(Schedule.from_dict(value)).mma_flops, 19327352832)

    def test_a_wrong_transposed_operand_tile_is_refused(self):
        value = inverse_document()
        next(buffer for buffer in value["buffers"] if buffer["name"] == "pt0")[
            "shape"
        ] = [16, 32]
        findings = verify(
            Schedule.from_dict(value),
            Target.load(ROOT / "compiler/targets/sm_103a.json"),
        )
        self.assertIn("MMA_INPUT_TILE_DOMAIN", {finding.code for finding in findings})


if __name__ == "__main__":
    unittest.main()
