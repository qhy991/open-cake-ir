"""Contract tests for Triton emission.

The second backend. What matters here is the difference from the first: Triton owns
placement, so a Schedule targeting it commits to tiling and semantics and stops. The
digest pin that used to cap this profile at seventy-two admissible points is gone,
because the source now follows the Schedule instead of being selected by it.
"""

from __future__ import annotations

import ast
import copy
import itertools
import json
import unittest
from pathlib import Path

from open_cake_ir.compiler import Compiler
from open_cake_ir.compiler.emit_cutedsl import EmitError
from open_cake_ir.compiler.emit_triton import emit
from open_cake_ir.compiler.ir import Schedule
from open_cake_ir.compiler.target import Target

ROOT = Path(__file__).resolve().parents[2]
TARGET = Target.load(ROOT / "compiler" / "targets" / "sm_100a.json")
SCHEDULE = ROOT / "corpus" / "schedules" / "flash-kmeans-b32-smoke.json"
DRAFT = ROOT / "compiler" / "revision.json"


def _variant(**changes) -> dict:
    document = copy.deepcopy(json.loads(SCHEDULE.read_text(encoding="utf-8")))
    buffers = {b["name"]: b for b in document["buffers"]}
    block_n = changes.get("block_n", 256)
    block_k = changes.get("block_k", 64)
    for axis in document["program_map"]["axes"]:
        if axis["name"] == "token_block":
            axis["tile"] = block_n
    loop = document["tile_loops"][0]
    loop["tile"] = block_k
    loop["range_options"].update(
        {k: v for k, v in changes.items() if k in loop["range_options"]}
    )
    document["roles"][0]["warps"] = list(range(changes.get("warps", 4)))
    buffers["token_tile"]["shape"] = [block_n, 128]
    buffers["centroid_tile"]["shape"] = [block_k, 128]
    buffers["distance_tile"]["shape"] = [block_n, block_k]
    buffers["best_index_tile"]["shape"] = [block_n]
    for operation in document["operations"]:
        if operation["kind"] == "mma":
            operation["parameters"]["tile_shape"] = [block_n, block_k, 128]
    return document


class EmissionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.schedule = Schedule.load(SCHEDULE)
        self.emission = emit(self.schedule, TARGET)

    def test_the_emitted_source_is_valid_python(self) -> None:
        ast.parse(self.emission.source)

    def test_pointer_arithmetic_comes_from_the_access_maps(self) -> None:
        self.assertIn(
            "tokens + batch * N_TOKEN_BLOCK * D_TOKENS_2 "
            "+ token_block_offsets[:, None] * D_TOKENS_2 + tokens_d2_offsets[None, :]",
            self.emission.source,
        )
        self.assertIn(
            "mask=token_block_offsets[:, None] < N_TOKEN_BLOCK", self.emission.source
        )

    def test_the_loop_carries_its_declared_knobs(self) -> None:
        options = self.schedule.tile_loops[0].range_options
        self.assertIn(f"num_stages={options.num_stages}", self.emission.source)
        self.assertIn("disallow_acc_multi_buffer=True", self.emission.source)

    def test_the_reduction_carries_its_declared_tie_break(self) -> None:
        self.assertIn("tie_break_left=True", self.emission.source)
        self.assertIn("candidate_index < best_index_tile", self.emission.source)

    def test_every_operation_is_marked(self) -> None:
        for operation in self.schedule.operations:
            with self.subTest(operation=operation.op_id):
                self.assertIn(f"# CAKE_OP:{operation.op_id}", self.emission.source)

    def test_the_compile_contract_is_derived_not_restated(self) -> None:
        toolchain = self.emission.toolchain
        assert toolchain is not None
        self.assertEqual(
            toolchain["signature"],
            {
                "tokens": "*bf16",
                "centroids": "*bf16",
                "centroid_sq": "*fp32",
                "assignments": "*i32",
            },
        )
        self.assertEqual(toolchain["grid"], [2, 32, 1])
        self.assertEqual(toolchain["compile_options"], {"num_warps": 4, "num_stages": 2})


class PlacementTest(unittest.TestCase):
    """A tile-level dot must not claim placement the backend would ignore."""

    def test_a_triton_contract_rejects_tensor_core_placement(self) -> None:
        from open_cake_ir.compiler.verifier import verify

        document = _variant()
        for operation in document["operations"]:
            if operation["kind"] == "mma":
                operation["parameters"]["instruction"]["cta_group"] = 1
        codes = {f.code for f in verify(Schedule.from_dict(document), TARGET)}
        self.assertIn("MMA_PLACEMENT_UNSUPPORTED", codes)

    def test_a_tensor_core_contract_wants_it(self) -> None:
        from open_cake_ir.compiler.verifier import verify

        full = Schedule.load(
            ROOT / "corpus" / "schedules" / "flash-kmeans-assignment-full.json"
        )
        codes = {f.code for f in verify(full, TARGET)}
        self.assertNotIn("MMA_PLACEMENT_UNDECLARED", codes)


class OpenSpaceTest(unittest.TestCase):
    """The digest pin admitted exactly one shape of Schedule. Emission admits a space."""

    def setUp(self) -> None:
        self.compiler = Compiler.load(ROOT, DRAFT)

    def test_the_original_grid_still_lowers(self) -> None:
        admitted = 0
        for block_n, block_k, stages, warps in itertools.product(
            (64, 128, 256), (32, 64, 128), (1, 2, 3, 4), (4, 8)
        ):
            assessment = self.compiler.assess(
                _variant(block_n=block_n, block_k=block_k, num_stages=stages, warps=warps)
            )
            admitted += assessment.lowering_eligible
        self.assertEqual(admitted, 72)

    def test_points_the_pin_refused_now_lower(self) -> None:
        """`disable_licm` and `loop_unroll_factor` were pinned shut, not unsupported."""

        for label, changes in (
            ("unroll", {"loop_unroll_factor": 2}),
            ("licm", {"disable_licm": True}),
            ("stages", {"num_stages": 6}),
        ):
            with self.subTest(knob=label):
                assessment = self.compiler.assess(_variant(**changes))
                self.assertTrue(assessment.lowering_eligible)
                self.assertTrue(self.compiler.lower(assessment).source)

    def test_the_knob_reaches_the_source(self) -> None:
        assessment = self.compiler.assess(_variant(disable_licm=True))
        self.assertIn("disable_licm=True", self.compiler.lower(assessment).source)

    def test_a_schedule_no_multiprocessor_can_hold_is_refused(self) -> None:
        """An open space needs a real gate, and this is one it can derive."""

        assessment = self.compiler.assess(_variant(block_n=512))
        self.assertFalse(assessment.lowering_eligible)
        finding = next(
            f for f in assessment.findings if f.code == "RESIDENCY_IMPOSSIBLE"
        )
        self.assertIn("no CTA is resident", finding.message)


class UnderSpecificationTest(unittest.TestCase):
    def test_an_unnamed_instruction_is_refused(self) -> None:
        document = _variant()
        for operation in document["operations"]:
            if operation["kind"] == "mma":
                operation["parameters"].pop("instruction")
        with self.assertRaisesRegex(EmitError, "instruction contract"):
            emit(Schedule.from_dict(document), TARGET)

    def test_an_instruction_the_target_forbids_is_refused(self) -> None:
        document = _variant()
        for operation in document["operations"]:
            if operation["kind"] == "mma":
                operation["parameters"]["instruction"]["contract"] = "triton.dot.fp8"
        with self.assertRaisesRegex(EmitError, "not admitted"):
            emit(Schedule.from_dict(document), TARGET)


ROW_SUM_SCHEDULE = {
    "schema_version": 1,
    "schedule_id": "row-sum-contract-v1",
    "target": "sm_100a",
    "roles": [{"name": "compute", "warps": [0, 1, 2, 3]}],
    "allocations": [],
    "pipelines": [],
    "barriers": [],
    "buffers": [
        {"name": "x", "space": "global", "dtype": "bf16", "shape": [32, 512, 128], "mode": "input"},
        {"name": "y", "space": "global", "dtype": "fp32", "shape": [32, 512], "mode": "output"},
        {"name": "x_tile", "space": "register", "dtype": "fp32", "shape": [256, 64], "mode": "scratch"},
        {"name": "acc", "space": "register", "dtype": "fp32", "shape": [256], "mode": "scratch"},
    ],
    "operations": [
        {"id": "load_x", "kind": "load", "role": "compute", "reads": ["x"],
         "writes": ["x_tile"], "parameters": {"movement": "global"}},
        {"id": "row_sum", "kind": "reduce_sum", "role": "compute", "reads": ["x_tile"],
         "writes": ["acc"], "depends_on": ["load_x"], "parameters": {"axis": 1, "scope": "cta"}},
        {"id": "store_y", "kind": "store", "role": "compute", "reads": ["acc"],
         "writes": ["y"], "depends_on": ["row_sum"], "parameters": {"coalesced": True}},
    ],
    "outputs": ["y"],
    "program_map": {"axes": [
        {"name": "row_block", "axis": 0, "buffer": "x", "dimension": 1, "tile": 256},
        {"name": "batch", "axis": 1, "buffer": "x", "dimension": 0, "tile": 1},
    ]},
    "tile_loops": [{"name": "feature_loop", "iterator": "feat_start", "buffer": "x",
                    "dimension": 2, "tile": 64, "body": ["load_x", "row_sum"],
                    "range_options": {"num_stages": 2, "loop_unroll_factor": 1,
                                      "flatten": False, "warp_specialize": False,
                                      "disallow_acc_multi_buffer": True,
                                      "disable_licm": False}}],
    "access_maps": [
        {"operation": "load_x", "buffer": "x", "indices": [
            {"source": "program", "name": "batch"},
            {"source": "program_tile", "name": "row_block"},
            {"source": "loop_tile", "name": "feat_start"}], "boundary": "mask_tiled_axes"},
        {"operation": "store_y", "buffer": "y", "indices": [
            {"source": "program", "name": "batch"},
            {"source": "program_tile", "name": "row_block"}], "boundary": "mask_tiled_axes"},
    ],
    "metadata": {"profile": "row_sum_contract", "workload_contract_sha256": "0" * 64},
}


class OperatorShapeIndependenceTest(unittest.TestCase):
    """The emitter must follow the Schedule, not the operator it was written against.

    Both admitted profiles contract and then reduce, so the emitter could require an mma
    and an argmin and still emit both correctly. A sum over an axis, with no contraction
    at all, is the smallest Schedule that tells those two apart.
    """

    def setUp(self) -> None:
        self.source = emit(Schedule.from_dict(ROW_SUM_SCHEDULE), TARGET).source

    def test_a_schedule_that_never_contracts_still_lowers(self) -> None:
        ast.parse(self.source)
        self.assertIn("# CAKE_OP:row_sum", self.source)
        self.assertNotIn("tl.dot", self.source)

    def test_the_sum_collapses_the_declared_axis(self) -> None:
        self.assertIn("acc += tl.sum(x_tile.to(tl.float32), axis=1)", self.source)
        # The identity has to exist before the loop that accumulates into it.
        self.assertLess(
            self.source.index("acc = tl.zeros"), self.source.index("acc += tl.sum")
        )

    def test_a_two_axis_mask_is_parenthesized(self) -> None:
        # `&` binds tighter than `<`, so a bare conjunction of comparisons becomes a
        # chained comparison against a bitwise and, and the load silently reads out of
        # bounds. Neither admitted profile masks two axes, so nothing caught this.
        mask = next(line for line in self.source.splitlines() if "mask=" in line and "&" in line)
        self.assertIn("(row_block_offsets[:, None] < N_ROW_BLOCK) &", mask)
        parsed = ast.parse(mask.strip().removeprefix("mask=").rstrip(","), mode="eval")
        self.assertIsInstance(parsed.body, ast.BinOp)
        self.assertIsInstance(parsed.body.op, ast.BitAnd)
