"""Contract tests for Triton emission.

The second backend. What matters here is the difference from the first: Triton owns
placement, so a Schedule targeting it commits to tiling and semantics and stops. The
digest pin that used to cap this profile at seventy-two admissible points is gone,
because the source now follows the Schedule instead of being selected by it.
"""

from __future__ import annotations

import ast
import builtins
import copy
from hashlib import sha256
import itertools
import json
import re
import struct
import unittest
from pathlib import Path

from open_cake_ir.compiler import Compiler
from open_cake_ir.compiler.emit_cutedsl import EmitError
from open_cake_ir.compiler.emit_triton import emit
from open_cake_ir.compiler.ir import Schedule, ScheduleParseError
from open_cake_ir.compiler.target import Target

ROOT = Path(__file__).resolve().parents[2]
TARGET = Target.load(ROOT / "compiler" / "targets" / "sm_100a.json")
SCHEDULE = ROOT / "corpus" / "schedules" / "flash-kmeans-b32-smoke-v2.json"
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
    # the composed distance names its intermediates, and a seed retiles all of them
    for name in ("cross", "scaled_cross"):
        if name in buffers:
            buffers[name]["shape"] = [block_n, block_k]
    if "norm_tile" in buffers:
        buffers["norm_tile"]["shape"] = [block_k]
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

    def test_a_reused_program_coordinate_is_bounded_by_the_accessed_buffer(self) -> None:
        schedule = Schedule.load(
            ROOT / "corpus" / "schedules" / "gemm-bias-b1-smoke-shape-drift.json"
        )

        source = emit(schedule, TARGET).source

        self.assertIn("D_BIAS_0=128", source)
        self.assertIn("mask=n_block_offsets < D_BIAS_0", source)
        self.assertNotIn("mask=n_block_offsets < N_N_BLOCK,\n        other=0.0", source)
        self.assertIn("m_block_offsets[:, None] < D_C_0", source)
        self.assertIn("n_block_offsets[None, :] < D_C_1", source)

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

    def test_a_large_logical_tile_reaches_the_backend(self) -> None:
        """Logical Buffer pressure cannot prove physical residency impossible."""

        assessment = self.compiler.assess(_variant(block_n=512))
        self.assertTrue(assessment.lowering_eligible)
        self.assertNotIn(
            "RESIDENCY_IMPOSSIBLE", {finding.code for finding in assessment.findings}
        )
        self.assertTrue(self.compiler.lower(assessment).source)


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
        {"id": "row_sum", "kind": "reduce", "role": "compute", "reads": ["x_tile"],
         "writes": ["acc"], "depends_on": ["load_x"],
         "parameters": {"op": "sum", "axis": 1, "scope": "cta"}},
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
    "lowering": {"backend": "triton", "entry_point": "cake_row_sum_contract"},
    "metadata": {"workload_contract_sha256": "0" * 64},
}


class OperatorShapeIndependenceTest(unittest.TestCase):
    """The emitter must follow the Schedule, not the operator it was written against.

    Both retained program slices contract and then reduce, so the emitter could require an mma
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

    def test_an_operation_outside_the_loop_is_not_dropped(self) -> None:
        # The skeleton used to emit prologue loads, the loop, then the store, so any
        # other operation declared outside the loop produced no code and no complaint.
        schedule = copy.deepcopy(ROW_SUM_SCHEDULE)
        schedule["buffers"].append(
            {"name": "scaled", "space": "register", "dtype": "fp32",
             "shape": [256], "mode": "scratch"}
        )
        schedule["operations"].insert(2, {
            "id": "halve", "kind": "elementwise", "role": "compute",
            "reads": ["acc"], "writes": ["scaled"], "depends_on": ["row_sum"],
            "parameters": {"op": "mul", "scalar": 0.5},
        })
        schedule["operations"][-1]["reads"] = ["scaled"]
        schedule["operations"][-1]["depends_on"] = ["halve"]
        source = emit(Schedule.from_dict(schedule), TARGET).source
        self.assertIn("# CAKE_OP:halve", source)
        self.assertIn("scaled = acc * 0.5", source)

    def test_a_two_axis_mask_is_parenthesized(self) -> None:
        # `&` binds tighter than `<`, so a bare conjunction of comparisons becomes a
        # chained comparison against a bitwise and, and the load silently reads out of
        # bounds. Neither admitted profile masks two axes, so nothing caught this.
        mask = next(line for line in self.source.splitlines() if "mask=" in line and "&" in line)
        self.assertIn("(row_block_offsets[:, None] < N_ROW_BLOCK) &", mask)
        parsed = ast.parse(mask.strip().removeprefix("mask=").rstrip(","), mode="eval")
        self.assertIsInstance(parsed.body, ast.BinOp)
        self.assertIsInstance(parsed.body.op, ast.BitAnd)


class ComposedArithmeticTest(unittest.TestCase):
    """A Schedule composes arithmetic rather than naming a whole operator's formula.

    `MmaFormula` and `EpilogueFormula` each name one operator's math in a single token,
    so a backend hardcodes that math and a new operator needs a new member and a new
    emitted body per backend. RMSNorm needs none of that: square, a scalar chain, a
    reciprocal square root and two broadcasts, all from the same five primitives.
    """

    def _rmsnorm(self, *, load_in_loop: bool) -> dict:
        rows, features = 64, 128
        body = ["load_x", "square", "sum_sq"] if load_in_loop else ["square", "sum_sq"]
        feature_index = (
            {"source": "loop_tile", "name": "feat"}
            if load_in_loop
            else {"source": "dimension", "dimension": 2}
        )
        register = lambda name, shape: {  # noqa: E731
            "name": name, "space": "register", "dtype": "fp32",
            "shape": shape, "mode": "scratch",
        }
        return {
            "schema_version": 1, "schedule_id": "rmsnorm-contract-v1", "target": "sm_100a",
            "roles": [{"name": "compute", "warps": [0, 1, 2, 3]}],
            "allocations": [], "pipelines": [], "barriers": [],
            "buffers": [
                {"name": "x", "space": "global", "dtype": "fp32",
                 "shape": [8, 512, features], "mode": "input"},
                {"name": "gamma", "space": "global", "dtype": "fp32",
                 "shape": [features], "mode": "input"},
                {"name": "y", "space": "global", "dtype": "fp32",
                 "shape": [8, 512, features], "mode": "output"},
                register("x_tile", [rows, features]), register("sq", [rows, features]),
                register("sumsq", [rows]), register("meansq", [rows]),
                register("shifted", [rows]), register("inv_rms", [rows]),
                register("gamma_tile", [features]),
                register("normed", [rows, features]), register("y_tile", [rows, features]),
            ],
            "operations": [
                {"id": "load_x", "kind": "load", "role": "compute", "reads": ["x"],
                 "writes": ["x_tile"], "parameters": {"movement": "global"}},
                {"id": "square", "kind": "elementwise", "role": "compute",
                 "reads": ["x_tile"], "writes": ["sq"], "depends_on": ["load_x"],
                 "parameters": {"op": "square"}},
                {"id": "sum_sq", "kind": "reduce", "role": "compute", "reads": ["sq"],
                 "writes": ["sumsq"], "depends_on": ["square"],
                 "parameters": {"op": "sum", "axis": 1, "scope": "cta"}},
                {"id": "mean", "kind": "elementwise", "role": "compute",
                 "reads": ["sumsq"], "writes": ["meansq"], "depends_on": ["sum_sq"],
                 "parameters": {"op": "mul", "scalar": 1.0 / features}},
                {"id": "shift", "kind": "elementwise", "role": "compute",
                 "reads": ["meansq"], "writes": ["shifted"], "depends_on": ["mean"],
                 "parameters": {"op": "add", "scalar": 1e-6}},
                {"id": "rsqrt", "kind": "elementwise", "role": "compute",
                 "reads": ["shifted"], "writes": ["inv_rms"], "depends_on": ["shift"],
                 "parameters": {"op": "rsqrt"}},
                {"id": "load_gamma", "kind": "load", "role": "compute", "reads": ["gamma"],
                 "writes": ["gamma_tile"], "parameters": {"movement": "global"}},
                {"id": "scale", "kind": "elementwise", "role": "compute",
                 "reads": ["x_tile", "inv_rms"], "writes": ["normed"],
                 "depends_on": ["rsqrt"],
                 "parameters": {"op": "mul", "broadcast_axis": 0}},
                {"id": "weight", "kind": "elementwise", "role": "compute",
                 "reads": ["normed", "gamma_tile"], "writes": ["y_tile"],
                 "depends_on": ["scale", "load_gamma"],
                 "parameters": {"op": "mul", "broadcast_axis": 1}},
                {"id": "store_y", "kind": "store", "role": "compute", "reads": ["y_tile"],
                 "writes": ["y"], "depends_on": ["weight"],
                 "parameters": {"coalesced": True}},
            ],
            "outputs": ["y"],
            "program_map": {"axes": [
                {"name": "row_block", "axis": 0, "buffer": "x", "dimension": 1, "tile": rows},
                {"name": "batch", "axis": 1, "buffer": "x", "dimension": 0, "tile": 1},
            ]},
            "tile_loops": [{"name": "feature_loop", "iterator": "feat", "buffer": "x",
                            "dimension": 2, "tile": features, "body": body,
                            "range_options": {"num_stages": 1, "loop_unroll_factor": 1,
                                              "flatten": False, "warp_specialize": False,
                                              "disallow_acc_multi_buffer": True,
                                              "disable_licm": False}}],
            "access_maps": [
                {"operation": "load_x", "buffer": "x", "indices": [
                    {"source": "program", "name": "batch"},
                    {"source": "program_tile", "name": "row_block"},
                    feature_index], "boundary": "mask_tiled_axes"},
                {"operation": "load_gamma", "buffer": "gamma", "indices": [
                    {"source": "dimension", "dimension": 0}], "boundary": "mask_tiled_axes"},
                {"operation": "store_y", "buffer": "y", "indices": [
                    {"source": "program", "name": "batch"},
                    {"source": "program_tile", "name": "row_block"},
                    {"source": "dimension", "dimension": 2}], "boundary": "mask_tiled_axes"},
            ],
            "lowering": {"backend": "triton", "entry_point": "cake_rmsnorm"},
            "metadata": {"workload_contract_sha256": "0" * 64},
        }

    def test_rmsnorm_lowers_without_a_formula_of_its_own(self) -> None:
        source = emit(Schedule.from_dict(self._rmsnorm(load_in_loop=False)), TARGET).source
        ast.parse(source)
        self.assertIn("sq = x_tile * x_tile", source)
        self.assertIn("inv_rms = tl.rsqrt(shifted)", source)
        # Both broadcast directions, chosen by the declared axis rather than guessed
        # from the shapes: a per-row scale spans axis 0, a per-column weight spans axis 1.
        self.assertIn("normed = x_tile * inv_rms[:, None]", source)
        self.assertIn("y_tile = normed * gamma_tile[None, :]", source)

    def test_a_register_tile_may_not_outlive_the_loop_that_writes_it(self) -> None:
        # Emitting this produced `NameError: x_tile is not defined` at Triton compile
        # time, because a register value the loop writes does not survive the loop. It
        # is a property of the declared Schedule, so the verifier answers it first.
        from open_cake_ir.compiler.verifier import verify

        findings = verify(Schedule.from_dict(self._rmsnorm(load_in_loop=True)), TARGET)
        escapes = [f for f in findings if f.code == "BUFFER_ESCAPES_LOOP"]
        self.assertTrue(escapes)
        self.assertTrue(all(f.blocks_lowering for f in escapes))
        self.assertIn("x_tile", escapes[0].message)


class ElementwiseArityTest(unittest.TestCase):
    """The backend's templates and the IR's arity must agree about every operator.

    `ElementwiseOp.arity` is what the verifier gates on; whether a template mentions a
    second operand is what the backend actually emits. Nothing connects them, so a unary
    op added to the enum and given a two-operand template would pass every gate and raise
    while formatting -- and a binary op with a one-operand template would silently drop
    an operand the Schedule declared, which is worse.
    """

    def test_every_template_uses_exactly_the_operands_its_arity_declares(self) -> None:
        from open_cake_ir.compiler.emit_triton import _TritonEmitter
        from open_cake_ir.compiler.ir import ElementwiseOp

        templates = _TritonEmitter._ELEMENTWISE_TEXT
        # Every operator the IR admits has a body, or the gate admits what cannot lower.
        self.assertEqual(set(templates) | {ElementwiseOp.TANH}, set(ElementwiseOp))
        for op, template in templates.items():
            with self.subTest(op=op.value):
                for index, operand in enumerate(("a", "b", "c"), start=1):
                    self.assertEqual("{" + operand + "}" in template, op.arity >= index)

    def test_tanh_is_emitted_only_when_the_schedule_declares_it(self) -> None:
        swiglu = emit(
            Schedule.load(ROOT / "corpus" / "schedules" / "swiglu-b8-smoke.json"),
            TARGET,
        ).source
        self.assertIn("from triton.language.extra import libdevice", swiglu)
        self.assertIn("tanh_gate = libdevice.tanh(half_gate)", swiglu)
        self.assertIn("silu_gate = gate_tile * sigmoid_gate", swiglu)

        existing = emit(Schedule.load(SCHEDULE), TARGET).source
        self.assertNotIn("triton.language.extra", existing)


class TopKEmissionTest(unittest.TestCase):
    @staticmethod
    def _key(value: float, index: int) -> int:
        value = 0.0 if value == 0.0 else value
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
        ordered = bits ^ (0xFFFFFFFF if bits & 0x80000000 else 0x80000000)
        return (ordered << 32) | (0xFFFFFFFF - index)

    def test_selection_is_ordered_distinct_and_inspectable(self) -> None:
        source = emit(
            Schedule.load(ROOT / "corpus" / "schedules" / "top-k-b8-smoke.json"),
            TARGET,
        ).source

        ast.parse(source)
        self.assertEqual(source.count("tl.topk(select_experts_keys, 8)"), 1)
        self.assertIn("select_experts_score_bits ^ 0xffffffff", source)
        self.assertIn("tl.where(score_row == 0.0, 0.0, score_row)", source)
        self.assertIn("0xffffffff - select_experts_source_positions", source)
        self.assertIn("select_experts_ranked_scores", source)
        self.assertIn("select_experts_ranked_indices", source)
        self.assertNotIn("select_experts_candidates_", source)

    def test_composite_key_preserves_float_order_and_lowest_index_ties(self) -> None:
        values = [float("-inf"), -7.0, -0.0, 0.0, 2.0, 2.0, float("inf")]
        observed = sorted(
            range(len(values)), key=lambda index: self._key(values[index], index), reverse=True
        )
        expected = sorted(range(len(values)), key=lambda index: (-values[index], index))
        self.assertEqual(observed, expected)


class DimensionSubRangeTest(unittest.TestCase):
    """Two operations may address disjoint halves of one buffer.

    Without this, an operator whose consumer wants half a row -- RoPE is the case that
    exposed it -- can only be authored if something upstream already split the buffer
    into two. That split is a materialization the IR would then be unable to describe
    removing, so the only expressible spelling was the un-optimized one.
    """

    FUSED = ROOT / "corpus" / "schedules" / "rope-b8-fused.json"

    def test_each_half_walks_its_own_offsets(self) -> None:
        schedule = Schedule.from_dict(json.loads(self.FUSED.read_text(encoding="utf-8")))

        source = emit(schedule, TARGET).source

        self.assertIn("x_d2_o0_e64_offsets = tl.arange(0, 64)", source)
        self.assertIn("x_d2_o64_eend_offsets = tl.arange(64, D_X_2)", source)
        # Both halves stride by the full row, or they would not be halves of one buffer.
        self.assertIn(
            "load_x_hi_ptrs = x + batch * N_ROW_BLOCK * D_X_2 "
            "+ row_block_offsets[:, None] * D_X_2 + x_d2_o64_eend_offsets[None, :]",
            source,
        )

    def test_a_whole_axis_keeps_its_original_spelling(self) -> None:
        """Every pre-existing case pins these bytes; the default must not drift."""

        schedule = Schedule.load(SCHEDULE)

        self.assertIn("tokens_d2_offsets = tl.arange(0, D_TOKENS_2)", emit(schedule, TARGET).source)

    def _with_hi_component(self, component: dict) -> dict:
        document = json.loads(self.FUSED.read_text(encoding="utf-8"))
        for access in document["access_maps"]:
            if access["operation"] == "load_x_hi":
                access["indices"][2] = component
        return document

    def test_a_sub_range_past_the_axis_is_refused(self) -> None:
        from open_cake_ir.compiler.verifier import verify

        document = self._with_hi_component(
            {"source": "dimension", "dimension": 2, "offset": 96, "extent": 64}
        )
        codes = {f.code for f in verify(Schedule.from_dict(document), TARGET)}

        self.assertIn("ACCESS_DIMENSION_SUBRANGE", codes)

    def test_one_range_has_one_spelling(self) -> None:
        """Two spellings of one range would lower to two sources, so two digests."""

        from open_cake_ir.compiler.verifier import verify

        # `[64, 128)` reaches the end of the axis, so `extent` must be omitted.
        redundant = self._with_hi_component(
            {"source": "dimension", "dimension": 2, "offset": 64, "extent": 64}
        )
        codes = {f.code for f in verify(Schedule.from_dict(redundant), TARGET)}
        self.assertIn("ACCESS_SUBRANGE_NONCANONICAL", codes)

        # A written zero offset is a second spelling of omitting it, refused at parse.
        with self.assertRaises(ScheduleParseError):
            Schedule.from_dict(
                self._with_hi_component(
                    {"source": "dimension", "dimension": 2, "offset": 0, "extent": 64}
                )
            )

    def test_a_sub_range_composes_with_a_runtime_indexed_gather(self) -> None:
        """Two primitives that each work must work together, or they are not orthogonal.

        The indexed value-domain derivation read the axis size while the emitter read
        the sub-range, so a gather the backend lowered correctly was refused for staging
        a buffer sized to what the access actually covers.
        """

        document = json.loads(
            (ROOT / "corpus" / "schedules" / "indexed-gather-b8-smoke.json")
            .read_text(encoding="utf-8")
        )
        for access in document["access_maps"]:
            if access["operation"] == "load_selected_rows":
                access["indices"][2] = {"source": "dimension", "dimension": 2, "extent": 8}
        for buffer in document["buffers"]:
            if buffer["name"] == "gathered_tile":
                buffer["shape"] = [8, 8]
            elif buffer["name"] == "gathered_rows":
                buffer["shape"] = [8, 8, 8]

        assessment = Compiler.load(ROOT, DRAFT).assess(document)

        self.assertNotIn(
            "ACCESS_INDEXED_VALUE_SHAPE", [f.code for f in assessment.findings]
        )
        self.assertTrue(assessment.accepted)
        self.assertTrue(assessment.lowering_eligible)
        self.assertIn(
            "expert_rows_d2_o0_e8_offsets = tl.arange(0, 8)",
            emit(Schedule.from_dict(document), TARGET).source,
        )

    def test_an_explicit_null_is_not_an_omitted_key(self) -> None:
        """`Compiler.assess` takes a document, so `null` reaches the parser directly.

        Reading the field with `.get` made `{"offset": null}` parse as the omitted
        spelling: identical kernel body, different Schedule bytes, so one program held
        two identities -- and the typed parser admitted what its own authoring Schema
        refuses. Presence of the key is what decides, not the value behind it.
        """

        for field, component in (
            ("offset", {"source": "dimension", "dimension": 2, "offset": None, "extent": 64}),
            ("extent", {"source": "dimension", "dimension": 2, "offset": 64, "extent": None}),
        ):
            with self.subTest(field=field):
                document = self._with_hi_component(component)
                with self.assertRaises(ScheduleParseError):
                    Schedule.from_dict(document)
                # and it must not slip through the public interface either
                assessment = Compiler.load(ROOT, DRAFT).assess(document)
                self.assertFalse(assessment.accepted)
                self.assertFalse(assessment.lowering_eligible)


class MultiOutputHostTest(unittest.TestCase):
    """A kernel with several outputs must hand back a host wrapper that can run.

    The store-count precondition used to read "exactly one store", but what it was
    protecting was the host wrapper, which bound a single tensor called `out`. Relaxing
    the count alone emitted a launch passing the second output's *buffer name* -- a name
    nothing defines -- so the emitter reported success and returned source that raises
    NameError when called. Syntax alone does not catch that, so this walks the wrapper
    and fails if any loaded name is unbound.
    """

    SCHEDULE = ROOT / "corpus" / "schedules" / "rope-b8-smoke.json"

    def _host(self, source: str) -> tuple[ast.FunctionDef, set[str]]:
        """The wrapper, and the module-level names it may legitimately reach for."""

        tree = ast.parse(source)
        module = {
            node.name for node in tree.body if isinstance(node, ast.FunctionDef)
        } | {"torch", "tl", "triton"}
        host = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
        )
        return host, module

    def test_every_output_reaches_the_kernel_through_a_bound_name(self) -> None:
        schedule = Schedule.from_dict(
            json.loads(self.SCHEDULE.read_text(encoding="utf-8"))
        )
        source = emit(schedule, TARGET).source
        host, module = self._host(source)

        bound = {argument.arg for argument in host.args.args} | module
        for node in ast.walk(host):
            targets = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, (ast.For, ast.comprehension)):
                targets = [node.target]
            for target in targets:
                bound |= {
                    name.id for name in ast.walk(target) if isinstance(name, ast.Name)
                }

        unbound = sorted(
            {
                node.id
                for node in ast.walk(host)
                if isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load)
                and node.id not in bound
                and not hasattr(builtins, node.id)
            }
        )
        self.assertEqual(unbound, [])

        launch = next(
            node
            for node in ast.walk(host)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript)
        )
        passed = [ast.unparse(argument) for argument in launch.args]
        self.assertEqual(passed[-2:], ["out[0]", "out[1]"])

    def test_a_stateful_kernel_binds_every_output_too(self) -> None:
        """Caller-owned state routes through a second host wrapper, which had the same
        defect. Both now share one output-binding owner, so this cannot diverge again."""

        document = json.loads(
            (ROOT / "corpus" / "schedules" / "reservation-owned-store-b8-smoke.json")
            .read_text(encoding="utf-8")
        )
        mirror = copy.deepcopy(
            next(b for b in document["buffers"] if b["name"] == "dispatched")
        )
        mirror["name"] = "dispatched_mirror"
        document["buffers"].append(mirror)
        store = copy.deepcopy(
            next(o for o in document["operations"] if o["id"] == "store_dispatched")
        )
        store["id"] = "store_dispatched_mirror"
        store["writes"] = ["dispatched_mirror"]
        document["operations"].append(store)
        access = copy.deepcopy(
            next(a for a in document["access_maps"] if a["operation"] == "store_dispatched")
        )
        access["operation"] = "store_dispatched_mirror"
        access["buffer"] = "dispatched_mirror"
        document["access_maps"].append(access)
        document["outputs"] = ["dispatched", "dispatched_mirror"]

        host, module = self._host(emit(Schedule.from_dict(document), TARGET).source)

        bound = {argument.arg for argument in host.args.args} | module
        for node in ast.walk(host):
            targets = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, (ast.For, ast.comprehension)):
                targets = [node.target]
            for target in targets:
                bound |= {
                    name.id for name in ast.walk(target) if isinstance(name, ast.Name)
                }
        unbound = sorted(
            {
                node.id
                for node in ast.walk(host)
                if isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load)
                and node.id not in bound
                and not hasattr(builtins, node.id)
            }
        )
        self.assertEqual(unbound, [])

        launch = next(
            node
            for node in ast.walk(host)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript)
        )
        passed = [ast.unparse(argument) for argument in launch.args]
        # the state buffer stays caller-owned by name; only the outputs take slots
        self.assertEqual(passed, ["expert_ids", "payloads", "counts", "out[0]", "out[1]"])

    def test_the_export_list_owns_the_slot_order(self) -> None:
        """`Schedule.outputs` is the order the caller reads results back in.

        Deriving it from buffer declaration order instead gave one fact two authorities,
        and they disagreed without saying so: a Schedule exporting `[y_hi, y_lo]` still
        handed back `y_lo` first, so every caller got its halves swapped.
        """

        document = json.loads(self.SCHEDULE.read_text(encoding="utf-8"))
        document["outputs"] = ["y_hi", "y_lo"]

        host, _ = self._host(emit(Schedule.from_dict(document), TARGET).source)
        launch = next(
            node
            for node in ast.walk(host)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript)
        )
        passed = [ast.unparse(argument) for argument in launch.args]

        # y_lo is declared first among the buffers but exported second, so it is out[1].
        self.assertEqual(passed[-2:], ["out[1]", "out[0]"])

    def test_one_buffer_cannot_take_two_slots(self) -> None:
        from open_cake_ir.compiler.verifier import verify

        document = json.loads(self.SCHEDULE.read_text(encoding="utf-8"))
        document["outputs"] = ["y_lo", "y_hi", "y_lo"]

        codes = {f.code for f in verify(Schedule.from_dict(document), TARGET)}

        self.assertIn("OUTPUT_DUPLICATE", codes)

    def test_one_output_still_binds_the_bare_name(self) -> None:
        """The single-output spelling is pinned by every other corpus case's digest."""

        schedule = Schedule.from_dict(json.loads(SCHEDULE.read_text(encoding="utf-8")))
        host, _ = self._host(emit(schedule, TARGET).source)
        launch = next(
            node
            for node in ast.walk(host)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript)
        )
        self.assertIn("out", [ast.unparse(argument) for argument in launch.args])


class EmittedObservationTest(unittest.TestCase):
    """The retained B200 observations for the operators this backend emits.

    Each record names the artifact that ran, so a change to a lowering detaches the
    evidence from the code and this says so. The answer is a new observation, taken with
    `tools/observe_lowered_kernel.py`, never an edit to a record.

    Softmax is the case that tested whether an operator is one profile row plus the
    Schedules that claim it. RMSNorm is here because admitting softmax changed it: a loop
    that runs once is the same program as no loop, and once the IR had both spellings one
    of them had to be the form. Correctness only; no timing was taken.
    """

    OBSERVED = (
        ("softmax", "SOFTMAX_OBSERVATION_20260824.json", "softmax-b8-smoke.json"),
        ("rmsnorm", "RMSNORM_OBSERVATION_20260824.json", "rmsnorm-b8-smoke.json"),
        (
            "layernorm",
            "LAYERNORM_OBSERVATION_20260824.json",
            "layernorm-b8-smoke.json",
        ),
        ("gemm-bias", "GEMM_OBSERVATION_20260824.json", "gemm-bias-b1-smoke.json"),
        (
            "rmsnorm-persistent",
            "PERSISTENT_OBSERVATION_20260824.json",
            "rmsnorm-b128-persistent.json",
        ),
        (
            "swiglu",
            "SWIGLU_IMPLEMENTATION_KIND_OBSERVATION_20260825.json",
            "swiglu-b8-smoke.json",
        ),
        ("top-k", "TOP_K_OBSERVATION_20260825.json", "top-k-b8-smoke.json"),
        (
            "block-scale",
            "BLOCK_SCALE_OBSERVATION_20260825.json",
            "block-scaled-gemm-b1-smoke.json",
        ),
        (
            "valid-extent",
            "VALID_EXTENT_OBSERVATION_20260825.json",
            "ragged-zero-pad-b1-smoke.json",
        ),
        (
            "ragged-grouped-gemm",
            "RAGGED_GROUPED_GEMM_OBSERVATION_20260825.json",
            "ragged-grouped-gemm-b1-smoke.json",
        ),
    )

    LOOPLESS = (
        "softmax-b8-smoke.json",
        "rmsnorm-b8-smoke.json",
        "layernorm-b8-smoke.json",
    )

    def test_each_observation_remains_historical_after_route_migration(self) -> None:
        from open_cake_ir.compiler import Compiler

        compiler = Compiler.load(ROOT, ROOT / "compiler" / "revision.lock.json")
        for name, record_name, schedule_name in self.OBSERVED:
            with self.subTest(operator=name):
                record = json.loads(
                    (ROOT / "inventory" / record_name).read_text(encoding="utf-8")
                )
                lowering = compiler.lower(
                    compiler.assess_file(ROOT / "corpus" / "schedules" / schedule_name)
                )
                self.assertEqual(lowering.generated, record["lowering"]["generated"])
                self.assertNotEqual(
                    lowering.schedule_sha256, record["schedule"]["canonical_sha256"]
                )
                self.assertNotEqual(
                    lowering.source_sha256, record["lowering"]["source_sha256"]
                )
                self.assertNotEqual(
                    lowering.compiler_revision_id,
                    record["compiler_revision"]["revision_id"],
                )
                self.assertEqual(record["result"]["mismatch_count"], 0)
                if "exact_match" in record["result"]:
                    self.assertTrue(record["result"]["exact_match"])
                else:
                    self.assertLessEqual(
                        record["result"]["max_deviation"],
                        record["result"]["tolerance"],
                    )
                self.assertTrue(record["result"]["passed"])
                self.assertFalse(record["performance_measured"])

    def test_a_row_wise_kernel_carries_no_loop(self) -> None:
        from open_cake_ir.compiler import Compiler

        compiler = Compiler.load(ROOT, ROOT / "compiler" / "revision.lock.json")
        for schedule_name in self.LOOPLESS:
            with self.subTest(schedule=schedule_name):
                source = compiler.lower(
                    compiler.assess_file(ROOT / "corpus" / "schedules" / schedule_name)
                ).source
                # Both hold their reduced axis whole, so a loop here would be a trip
                # count of one with an iterator nothing reads. The verifier refuses to
                # let a Schedule spell it that way.
                self.assertNotIn("tl.range(", source)
                self.assertIn("tl.sum(", source)

    def test_a_contraction_over_the_loop_accumulates(self) -> None:
        """The shape the accumulation derivation exists for, and the only one that has it.

        Flash-KMeans tiles the centroid axis, which is the output's N, so each iteration
        computes a fresh block and the dot assigns. This GEMM tiles K, so the iterations
        are a sum and the dot has to add. Which one a loop is doing is derived from the
        operands' access maps, not declared, so both shapes reach the same emitter.
        """

        from open_cake_ir.compiler import Compiler

        compiler = Compiler.load(ROOT, ROOT / "compiler" / "revision.lock.json")
        schedules = ROOT / "corpus" / "schedules"
        gemm = compiler.lower(
            compiler.assess_file(schedules / "gemm-bias-b1-smoke.json")
        ).source
        self.assertIn("acc = tl.zeros((64, 64), tl.float32)", gemm)
        self.assertIn("acc += tl.dot(", gemm)

        kmeans = compiler.lower(
            compiler.assess_file(schedules / "flash-kmeans-b32-smoke-v2.json")
        ).source
        # The same emitter, the other shape: no accumulator before the loop and no add.
        self.assertIn("cross = tl.dot(", kmeans)
        self.assertNotIn("cross +=", kmeans)

    def test_the_persistent_walk_actually_strides(self) -> None:
        """The one loop left, and the reason its Schedule was reshaped.

        A persistent grid is sized from the residency the Schedule commits to and capped
        at the tile count, so at the old shape it launched one CTA per tile and the
        stride never strode. The emitted grid-stride and its second iteration had never
        run. They do now: 592 CTAs over 1024 tiles, so most CTAs decode two.
        """

        from open_cake_ir.compiler import Compiler

        compiler = Compiler.load(ROOT, ROOT / "compiler" / "revision.lock.json")
        source = compiler.lower(
            compiler.assess_file(
                ROOT / "corpus" / "schedules" / "rmsnorm-b128-persistent.json"
            )
        ).source
        self.assertIn("for _work in tl.range(tl.program_id(0), TOTAL_TILES, NUM_CTAS)", source)
        constants = dict(
            re.findall(r"^\s+(TOTAL_TILES|NUM_CTAS)=(\d+),$", source, re.MULTILINE)
        )
        self.assertGreater(int(constants["TOTAL_TILES"]), int(constants["NUM_CTAS"]))


class FailedV24ObservationAttemptTest(unittest.TestCase):
    """The current-lowering probe failed, and that negative must stay legible."""

    def test_frozen_plan_and_every_written_record_remain_bound(self) -> None:
        attempt = json.loads(
            (ROOT / "inventory" / "V24_B200_CORRECTNESS_ATTEMPT_20260825.json").read_text()
        )
        plan_path = ROOT / attempt["plan"]["path"]
        self.assertEqual(
            sha256(plan_path.read_bytes()).hexdigest(), attempt["plan"]["raw_sha256"]
        )
        plan = json.loads(plan_path.read_text())
        self.assertEqual(plan["state"], "frozen")
        self.assertEqual(
            plan["partitions"]["generated_external_oracle"]["expected_case_count"],
            14,
        )

        passed = 0
        failed = 0
        schedules: set[str] = set()
        for authority in attempt["recorded_results"]:
            path = ROOT / authority["path"]
            self.assertEqual(
                sha256(path.read_bytes()).hexdigest(), authority["raw_sha256"]
            )
            record = json.loads(path.read_text())
            self.assertEqual(
                record["compiler_revision"]["revision_id"],
                "open-cake-ir-sm100a-v24",
            )
            self.assertTrue(record["lowering"]["generated"])
            self.assertFalse(record["scientific_claim_authorized"])
            self.assertFalse(record["performance_measured"])
            schedules.add(record["schedule"]["path"])
            if record["result"]["passed"]:
                passed += 1
            else:
                failed += 1

        self.assertEqual(len(schedules), 12)
        self.assertEqual((passed, failed), (11, 1))
        self.assertEqual(
            {item["case_id"] for item in attempt["unrecorded_failures"]},
            {"flash-kmeans-r16-accepted", "gemm-bias-shape-drift"},
        )
        self.assertFalse(attempt["disposition"]["generated_partition_passed"])
        self.assertEqual(attempt["disposition"]["checked_asset_partition"], "missing")
