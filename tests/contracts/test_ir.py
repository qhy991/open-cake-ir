"""Contract tests for the canonical typed Schedule IR.

The IR is the single owner of Schedule structure. These tests fix three properties:

1. every retained Schedule parses, and the unified operation vocabulary carries each
   operation's real kind and typed parameters (no legacy kind rewriting);
2. parsing is total -- strict field checking means a parsed Schedule accounts for every
   field of its source document;
3. rejections are localized: the message names the offending path and, for closed
   vocabularies, the admitted values.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import sys
import unittest
from pathlib import Path
from typing import get_args, get_type_hints

from jsonschema import Draft202012Validator

from open_cake_ir.compiler import ir
from open_cake_ir.compiler.ir import (
    PLACED_CONTRACT_PREFIXES,
    AccessIndexKind,
    BoundaryPolicy,
    BufferMode,
    DType,
    ElementwiseInstruction,
    ElementwiseOp,
    ElementwiseParameters,
    EpilogueFormula,
    EpilogueParameters,
    IndexTieBreak,
    LoadMovement,
    MemorySpace,
    NaNPolicy,
    OperationKind,
    ReduceOp,
    ReduceParameters,
    ReductionScope,
    Schedule,
    ScheduleParseError,
    TopKParameters,
)
from open_cake_ir.compiler.schema import schedule_schema

ROOT = Path(__file__).resolve().parents[2]
# The manifest is what the Corpus is; the directory also holds schedules
# retained as history that the current Revision no longer admits.
CORPUS = sorted(
    ROOT / case["schedule"]
    for case in json.loads(
        (ROOT / "corpus" / "manifest.json").read_text(encoding="utf-8")
    )["cases"]
)
B32 = ROOT / "corpus" / "schedules" / "flash-kmeans-b32-smoke-v2.json"
TINYGEMM = ROOT / "corpus" / "schedules" / "tinygemm2-stage4-split-k.json"
ASSIGNMENT_FULL = ROOT / "corpus" / "schedules" / "flash-kmeans-assignment-full.json"
TOP_K = ROOT / "corpus" / "schedules" / "top-k-b8-smoke.json"
SWIGLU = ROOT / "corpus" / "schedules" / "swiglu-b8-smoke.json"


def _document(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _mutated(path: Path, mutate) -> dict:
    document = _document(path)
    mutate(document)
    return document


def _op(document: dict, op_id: str) -> dict:
    """Address an operation by name.

    Index-addressed mutations silently retarget when a Schedule gains an operation, so
    the composed distance turned several of these into assertions about a different one.
    """

    return next(o for o in document["operations"] if o["id"] == op_id)


class PublicIrBoundaryTest(unittest.TestCase):
    def test_public_types_resolve_to_the_same_ir_objects(self) -> None:
        """Moving a definition must not strand its public name or type annotations."""
        from open_cake_ir.compiler import Schedule as PublicSchedule

        self.assertIs(PublicSchedule, ir.Schedule)

        def check_annotation(annotation):
            if isinstance(annotation, type) and annotation.__module__.startswith(ir.__name__):
                self.assertIs(getattr(ir, annotation.__name__), annotation)
            for argument in get_args(annotation):
                check_annotation(argument)

        for name, value in vars(ir).items():
            if dataclasses.is_dataclass(value):
                with self.subTest(type=name):
                    hints = get_type_hints(value)
                    self.assertEqual(set(hints), {field.name for field in dataclasses.fields(value)})
                    for annotation in hints.values():
                        check_annotation(annotation)

    def test_compiler_source_set_covers_imported_ir_code(self) -> None:
        """A new IR module must remain inside the frozen Compiler source closure."""
        paths = set(json.loads((ROOT / "compiler/source_set.json").read_text())["paths"])
        for name, module in tuple(sys.modules.items()):
            if name == ir.__name__ or name.startswith(ir.__name__ + "."):
                with self.subTest(module=name):
                    self.assertIn(Path(module.__file__).resolve().relative_to(ROOT).as_posix(), paths)


class RetainedScheduleTest(unittest.TestCase):
    def test_every_corpus_schedule_parses(self) -> None:
        self.assertTrue(CORPUS)
        for path in CORPUS:
            with self.subTest(schedule=path.name):
                schedule = Schedule.load(path)
                self.assertEqual(schedule.schema_version, 1)
                self.assertEqual(schedule.target, _document(path)["target"])
                self.assertTrue(schedule.operations)
                self.assertTrue(
                    schedule.outputs
                    or any(buffer.mode is BufferMode.STATE for buffer in schedule.buffers)
                )

    def test_every_corpus_schedule_is_admitted_by_the_authoring_schema(self) -> None:
        """The prompt projection may not refuse bytes the canonical parser admits.

        The schema is the agent's authoring surface. A parser-only test missed fields and
        operation parameters that made sixteen of seventeen current Corpus documents
        invalid JSON-Schema instances even though the Compiler accepted them.
        """

        validator = Draft202012Validator(schedule_schema())
        for path in CORPUS:
            with self.subTest(schedule=path.name):
                errors = sorted(
                    validator.iter_errors(_document(path)),
                    key=lambda error: tuple(str(part) for part in error.absolute_path),
                )
                self.assertEqual(
                    errors,
                    [],
                    "\n".join(
                        f"{'.'.join(map(str, error.absolute_path))}: {error.message}"
                        for error in errors
                    ),
                )

    def test_allow_spill_is_not_an_unenforced_schedule_claim(self) -> None:
        document = _document(B32)
        document["residency"] = {
            "registers_per_thread": 128,
            "allow_spill": True,
        }
        with self.assertRaisesRegex(
            ScheduleParseError, "schedule.residency unknown fields.*allow_spill"
        ):
            Schedule.from_dict(document)

        errors = list(Draft202012Validator(schedule_schema()).iter_errors(document))
        self.assertTrue(any("allow_spill" in error.message for error in errors))

    def test_schema_refuses_operand_placement_the_contract_cannot_honour(self) -> None:
        """The authoring surface may not admit what the Verifier fatally refuses.

        Every first-Turn candidate of both completed open_cake Runs in the retained
        scientific campaign was rejected for exactly this: placement fields declared on
        a `triton.dot` contract that does not place its operands. The schema offered the
        fields with closed enums and said nothing about which contracts admit them, so
        the agent could only learn the rule by spending a Turn on a blocking Finding.
        """

        validator = Draft202012Validator(schedule_schema())
        document = _document(ROOT / "corpus" / "schedules" / "gemm-bias-b1-smoke.json")
        instruction = next(
            operation for operation in document["operations"]
            if operation["kind"] == "mma"
        )["parameters"]["instruction"]
        self.assertFalse(instruction["contract"].startswith(PLACED_CONTRACT_PREFIXES))
        self.assertEqual(list(validator.iter_errors(document)), [])

        for field, value in (
            ("cta_group", 1),
            ("operand_source", "shared"),
            ("operand_major", ["k", "k"]),
            ("shape", [64, 64, 16]),
        ):
            with self.subTest(field=field):
                drifted = copy.deepcopy(document)
                next(
                    operation for operation in drifted["operations"]
                    if operation["kind"] == "mma"
                )["parameters"]["instruction"][field] = value
                self.assertTrue(list(validator.iter_errors(drifted)))

        placed = copy.deepcopy(document)
        placed_instruction = next(
            operation for operation in placed["operations"]
            if operation["kind"] == "mma"
        )["parameters"]["instruction"]
        placed_instruction["contract"] = PLACED_CONTRACT_PREFIXES[0] + "mma.bf16"
        placed_instruction.update(
            {"shape": [64, 64, 16], "cta_group": 1,
             "operand_source": "shared", "operand_major": ["k", "k"]}
        )
        self.assertEqual(list(validator.iter_errors(placed)), [])

    def test_gpu_quickstart_schedule_parses(self) -> None:
        schedule = Schedule.load(ROOT / "examples" / "gpu" / "flash-kmeans-b32-smoke-v2.json")
        self.assertEqual(schedule.lowering.backend.value, "triton")
        self.assertEqual(schedule.lowering.entry_point, "cake_flash_kmeans_assign")

    def test_workload_profile_is_not_a_second_route_spelling(self) -> None:
        document = _document(B32)
        document["metadata"]["profile"] = "flash_kmeans_b32_smoke"

        with self.assertRaisesRegex(ScheduleParseError, "schedule.metadata unknown fields"):
            Schedule.from_dict(document)
        self.assertTrue(list(Draft202012Validator(schedule_schema()).iter_errors(document)))

    def test_grid_and_program_map_are_exclusive(self) -> None:
        b32 = Schedule.load(B32)
        self.assertIsNone(b32.grid)
        self.assertIsNotNone(b32.program_map)

        tinygemm = Schedule.load(TINYGEMM)
        self.assertEqual(tinygemm.grid, (64, 1, 1))
        self.assertIsNone(tinygemm.program_map)


class UnifiedVocabularyTest(unittest.TestCase):
    """The four legacy models are resolved into one.

    `explicit_resource_ir` rewrote `epilogue` -> `store` and `production_resource_ir`
    rewrote `reduce` -> `reduce_argmin` (fabricating tie_break/nan_policy) so the
    frozen base parser would accept them. Both operations now parse as themselves.
    """

    def test_a_reduction_keeps_its_kind_and_parameters(self) -> None:
        schedule = Schedule.load(TINYGEMM)
        operation = schedule.operation("reduce_partials")
        assert operation is not None
        self.assertIs(operation.kind, OperationKind.REDUCE)
        # The operator is a parameter, not the kind. A second fold was the moment that
        # choice had to be made, and making it the kind would have been two spellings of
        # collapsing an axis.
        self.assertEqual(
            operation.parameters,
            ReduceParameters(op=ReduceOp.SUM, axis=0, scope=ReductionScope.CTA),
        )

    def test_epilogue_keeps_its_kind_and_parameters(self) -> None:
        schedule = Schedule.load(TINYGEMM)
        operation = schedule.operation("bias_epilogue")
        assert operation is not None
        self.assertIs(operation.kind, OperationKind.EPILOGUE)
        self.assertEqual(
            operation.parameters,
            EpilogueParameters(
                formula=EpilogueFormula.BIAS_ADD_BF16_ROUND,
                coalesced=True,
                subtile=None,
                source_atom=None,
            ),
        )

    def test_both_legacy_epilogue_families_are_admitted(self) -> None:
        self.assertEqual(
            {member.value for member in EpilogueFormula},
            {"centroid_sq_minus_two_dot", "bias_add_bf16_round"},
        )

    def test_operation_vocabulary_covers_the_legacy_enums(self) -> None:
        # The migration must lose nothing, which is what this pins. It is a subset rather
        # than an equality because the vocabulary has since grown a kind the legacy models
        # had no equivalent for: arithmetic the formulas used to name whole-operator-wise.
        self.assertLessEqual(
            {
                "load",
                "mma",
                "epilogue",
                "reduce_argmin",
                "reduce",
                "store",
            },
            {member.value for member in OperationKind},
        )
        self.assertIn("elementwise", {member.value for member in OperationKind})
        self.assertIn("top_k", {member.value for member in OperationKind})

    def test_top_k_has_one_deterministic_spelling(self) -> None:
        operation = Schedule.load(TOP_K).operation("select_experts")
        assert operation is not None
        self.assertIs(operation.kind, OperationKind.TOP_K)
        self.assertEqual(
            operation.parameters,
            TopKParameters(
                k=8,
                tie_break=IndexTieBreak.LOWEST_INDEX,
                nan_policy=NaNPolicy.REJECT_INPUT,
            ),
        )

        # Descending and ordered are the operation contract, not independent modes.
        with self.assertRaisesRegex(ScheduleParseError, "unknown fields.*largest"):
            Schedule.from_dict(
                _mutated(
                    TOP_K,
                    lambda d: _op(d, "select_experts")["parameters"].update(
                        largest=True
                    ),
                )
            )

    def test_tanh_names_one_target_instruction_contract(self) -> None:
        operation = Schedule.load(SWIGLU).operation("tanh_gate")
        assert operation is not None
        self.assertEqual(
            operation.parameters,
            ElementwiseParameters(
                op=ElementwiseOp.TANH,
                scalar=None,
                broadcast_axis=None,
                instruction=ElementwiseInstruction("libdevice.tanh.f32"),
            ),
        )

        with self.assertRaisesRegex(ScheduleParseError, "instruction is required for tanh"):
            Schedule.from_dict(
                _mutated(
                    SWIGLU,
                    lambda d: _op(d, "tanh_gate")["parameters"].pop(
                        "instruction"
                    ),
                )
            )

        with self.assertRaisesRegex(ScheduleParseError, "no defined effect for mul"):
            Schedule.from_dict(
                _mutated(
                    SWIGLU,
                    lambda d: _op(d, "half_gate")["parameters"].update(
                        instruction={"contract": "libdevice.tanh.f32"}
                    ),
                )
            )

    def test_warp_specialized_roles_carry_their_pipeline(self) -> None:
        schedule = Schedule.load(ASSIGNMENT_FULL)
        self.assertEqual(
            [role.name for role in schedule.roles],
            ["epilogue", "mma", "tma", "reduce"],
        )
        self.assertEqual(schedule.total_warp_extent, 7)
        self.assertEqual([pipeline.name for pipeline in schedule.pipelines], ["main"])
        tiles_ready = next(b for b in schedule.barriers if b.name == "tiles_ready")
        self.assertEqual(tiles_ready.producers, ("tma",))
        self.assertEqual(tiles_ready.consumers, ("mma",))
        self.assertEqual(tiles_ready.pipeline, "main")
        loads = [o for o in schedule.operations if o.kind is OperationKind.LOAD]
        self.assertTrue(all(o.parameters.movement is LoadMovement.TMA for o in loads))

    def test_declared_resources_are_typed(self) -> None:
        schedule = Schedule.load(ASSIGNMENT_FULL)
        spaces = {a.name: a.space for a in schedule.allocations}
        self.assertEqual(spaces["smem_operands"], MemorySpace.SHARED)
        self.assertEqual(spaces["tmem_accumulator"], MemorySpace.TENSOR)

    def test_access_maps_are_typed(self) -> None:
        schedule = Schedule.load(B32)
        access = schedule.access_map("load_tokens", "tokens")
        assert access is not None
        self.assertIs(access.boundary, BoundaryPolicy.MASK_TILED_AXES)
        self.assertEqual(
            [index.source for index in access.indices],
            [
                AccessIndexKind.PROGRAM,
                AccessIndexKind.PROGRAM_TILE,
                AccessIndexKind.DIMENSION,
            ],
        )
        self.assertFalse(access.indices[0].is_vector)
        self.assertTrue(access.indices[1].is_vector)


class DerivedViewTest(unittest.TestCase):
    def test_buffer_byte_extent_accounts_for_stages(self) -> None:
        schedule = Schedule.load(ASSIGNMENT_FULL)
        staged = [b for b in schedule.buffers if b.stages > 1]
        self.assertTrue(staged)
        for buffer in staged:
            start, stop = buffer.byte_extent
            self.assertEqual(
                stop - start, buffer.elements * buffer.dtype.itemsize * buffer.stages
            )

    def test_warp_extent_uses_the_highest_index_not_the_count(self) -> None:
        """A contiguous role can still begin beyond the Target's warp range."""

        document = _mutated(
            B32, lambda d: d["roles"][0].update(warps=[4096, 4097, 4098, 4099])
        )
        schedule = Schedule.from_dict(document)
        self.assertEqual(len(schedule.roles[0].warps), 4)
        self.assertEqual(schedule.total_warp_extent, 4100)

    def test_dtype_itemsize(self) -> None:
        self.assertEqual(DType.BF16.itemsize, 2)
        self.assertEqual(DType.FP32.itemsize, 4)
        self.assertEqual(DType.FP8_E4M3.itemsize, 1)


class StrictStructureTest(unittest.TestCase):
    def _reject(self, document: dict) -> str:
        with self.assertRaises(ScheduleParseError) as caught:
            Schedule.from_dict(document)
        return str(caught.exception)

    def test_role_warps_have_one_canonical_interval_form(self) -> None:
        for warps in ([0, 2, 4, 6], [3, 2, 1, 0]):
            with self.subTest(warps=warps):
                message = self._reject(
                    _mutated(B32, lambda d: d["roles"][0].update(warps=warps))
                )
                self.assertIn("must be one ascending contiguous interval", message)

    def test_unknown_fields_are_rejected_at_every_level(self) -> None:
        cases = {
            "schedule": lambda d: d.update(bogus=1),
            "schedule.buffers[0]": lambda d: d["buffers"][0].update(cache_policy="x"),
            "schedule.roles[0]": lambda d: d["roles"][0].update(priority=1),
            "schedule.operations[0]": lambda d: _op(d, "load_tokens").update(cost=1),
        }
        for path, mutate in cases.items():
            with self.subTest(path=path):
                message = self._reject(_mutated(B32, mutate))
                self.assertIn(path, message)
                self.assertIn("unknown fields", message)

    def test_program_mapping_is_exclusive(self) -> None:
        both = self._reject(_mutated(B32, lambda d: d.update(grid=[1, 1, 1])))
        self.assertIn("exactly one of grid or program_map", both)
        neither = self._reject(_mutated(B32, lambda d: d.pop("program_map")))
        self.assertIn("exactly one of grid or program_map", neither)

    def test_role_warps_must_be_distinct_and_non_negative(self) -> None:
        self.assertIn(
            "repeats a warp index",
            self._reject(_mutated(B32, lambda d: d["roles"][0].update(warps=[0, 1, 1]))),
        )
        self.assertIn(
            "non-negative integer",
            self._reject(_mutated(B32, lambda d: d["roles"][0].update(warps=[0, -1]))),
        )

    def test_schema_version_is_pinned(self) -> None:
        self.assertIn(
            "schema_version",
            self._reject(_mutated(B32, lambda d: d.update(schema_version=2))),
        )


class LocalizedDiagnosticTest(unittest.TestCase):
    """A rejection must name the path and the admitted values.

    The former use-case-keyed compiler reported one opaque semantics mismatch for every
    one of these; the agent gets no repair target from that.
    """

    def _message(self, path: Path, mutate) -> str:
        with self.assertRaises(ScheduleParseError) as caught:
            Schedule.from_dict(_mutated(path, mutate))
        return str(caught.exception)

    def test_closed_vocabularies_report_path_and_admitted_values(self) -> None:
        cases = [
            (
                B32,
                lambda d: _op(d, "argmin")["parameters"].update(
                    tie_break="highest_index"
                ),
                "schedule.operations[6].parameters.tie_break",
                "lowest_index",
            ),
            (
                B32,
                lambda d: d["buffers"][0].update(dtype="fp4"),
                "schedule.buffers[0].dtype",
                "bf16, fp16, fp32, fp8_e4m3, int32",
            ),
            (
                B32,
                lambda d: d["buffers"][4].update(space="constant"),
                "schedule.buffers[4].space",
                "global, register, shared, tensor",
            ),
            (
                B32,
                lambda d: _op(d, "load_centroids").update(kind="tma_load"),
                "schedule.operations[1].kind",
                "atomic_rmw, cast, elementwise, epilogue, index_expand, load, mma, "
                "online_softmax, reduce, reduce_argmin, scan, store, top_k",
            ),
            (
                B32,
                lambda d: d["access_maps"][0].update(boundary="clamp"),
                "schedule.access_maps[0].boundary",
                "mask_tiled_axes",
            ),
            (
                B32,
                lambda d: d["access_maps"][0]["indices"][0].update(source="thread"),
                "schedule.access_maps[0].indices[0].source",
                "dimension, loop_tile, program, program_tile",
            ),
        ]
        for path, mutate, expected_path, admitted in cases:
            with self.subTest(path=expected_path):
                message = self._message(path, mutate)
                self.assertIn(expected_path, message)
                self.assertIn(admitted, message)

    def test_parameters_are_bound_to_their_operation_kind(self) -> None:
        self.assertIn(
            "must be fp32",
            self._message(
                B32, lambda d: _op(d, "distance_mma")["parameters"].update(accumulator="bf16")
            ),
        )
        self.assertIn(
            "axis must be a non-negative integer",
            self._message(
                TINYGEMM, lambda d: _op(d, "reduce_partials")["parameters"].update(axis=-1)
            ),
        )
        # a load carries movement, never coalesced
        self.assertIn(
            "unknown fields",
            self._message(
                B32, lambda d: _op(d, "load_tokens")["parameters"].update(coalesced=True)
            ),
        )


class UnificationCostTest(unittest.TestCase):
    """Merging the two disjoint legacy `EpilogueFormula` enums has one consequence.

    Legacy enforced "this formula belongs to this operator family" structurally, by
    keeping two enums in two modules. One canonical enum admits both members, so the
    family constraint is no longer structural. It is a semantic gate and belongs to the
    verifier; this test pins the boundary so the regression is deliberate, not silent.
    """

    def test_cross_family_epilogue_formula_is_structurally_admissible(self) -> None:
        document = _mutated(
            TINYGEMM,
            lambda d: _op(d, "bias_epilogue")["parameters"].update(
                formula="centroid_sq_minus_two_dot"
            ),
        )
        schedule = Schedule.from_dict(document)
        operation = schedule.operation("bias_epilogue")
        assert operation is not None
        self.assertEqual(
            operation.parameters.formula, EpilogueFormula.CENTROID_SQ_MINUS_TWO_DOT
        )


if __name__ == "__main__":
    unittest.main()
