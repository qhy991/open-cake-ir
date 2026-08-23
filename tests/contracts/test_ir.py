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
import json
import unittest
from pathlib import Path

from open_cake_ir.compiler.ir import (
    AccessIndexKind,
    BoundaryPolicy,
    DType,
    EpilogueFormula,
    EpilogueParameters,
    LoadMovement,
    MemorySpace,
    OperationKind,
    ReduceSumParameters,
    ReductionScope,
    Schedule,
    ScheduleParseError,
)

ROOT = Path(__file__).resolve().parents[2]
CORPUS = sorted((ROOT / "corpus" / "schedules").glob("*.json"))
B32 = ROOT / "corpus" / "schedules" / "flash-kmeans-b32-smoke.json"
TINYGEMM = ROOT / "corpus" / "schedules" / "tinygemm2-stage4-split-k.json"
ASSIGNMENT_FULL = ROOT / "corpus" / "schedules" / "flash-kmeans-assignment-full.json"


def _document(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _mutated(path: Path, mutate) -> dict:
    document = _document(path)
    mutate(document)
    return document


class RetainedScheduleTest(unittest.TestCase):
    def test_every_corpus_schedule_parses(self) -> None:
        self.assertEqual(len(CORPUS), 6)
        for path in CORPUS:
            with self.subTest(schedule=path.name):
                schedule = Schedule.load(path)
                self.assertEqual(schedule.schema_version, 1)
                self.assertEqual(schedule.target, "sm_100a")
                self.assertTrue(schedule.operations)
                self.assertTrue(schedule.outputs)

    def test_gpu_quickstart_schedule_parses(self) -> None:
        schedule = Schedule.load(ROOT / "examples" / "gpu" / "flash-kmeans-b32-smoke.json")
        self.assertEqual(schedule.profile, "flash_kmeans_b32_smoke")

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
    rewrote `reduce_sum` -> `reduce_argmin` (fabricating tie_break/nan_policy) so the
    frozen base parser would accept them. Both operations now parse as themselves.
    """

    def test_reduce_sum_keeps_its_kind_and_parameters(self) -> None:
        schedule = Schedule.load(TINYGEMM)
        operation = schedule.operation("reduce_partials")
        assert operation is not None
        self.assertIs(operation.kind, OperationKind.REDUCE_SUM)
        self.assertEqual(
            operation.parameters, ReduceSumParameters(parts=4, scope=ReductionScope.CTA)
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

    def test_operation_vocabulary_is_the_union_of_the_legacy_enums(self) -> None:
        self.assertEqual(
            {member.value for member in OperationKind},
            {
                "load",
                "mma",
                "epilogue",
                "reduce_argmin",
                "reduce_sum",
                "store",
                "fence_proxy",
            },
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
        """`len(warps)` admits sparse or out-of-range warp ids; the extent does not."""

        document = _mutated(B32, lambda d: d["roles"][0].update(warps=[0, 1, 2, 4096]))
        schedule = Schedule.from_dict(document)
        self.assertEqual(len(schedule.roles[0].warps), 4)
        self.assertEqual(schedule.total_warp_extent, 4097)

    def test_dtype_itemsize(self) -> None:
        self.assertEqual(DType.BF16.itemsize, 2)
        self.assertEqual(DType.FP32.itemsize, 4)
        self.assertEqual(DType.FP8_E4M3.itemsize, 1)


class StrictStructureTest(unittest.TestCase):
    def _reject(self, document: dict) -> str:
        with self.assertRaises(ScheduleParseError) as caught:
            Schedule.from_dict(document)
        return str(caught.exception)

    def test_unknown_fields_are_rejected_at_every_level(self) -> None:
        cases = {
            "schedule": lambda d: d.update(bogus=1),
            "schedule.buffers[0]": lambda d: d["buffers"][0].update(cache_policy="x"),
            "schedule.roles[0]": lambda d: d["roles"][0].update(priority=1),
            "schedule.operations[0]": lambda d: d["operations"][0].update(cost=1),
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

    The string-keyed compiler reports one opaque `PROFILE_SEMANTICS_MISMATCH` for every
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
                lambda d: d["operations"][3]["parameters"].update(
                    tie_break="highest_index"
                ),
                "schedule.operations[3].parameters.tie_break",
                "lowest_index",
            ),
            (
                B32,
                lambda d: d["buffers"][0].update(dtype="fp4"),
                "schedule.buffers[0].dtype",
                "bf16, fp16, fp32, fp8_e4m3, int32, int64",
            ),
            (
                B32,
                lambda d: d["buffers"][4].update(space="constant"),
                "schedule.buffers[4].space",
                "global, register, shared, tensor",
            ),
            (
                B32,
                lambda d: d["operations"][1].update(kind="tma_load"),
                "schedule.operations[1].kind",
                "epilogue, fence_proxy, load, mma, reduce_argmin, reduce_sum, store",
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
                B32, lambda d: d["operations"][2]["parameters"].update(accumulator="bf16")
            ),
        )
        self.assertIn(
            "parts must be a positive integer",
            self._message(
                TINYGEMM, lambda d: d["operations"][4]["parameters"].update(parts=0)
            ),
        )
        # a load carries movement, never coalesced
        self.assertIn(
            "unknown fields",
            self._message(
                B32, lambda d: d["operations"][0]["parameters"].update(coalesced=True)
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
            lambda d: d["operations"][5]["parameters"].update(
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
