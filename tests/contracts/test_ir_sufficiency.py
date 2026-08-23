"""Is the IR sufficient to derive the artifact it is supposed to describe?

`lower()` selects one of three checked-in files and stamps a digest into a comment.
That is not laziness: for the warp-specialized profile the Schedule genuinely
under-specifies its own kernel, so there is nothing to generate from. These tests fix
what the IR now determines and, just as importantly, what it still does not.

`examples/schedules/flash-kmeans-assignment-full-declared.json` is the retained
Schedule with every commitment the IR can currently express actually made. Comparing it
against the artifact is how the residual gap stays honest.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from open_cake_ir.compiler.ir import (
    LoadMovement,
    MemorySpace,
    OperandMajorMode,
    OperandSource,
    OperationKind,
    Schedule,
    Swizzle,
    TMEM_COLUMN_BYTES,
)
from open_cake_ir.compiler.target import Target
from open_cake_ir.compiler.verifier import verify

ROOT = Path(__file__).resolve().parents[2]
TARGET = Target.load(ROOT / "compiler" / "targets" / "sm_100a.json")
RETAINED = ROOT / "corpus" / "schedules" / "flash-kmeans-assignment-full.json"
DECLARED = ROOT / "examples" / "schedules" / "flash-kmeans-assignment-full-declared.json"
ARTIFACT = (
    ROOT
    / "src"
    / "open_cake_ir"
    / "compiler"
    / "assets"
    / "flash_kmeans_assignment_full.py.tmpl"
)


class DeclaredScheduleTest(unittest.TestCase):
    def test_the_declared_schedule_makes_every_expressible_commitment(self) -> None:
        findings = verify(Schedule.load(DECLARED), TARGET)
        self.assertEqual(findings, (), f"unexpected findings: {[str(f) for f in findings]}")

    def test_the_retained_schedule_declines_nine_commitments(self) -> None:
        findings = verify(Schedule.load(RETAINED), TARGET)
        self.assertEqual([f for f in findings if f.blocks_lowering], [])
        self.assertEqual(
            sorted(f.code for f in findings),
            [
                "ALLOCATION_TENSOR_COLUMNS_UNDECLARED",
                "BUFFER_SWIZZLE_UNDECLARED",
                "BUFFER_SWIZZLE_UNDECLARED",
                "EPILOGUE_SOURCE_ATOM_UNDECLARED",
                "EPILOGUE_SUBTILE_UNDECLARED",
                "MMA_INSTRUCTION_UNDECLARED",
                "MMA_TILE_UNDECLARED",
                "TMA_DESCRIPTOR_UNDECLARED",
                "TMA_DESCRIPTOR_UNDECLARED",
            ],
        )

    def test_the_two_schedules_differ_only_in_commitments(self) -> None:
        """The declared variant must not change semantics, only make them explicit."""

        retained = json.loads(RETAINED.read_text(encoding="utf-8"))
        declared = json.loads(DECLARED.read_text(encoding="utf-8"))
        for document in (retained, declared):
            document.pop("schedule_id")
            document.pop("tile_loops", None)
            for allocation in document["allocations"]:
                allocation.pop("tensor_columns", None)
            for buffer in document["buffers"]:
                buffer.pop("swizzle", None)
            for operation in document["operations"]:
                for field in (
                    "instruction",
                    "tile_shape",
                    "descriptor_box",
                    "subtile",
                    "source_atom",
                ):
                    operation["parameters"].pop(field, None)
        self.assertEqual(retained, declared)


class CommitmentsNowDerivableTest(unittest.TestCase):
    """What the artifact hardcodes that the declared Schedule now determines."""

    def setUp(self) -> None:
        self.schedule = Schedule.load(DECLARED)
        self.source = ARTIFACT.read_text(encoding="utf-8")

    def test_warp_identity_and_cta_width(self) -> None:
        roles = {role.name: role.warps for role in self.schedule.roles}
        self.assertEqual(roles["epilogue"], (0, 1, 2, 3))  # EPILOGUE_WARPS
        self.assertEqual(roles["mma"], (4,))  # MMA_WARP
        self.assertEqual(roles["tma"], (5,))  # TMA_WARP
        self.assertEqual(roles["reduce"], (6,))  # REDUCE_WARP
        self.assertIn("THREADS_PER_CTA = 224", self.source)
        self.assertEqual(self.schedule.total_warp_extent * 32, 224)
        self.assertIn("EPILOGUE_THREADS = 128", self.source)
        self.assertEqual(len(roles["epilogue"]) * 32, 128)

    def test_pipeline_depth(self) -> None:
        self.assertIn("PIPELINE_STAGES = 2", self.source)
        self.assertEqual(self.schedule.pipelines[0].stages, 2)

    def test_the_mma_atom_is_one_commitment(self) -> None:
        """Five module-level decisions in the artifact; one object in the Schedule."""

        for marker in (
            "MMA_TILE = (128, 256, 64)",
            "MMA_INSTRUCTION_SHAPE = (128, 256, 16)",
            "tcgen05.CtaGroup.ONE",
            "tcgen05.OperandSource.SMEM",
            "tcgen05.OperandMajorMode.K",
        ):
            self.assertIn(marker, self.source)

        mma = self.schedule.operation("dot_mma")
        assert mma is not None
        atom = mma.parameters.instruction
        assert atom is not None
        self.assertEqual(mma.parameters.tile_shape, (128, 256, 64))
        self.assertEqual(atom.shape, (128, 256, 16))
        self.assertEqual(atom.cta_group, 1)
        self.assertIs(atom.operand_source, OperandSource.SHARED)
        self.assertEqual(atom.operand_major, (OperandMajorMode.K, OperandMajorMode.K))
        self.assertIn(atom.contract, TARGET.instruction_contracts)
        # the tile is four atom steps deep in K
        self.assertEqual(mma.parameters.tile_shape[2] // atom.shape[2], 4)

    def test_tensor_memory_column_range(self) -> None:
        """The artifact reserves the whole array for a half-sized accumulator."""

        self.assertIn("tmem.allocate(512)", self.source)
        allocation = next(
            a for a in self.schedule.allocations if a.space is MemorySpace.TENSOR
        )
        self.assertEqual(allocation.tensor_columns, 256)
        self.assertEqual(allocation.size_bytes // TMEM_COLUMN_BYTES, 256)
        self.assertEqual(
            TARGET.resource_limits.maximum_tensor_memory_bytes // TMEM_COLUMN_BYTES, 512
        )

    def test_swizzle_and_descriptor_boxes(self) -> None:
        operands = {
            b.name: b.swizzle
            for b in self.schedule.buffers
            if b.space is MemorySpace.SHARED
        }
        self.assertEqual(
            operands, {"token_stage": Swizzle.B128, "centroid_stage": Swizzle.B128}
        )
        for operation in self.schedule.operations:
            if operation.kind is OperationKind.LOAD:
                self.assertIs(operation.parameters.movement, LoadMovement.TMA)
                staged = self.schedule.buffer(operation.writes[0])
                assert staged is not None
                self.assertEqual(operation.parameters.descriptor_box, staged.shape)


class EpilogueCommitmentTest(unittest.TestCase):
    """The last two decisions the artifact used to own alone."""

    def setUp(self) -> None:
        self.schedule = Schedule.load(DECLARED)
        self.source = ARTIFACT.read_text(encoding="utf-8")

    def test_the_epilogue_subtile_is_declared(self) -> None:
        self.assertIn("epilogue_tiler", self.source)
        epilogue = self.schedule.operation("distance_epilogue")
        assert epilogue is not None
        accumulator = self.schedule.buffer("accumulator")
        assert accumulator is not None
        subtile = epilogue.parameters.subtile
        self.assertEqual(subtile, (128, 64))
        # the artifact derives this as (size(acc, [0, 0]), size(acc, [0, 1]) // 4)
        self.assertEqual(subtile[0], accumulator.shape[0])
        self.assertEqual(subtile[1], accumulator.shape[1] // 4)

    def test_the_tmem_copy_atom_is_declared(self) -> None:
        self.assertIn("Ld32x32bOp", self.source)
        self.assertIn("Repetition.x64", self.source)
        epilogue = self.schedule.operation("distance_epilogue")
        assert epilogue is not None
        atom = epilogue.parameters.source_atom
        assert atom is not None
        self.assertEqual(atom.op, "tcgen05.Ld32x32b")
        self.assertEqual(atom.repetition, 64)


class SufficiencyTest(unittest.TestCase):
    """Every decision the artifact carries is now expressible in a Schedule.

    This does not mean an emitter exists -- only that writing one is no longer blocked
    on the representation. `lower()` is still a template stamp.
    """

    def test_the_declared_schedule_leaves_nothing_to_the_backend(self) -> None:
        findings = verify(Schedule.load(DECLARED), TARGET)
        self.assertEqual(findings, (), f"residual: {[str(f) for f in findings]}")

    def test_the_retained_schedule_shows_the_distance_travelled(self) -> None:
        """Nine commitments separate the retained Schedule from a complete one."""

        findings = verify(Schedule.load(RETAINED), TARGET)
        self.assertEqual([f for f in findings if f.blocks_lowering], [])
        self.assertEqual(len(findings), 9)

class LoopNestTest(unittest.TestCase):
    """The nest was the structural gap; a loop body may now name a nested loop."""

    def setUp(self) -> None:
        self.schedule = Schedule.load(DECLARED)
        self.source = ARTIFACT.read_text(encoding="utf-8")

    def test_the_nest_is_two_deep(self) -> None:
        self.assertEqual(self.schedule.loop_parent(), {"k_loop": "centroid_loop"})
        self.assertEqual(self.schedule.loop_depth("centroid_loop"), 0)
        self.assertEqual(self.schedule.loop_depth("k_loop"), 1)

    def test_a_body_may_mix_a_nested_loop_and_an_operation(self) -> None:
        """The epilogue runs once per centroid tile, beside the inner loop."""

        outer = self.schedule.tile_loop("centroid_loop")
        assert outer is not None
        self.assertEqual(outer.body, ("k_loop", "distance_epilogue"))

    def test_the_artifact_extents_are_now_derived(self) -> None:
        self.assertIn("NUM_CENTROID_TILES = 4", self.source)
        self.assertIn("NUM_K_TILES = 2", self.source)
        centroids = self.schedule.buffer("centroids")
        outer = self.schedule.tile_loop("centroid_loop")
        inner = self.schedule.tile_loop("k_loop")
        assert centroids is not None and outer is not None and inner is not None
        self.assertEqual(centroids.shape[outer.dimension] // outer.tile, 4)
        self.assertEqual(centroids.shape[inner.dimension] // inner.tile, 2)


if __name__ == "__main__":
    unittest.main()
