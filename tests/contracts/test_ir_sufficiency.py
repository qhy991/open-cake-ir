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

    def test_the_retained_schedule_declines_eight_commitments(self) -> None:
        findings = verify(Schedule.load(RETAINED), TARGET)
        self.assertEqual([f for f in findings if f.blocks_lowering], [])
        self.assertEqual(
            sorted(f.code for f in findings),
            [
                "ALLOCATION_TENSOR_COLUMNS_UNDECLARED",
                "BUFFER_SWIZZLE_UNDECLARED",
                "BUFFER_SWIZZLE_UNDECLARED",
                "MMA_INSTRUCTION_SHAPE_UNDECLARED",
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
                for field in ("instruction", "tile_shape", "instruction_shape", "descriptor_box"):
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

    def test_mma_tile_and_instruction_shape(self) -> None:
        self.assertIn("MMA_TILE = (128, 256, 64)", self.source)
        self.assertIn("MMA_INSTRUCTION_SHAPE = (128, 256, 16)", self.source)
        mma = self.schedule.operation("dot_mma")
        assert mma is not None
        self.assertEqual(mma.parameters.tile_shape, (128, 256, 64))
        self.assertEqual(mma.parameters.instruction_shape, (128, 256, 16))
        self.assertIn(mma.parameters.instruction, TARGET.instruction_contracts)

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


class ResidualGapTest(unittest.TestCase):
    """What the artifact still decides and no Schedule can currently say.

    Pinned so the list shrinks deliberately rather than being forgotten. Each entry is
    a decision present in the artifact with no representation in the IR.
    """

    RESIDUAL = {
        "loop_nest": (
            "for centroid_tile in range(4): for k_tile in range(2). tile_loops is a "
            "flat list whose body names operations, so a nest cannot be expressed."
        ),
        "cta_group": "tcgen05.CtaGroup.ONE -- one-SM versus two-SM cooperative MMA.",
        "operand_source": "tcgen05.OperandSource.SMEM -- operands from SMEM or TMEM.",
        "operand_major_mode": "OperandMajorMode.K for both A and B.",
        "epilogue_tiler": (
            "(size(acc, [0, 0]), size(acc, [0, 1]) // 4) -- the epilogue sub-tiling, "
            "including a magic divisor."
        ),
        "tmem_load_atom": "tcgen05.Ld32x32bOp(Repetition.x64) -- the TMEM copy atom.",
    }

    def test_the_residual_gap_is_six_decisions(self) -> None:
        self.assertEqual(len(self.RESIDUAL), 6)

    def test_each_residual_decision_is_present_in_the_artifact(self) -> None:
        source = ARTIFACT.read_text(encoding="utf-8")
        markers = {
            "loop_nest": "NUM_CENTROID_TILES",
            "cta_group": "tcgen05.CtaGroup.ONE",
            "operand_source": "tcgen05.OperandSource.SMEM",
            "operand_major_mode": "tcgen05.OperandMajorMode.K",
            "epilogue_tiler": "epilogue_tiler",
            "tmem_load_atom": "Ld32x32bOp",
        }
        self.assertEqual(set(markers), set(self.RESIDUAL))
        for key, marker in markers.items():
            with self.subTest(gap=key):
                self.assertIn(marker, source)

    def test_the_loop_nest_is_the_structural_one(self) -> None:
        """A flat `tile_loops` cannot carry the artifact's two-deep nest.

        The declared Schedule expresses only the inner loop. 1024 centroids in tiles of
        256 and 128 features in tiles of 64 are the two extents the artifact hardcodes
        as NUM_CENTROID_TILES and NUM_K_TILES.
        """

        schedule = Schedule.load(DECLARED)
        self.assertEqual([loop.name for loop in schedule.tile_loops], ["k_loop"])
        centroids = schedule.buffer("centroids")
        assert centroids is not None
        mma = schedule.operation("dot_mma")
        assert mma is not None and mma.parameters.tile_shape is not None
        self.assertEqual(centroids.shape[0] // mma.parameters.tile_shape[1], 4)
        self.assertEqual(centroids.shape[1] // mma.parameters.tile_shape[2], 2)
        self.assertEqual(schedule.tile_loops[0].tile, 64)


if __name__ == "__main__":
    unittest.main()
