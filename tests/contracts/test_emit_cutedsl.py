"""Contract tests for CuTe-DSL emission.

The hand-written artifact carries thirteen module-level constants. Each one is a
decision, and each is now computed from the Schedule rather than typed by hand. These
tests compare the emitter's derivation against those constants directly, so the two
cannot drift apart silently.

Scope: this fixes the derivation and the structural scaffolding. Operation bodies are
still placeholders, so the emitted source is not yet a runnable kernel and no test here
claims otherwise.
"""

from __future__ import annotations

import ast
import json
import re
import unittest
from pathlib import Path

from open_cake_ir.compiler.emit_cutedsl import EmitError, emit
from open_cake_ir.compiler.ir import BarrierMechanism, Schedule
from open_cake_ir.compiler.target import Target

ROOT = Path(__file__).resolve().parents[2]
TARGET = Target.load(ROOT / "compiler" / "targets" / "sm_100a.json")
DECLARED = ROOT / "examples" / "schedules" / "flash-kmeans-assignment-full-declared.json"
RETAINED = ROOT / "corpus" / "schedules" / "flash-kmeans-assignment-full.json"
ARTIFACT = (
    ROOT
    / "src"
    / "open_cake_ir"
    / "compiler"
    / "assets"
    / "flash_kmeans_assignment_full.py.tmpl"
)

# Artifact constant -> the name the emitter derives it under. The two loop trip counts
# are renamed because the emitter names them after the declared loops.
CONSTANT_MAP = {
    "IO_DTYPE": "IO_DTYPE",
    "ACC_DTYPE": "ACC_DTYPE",
    "MMA_INSTRUCTION_SHAPE": "MMA_INSTRUCTION_SHAPE",
    "MMA_TILE": "MMA_TILE",
    "PIPELINE_STAGES": "PIPELINE_STAGES",
    "THREADS_PER_CTA": "THREADS_PER_CTA",
    "EPILOGUE_THREADS": "EPILOGUE_THREADS",
    "EPILOGUE_WARPS": "EPILOGUE_WARPS",
    "MMA_WARP": "MMA_WARP",
    "TMA_WARP": "TMA_WARP",
    "REDUCE_WARP": "REDUCE_WARP",
    "NUM_K_TILES": "NUM_K_LOOP_TRIPS",
    "NUM_CENTROID_TILES": "NUM_CENTROID_LOOP_TRIPS",
}


def _artifact_constants() -> dict[str, str]:
    source = ARTIFACT.read_text(encoding="utf-8")
    return dict(re.findall(r"^([A-Z][A-Z0-9_]*) = (.+)$", source, re.M))


def _rendered(value: object) -> str:
    return value if isinstance(value, str) else repr(value)


class DerivationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.emission = emit(Schedule.load(DECLARED), TARGET)
        self.artifact = _artifact_constants()

    def test_every_artifact_constant_is_derived(self) -> None:
        for hardcoded, derived in CONSTANT_MAP.items():
            with self.subTest(constant=hardcoded):
                self.assertIn(hardcoded, self.artifact)
                self.assertIn(derived, self.emission.constants)
                self.assertEqual(
                    self.artifact[hardcoded],
                    _rendered(self.emission.constants[derived]),
                )

    def test_the_derivation_covers_thirteen_decisions(self) -> None:
        self.assertEqual(len(CONSTANT_MAP), 13)

    def test_tensor_memory_columns_are_derived_not_hardcoded(self) -> None:
        """The artifact writes `tmem.allocate(512)`; the Schedule says 256."""

        self.assertIn("tmem.allocate(512)", ARTIFACT.read_text(encoding="utf-8"))
        self.assertEqual(self.emission.constants["TMEM_COLUMNS"], 256)


class StructureTest(unittest.TestCase):
    def setUp(self) -> None:
        self.schedule = Schedule.load(DECLARED)
        self.source = emit(self.schedule, TARGET).source

    def test_the_emitted_source_is_valid_python(self) -> None:
        ast.parse(self.source)

    def test_shared_storage_holds_only_mbarrier_state(self) -> None:
        """A named CTA barrier carries no shared-memory state; an mbarrier does."""

        block = re.search(r"class SharedStorage:\n(.*?)\n\n", self.source, re.S)
        assert block is not None
        body = block.group(1)
        for barrier in self.schedule.barriers:
            field = f"{barrier.name}_mbarriers"
            with self.subTest(barrier=barrier.name):
                if barrier.mechanism is BarrierMechanism.MBARRIER:
                    self.assertIn(field, body)
                else:
                    self.assertNotIn(field, body)
        self.assertIn("tmem_holding_buffer", body)

    def test_a_pipelined_barrier_is_sized_by_its_pipeline(self) -> None:
        self.assertIn(
            "tiles_ready_mbarriers: cute.struct.MemRange[cutlass.Int64, PIPELINE_STAGES * 2]",
            self.source,
        )
        self.assertIn(
            "accumulator_ready_mbarriers: cute.struct.MemRange[cutlass.Int64, 2]",
            self.source,
        )

    def test_warp_dispatch_comes_from_the_roles(self) -> None:
        for role in self.schedule.roles:
            with self.subTest(role=role.name):
                if len(role.warps) == 1:
                    self.assertIn(f"warp_idx == {role.name.upper()}_WARP", self.source)
                else:
                    low, high = min(role.warps), max(role.warps)
                    self.assertIn(f"{low} <= warp_idx <= {high}", self.source)

    def test_every_operation_is_marked(self) -> None:
        for operation in self.schedule.operations:
            with self.subTest(operation=operation.op_id):
                self.assertIn(f"# CAKE_OP:{operation.op_id}", self.source)

    def test_identifiers_trace_back_to_declarations(self) -> None:
        for buffer in self.schedule.buffers:
            if buffer.swizzle is not None:
                self.assertIn(buffer.name, self.source)
        for load in self.schedule.operations:
            if load.op_id.startswith("load_"):
                self.assertIn(f"{load.op_id}_atom", self.source)
                self.assertIn(f"{load.op_id}_smem_layout", self.source)

    def test_the_atom_commitment_reaches_the_host_function(self) -> None:
        atom = self.schedule.operation("dot_mma").parameters.instruction
        self.assertEqual(atom.cta_group, 1)
        self.assertIn("tcgen05.CtaGroup.ONE", self.source)
        self.assertIn("tcgen05.OperandSource.SMEM", self.source)
        self.assertIn("tcgen05.OperandMajorMode.K", self.source)


class UnderSpecificationTest(unittest.TestCase):
    """Emission fails loudly rather than inventing a decision the Schedule declined."""

    def _without(self, mutate) -> Schedule:
        document = json.loads(DECLARED.read_text(encoding="utf-8"))
        mutate(document)
        return Schedule.from_dict(document)

    def test_the_retained_schedule_cannot_be_emitted(self) -> None:
        with self.assertRaises(EmitError):
            emit(Schedule.load(RETAINED), TARGET)

    def test_a_missing_atom_is_refused(self) -> None:
        schedule = self._without(
            lambda d: [
                o["parameters"].pop("instruction")
                for o in d["operations"]
                if o["kind"] == "mma"
            ]
        )
        with self.assertRaisesRegex(EmitError, "instruction atom"):
            emit(schedule, TARGET)

    def test_a_missing_descriptor_box_is_refused(self) -> None:
        schedule = self._without(
            lambda d: [
                o["parameters"].pop("descriptor_box")
                for o in d["operations"]
                if o["kind"] == "load"
            ]
        )
        with self.assertRaisesRegex(EmitError, "descriptor box"):
            emit(schedule, TARGET)

    def test_a_missing_barrier_mechanism_is_refused(self) -> None:
        schedule = self._without(
            lambda d: [b.pop("mechanism") for b in d["barriers"]]
        )
        with self.assertRaisesRegex(EmitError, "how it is realized"):
            emit(schedule, TARGET)

    def test_a_tile_that_does_not_divide_its_extent_is_refused(self) -> None:
        def bad_tile(document: dict) -> None:
            for loop in document["tile_loops"]:
                if loop["name"] == "k_loop":
                    loop["tile"] = 48

        with self.assertRaisesRegex(EmitError, "does not divide extent"):
            emit(self._without(bad_tile), TARGET)


if __name__ == "__main__":
    unittest.main()
