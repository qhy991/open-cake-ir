"""Contract tests for the Schedule verifier.

The verifier is the pre-compile hard gate. Its contract has three parts:

1. it is quiet on well-formed Schedules -- every retained Schedule verifies clean, so a
   finding always means something;
2. it covers the four contract classes the paper's harness reports, and each class
   actually fires on a Schedule that violates it;
3. every finding is localized -- it names the offending path and the violated contract,
   which is what makes it a repair target rather than a rejection notice.
"""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from open_cake_ir.compiler.ir import Schedule
from open_cake_ir.compiler.target import Target
from open_cake_ir.compiler.verifier import (
    Finding,
    FindingCategory,
    FindingSeverity,
    verify,
)

ROOT = Path(__file__).resolve().parents[2]
TARGET = Target.load(ROOT / "compiler" / "targets" / "sm_100a.json")
CORPUS = sorted((ROOT / "corpus" / "schedules").glob("*.json"))
B32 = ROOT / "corpus" / "schedules" / "flash-kmeans-b32-smoke.json"
TINYGEMM = ROOT / "corpus" / "schedules" / "tinygemm2-stage4-split-k.json"
ASSIGNMENT_FULL = ROOT / "corpus" / "schedules" / "flash-kmeans-assignment-full.json"


def _mutated(path: Path, mutate) -> Schedule:
    document = json.loads(path.read_text(encoding="utf-8"))
    mutate(document)
    return Schedule.from_dict(document)


def _codes(findings: tuple[Finding, ...]) -> set[str]:
    return {finding.code for finding in findings}


def _blocking(findings: tuple[Finding, ...]) -> tuple[Finding, ...]:
    return tuple(finding for finding in findings if finding.blocks_lowering)


class QuietOnValidScheduleTest(unittest.TestCase):
    def test_every_retained_schedule_verifies_clean(self) -> None:
        for path in CORPUS + [ROOT / "examples" / "gpu" / "flash-kmeans-b32-smoke.json"]:
            with self.subTest(schedule=path.name):
                self.assertEqual(_blocking(verify(Schedule.load(path), TARGET)), ())

    def test_shape_drift_is_semantic_not_structural(self) -> None:
        """The drift corpus cases are wrong about shapes, not about structure.

        They must stay clean here so that a structural finding keeps its meaning; the
        shape contract belongs to the Workload, not to the Schedule verifier.
        """

        for name in (
            "flash-kmeans-b32-smoke-shape-drift.json",
            "flash-kmeans-assignment-full-shape-drift.json",
        ):
            with self.subTest(schedule=name):
                schedule = Schedule.load(ROOT / "corpus" / "schedules" / name)
                self.assertEqual(_blocking(verify(schedule, TARGET)), ())

    def test_findings_are_deterministically_ordered(self) -> None:
        schedule = _mutated(
            B32,
            lambda d: d["roles"][0].update(warps=[0, 1, 4096, 99999]),
        )
        first = verify(schedule, TARGET)
        self.assertEqual(first, verify(schedule, TARGET))
        keys = [
            (f.severity is not FindingSeverity.BLOCKING, f.category.value, f.code, f.path)
            for f in first
        ]
        self.assertEqual(keys, sorted(keys))

    def test_verification_never_raises(self) -> None:
        schedule = _mutated(
            B32,
            lambda d: (
                d["operations"][0].update(role="ghost", reads=["nope"], writes=["nope"]),
                d.update(outputs=["absent"]),
            ),
        )
        self.assertTrue(verify(schedule, TARGET))


class HardwareConformanceTest(unittest.TestCase):
    def test_warp_index_beyond_the_target_range_is_reported(self) -> None:
        """`len(warps)` treats a warp id as a count; the range check does not."""

        findings = verify(
            _mutated(B32, lambda d: d["roles"][0].update(warps=[0, 1, 4096, 99999])),
            TARGET,
        )
        self.assertIn("ROLE_WARP_RANGE", _codes(findings))
        self.assertIn("TARGET_WARP_LIMIT", _codes(findings))
        self.assertIn("TARGET_THREAD_LIMIT", _codes(findings))
        ranged = [f for f in findings if f.code == "ROLE_WARP_RANGE"]
        self.assertEqual([f.path for f in ranged], ["roles[0].warps[2]", "roles[0].warps[3]"])
        self.assertIn("outside the Target CTA range [0, 32)", ranged[0].message)

    def test_shared_memory_budget(self) -> None:
        findings = verify(
            _mutated(
                ASSIGNMENT_FULL,
                lambda d: d["allocations"][0].update(size_bytes=999_999),
            ),
            TARGET,
        )
        self.assertIn("TARGET_SHARED_MEMORY_LIMIT", _codes(findings))

    def test_target_mismatch_short_circuits_with_no_silent_fallback(self) -> None:
        findings = verify(_mutated(B32, lambda d: d.update(target="sm_90a")), TARGET)
        self.assertEqual(_codes(findings), {"TARGET_UNSUPPORTED"})
        self.assertIn("no silent architecture fallback", findings[0].message)

    def test_allocation_cannot_be_global(self) -> None:
        findings = verify(
            _mutated(
                ASSIGNMENT_FULL, lambda d: d["allocations"][0].update(space="global")
            ),
            TARGET,
        )
        self.assertIn("ALLOCATION_GLOBAL", _codes(findings))


class DataConsistencyTest(unittest.TestCase):
    def test_buffers_sharing_an_allocation_must_not_overlap(self) -> None:
        """Hand-authored byte offsets are the only thing keeping these tiles apart."""

        def alias(document: dict) -> None:
            for buffer in document["buffers"]:
                if buffer["name"] == "activation_stage":
                    buffer["byte_offset"] = 1024

        findings = verify(_mutated(TINYGEMM, alias), TARGET)
        self.assertIn("BUFFER_VIEW_OVERLAP", _codes(findings))
        overlap = next(f for f in findings if f.code == "BUFFER_VIEW_OVERLAP")
        self.assertIn("overlapping", overlap.message)
        self.assertIs(overlap.category, FindingCategory.DATA_CONSISTENCY)

    def test_writing_an_input_buffer_is_reported(self) -> None:
        """The static form of a candidate mutating the buffers its oracle is built from."""

        findings = verify(
            _mutated(
                B32,
                lambda d: d["operations"][0].update(writes=["token_tile", "centroids"]),
            ),
            TARGET,
        )
        self.assertIn("INPUT_WRITTEN", _codes(findings))
        written = next(f for f in findings if f.code == "INPUT_WRITTEN")
        self.assertIn("read-only for the whole Schedule", written.message)

    def test_declared_output_must_be_written_and_exported(self) -> None:
        unwritten = verify(
            _mutated(B32, lambda d: d["operations"][4].update(writes=[])), TARGET
        )
        self.assertIn("OUTPUT_UNWRITTEN", _codes(unwritten))
        unexported = verify(_mutated(B32, lambda d: d.update(outputs=[])), TARGET)
        self.assertIn("OUTPUT_NOT_EXPORTED", _codes(unexported))

    def test_dependency_cycles_are_detected(self) -> None:
        def cycle(document: dict) -> None:
            document["operations"][0]["depends_on"] = ["store_assignment"]

        findings = verify(_mutated(B32, cycle), TARGET)
        self.assertIn("OP_DEPENDENCY_CYCLE", _codes(findings))

    def test_load_source_must_be_global(self) -> None:
        findings = verify(
            _mutated(B32, lambda d: d["operations"][0].update(reads=["token_tile"])),
            TARGET,
        )
        self.assertIn("OP_LOAD_SOURCE", _codes(findings))

    def test_access_map_rank_must_match_the_buffer(self) -> None:
        findings = verify(
            _mutated(
                B32,
                lambda d: d["access_maps"][0]["indices"].append(
                    {"source": "dimension", "dimension": 2}
                ),
            ),
            TARGET,
        )
        self.assertIn("ACCESS_RANK", _codes(findings))

    def test_access_map_index_sources_must_resolve(self) -> None:
        unknown_axis = verify(
            _mutated(
                B32,
                lambda d: d["access_maps"][0]["indices"][0].update(name="ghost_axis"),
            ),
            TARGET,
        )
        self.assertIn("ACCESS_PROGRAM_AXIS_UNKNOWN", _codes(unknown_axis))
        unknown_loop = verify(
            _mutated(
                B32,
                lambda d: d["access_maps"][1]["indices"][1].update(name="ghost_loop"),
            ),
            TARGET,
        )
        self.assertIn("ACCESS_LOOP_UNKNOWN", _codes(unknown_loop))

    def test_global_buffer_access_requires_an_addressing_rule(self) -> None:
        findings = verify(
            _mutated(B32, lambda d: d["access_maps"].pop(0)), TARGET
        )
        self.assertIn("ACCESS_MAP_MISSING", _codes(findings))


class ProgramSafetyTest(unittest.TestCase):
    """Synchronization is the category the string-keyed compiler never reports.

    `TargetDefinition.synchronization_contracts` is parsed there and read by no rule,
    and barrier `count` / `producers` / `consumers` are parsed and dropped.
    """

    @staticmethod
    def _desynchronized(document: dict) -> None:
        document["barriers"] = []
        for operation in document["operations"]:
            operation.pop("waits", None)
            operation.pop("signals", None)

    def test_cross_role_read_without_a_barrier_is_a_race(self) -> None:
        findings = verify(_mutated(ASSIGNMENT_FULL, self._desynchronized), TARGET)
        races = [f for f in findings if f.code == "OP_CROSS_ROLE_RACE"]
        self.assertEqual(len(races), 4)
        self.assertEqual(
            [f.path for f in races],
            [
                "operations[2].reads",
                "operations[2].reads",
                "operations[3].reads",
                "operations[4].reads",
            ],
        )
        for finding in races:
            self.assertIs(finding.category, FindingCategory.PROGRAM_SAFETY)

    def test_depends_on_does_not_synchronize_warps(self) -> None:
        """Every cross-role edge here keeps its `depends_on`; only the barrier is gone.

        `depends_on` is program order, which is meaningful inside one warp role and
        meaningless between two roles running concurrently. The finding says so.
        """

        findings = verify(_mutated(ASSIGNMENT_FULL, self._desynchronized), TARGET)
        race = next(f for f in findings if f.code == "OP_CROSS_ROLE_RACE")
        self.assertIn("declares depends_on", race.message)
        self.assertIn("does not synchronize warps", race.message)

    def test_same_role_ordering_is_accepted(self) -> None:
        """`store_assignment` reads what `argmin` writes and both are role `reduce`."""

        findings = verify(_mutated(ASSIGNMENT_FULL, self._desynchronized), TARGET)
        self.assertNotIn(
            "store_assignment", " ".join(f.message for f in findings if f.code == "OP_CROSS_ROLE_RACE")
        )

    def test_signalling_role_must_be_a_declared_producer(self) -> None:
        def retarget(document: dict) -> None:
            for barrier in document["barriers"]:
                if barrier["name"] == "tiles_ready":
                    barrier["producers"] = ["reduce"]

        findings = verify(_mutated(ASSIGNMENT_FULL, retarget), TARGET)
        self.assertIn("OP_BARRIER_PRODUCER", _codes(findings))

    def test_waiting_role_must_be_a_declared_consumer(self) -> None:
        def retarget(document: dict) -> None:
            for barrier in document["barriers"]:
                if barrier["name"] == "accumulator_ready":
                    barrier["consumers"] = ["reduce"]

        findings = verify(_mutated(ASSIGNMENT_FULL, retarget), TARGET)
        consumer = next(f for f in findings if f.code == "OP_BARRIER_CONSUMER")
        self.assertEqual(consumer.path, "operations[3].waits")
        self.assertIn("not one of its declared consumers", consumer.message)

    def test_a_barrier_nobody_signals_never_completes(self) -> None:
        def orphan(document: dict) -> None:
            for operation in document["operations"]:
                operation["signals"] = []

        findings = verify(_mutated(ASSIGNMENT_FULL, orphan), TARGET)
        self.assertIn("BARRIER_UNUSED_PRODUCER", _codes(findings))

    def test_barrier_endpoints_must_resolve(self) -> None:
        findings = verify(
            _mutated(
                ASSIGNMENT_FULL,
                lambda d: d["barriers"][0].update(producers=["ghost_role"]),
            ),
            TARGET,
        )
        self.assertIn("BARRIER_ROLE_UNKNOWN", _codes(findings))


class CategoryCoverageTest(unittest.TestCase):
    def test_all_four_paper_contract_classes_are_reachable(self) -> None:
        observed = set()
        cases = [
            (B32, lambda d: d["roles"][0].update(warps=[0, 4096])),
            (B32, lambda d: d["operations"][0].update(writes=["token_tile", "centroids"])),
            (ASSIGNMENT_FULL, ProgramSafetyTest._desynchronized),
            (B32, lambda d: d["roles"].append(dict(d["roles"][0]))),
        ]
        for path, mutate in cases:
            for finding in verify(_mutated(path, mutate), TARGET):
                observed.add(finding.category)
        self.assertEqual(observed, set(FindingCategory))


if __name__ == "__main__":
    unittest.main()


class HardwareCommitmentTest(unittest.TestCase):
    """The paper requires the agent to write down five concrete hardware commitments.

    "an SMEM view offset, an operand byte offset, a TMEM column range, a swizzle tag, a
    TMA descriptor coordinate" (arXiv:2608.12629v1 S2). The Schedule schema carried the
    first two as `byte_offset`; the rest had no representation, so the backend chose and
    the choice was neither inspectable nor verifiable. These rules report the gap.
    """

    def _advisory(self, path: Path) -> dict[str, Finding]:
        return {
            finding.code: finding
            for finding in verify(Schedule.load(path), TARGET)
            if not finding.blocks_lowering
        }

    def test_retained_schedules_report_their_missing_commitments(self) -> None:
        advisory = self._advisory(ASSIGNMENT_FULL)
        self.assertEqual(
            set(advisory),
            {
                "ALLOCATION_TENSOR_COLUMNS_UNDECLARED",
                "BUFFER_SWIZZLE_UNDECLARED",
                "MMA_INSTRUCTION_SHAPE_UNDECLARED",
                "MMA_INSTRUCTION_UNDECLARED",
                "MMA_TILE_UNDECLARED",
                "TMA_DESCRIPTOR_UNDECLARED",
            },
        )
        for finding in advisory.values():
            self.assertIs(finding.severity, FindingSeverity.HINT)
            self.assertIs(finding.category, FindingCategory.HARDWARE_CONFORMANCE)

    def test_tensor_memory_is_reported_in_columns_not_bytes(self) -> None:
        """The artifact allocates 512 columns for a 256-column accumulator.

        `tmem.allocate(512)` takes the whole array while the Allocation is 131072 bytes,
        which is 256 columns. Nothing related the two units, so the over-allocation --
        and the halved TMEM occupancy it causes -- was invisible.
        """

        finding = self._advisory(ASSIGNMENT_FULL)["ALLOCATION_TENSOR_COLUMNS_UNDECLARED"]
        self.assertIn("131072 bytes implies 256 columns", finding.message)

    def test_a_declared_column_range_must_match_the_byte_size(self) -> None:
        findings = verify(
            _mutated(
                ASSIGNMENT_FULL,
                lambda d: d["allocations"][1].update(tensor_columns=512),
            ),
            TARGET,
        )
        mismatch = next(
            f for f in findings if f.code == "ALLOCATION_TENSOR_COLUMNS_MISMATCH"
        )
        self.assertTrue(mismatch.blocks_lowering)
        self.assertIn("512 columns but its 131072 bytes are 256 columns", mismatch.message)

    def test_a_matching_column_range_clears_the_hint(self) -> None:
        findings = verify(
            _mutated(
                ASSIGNMENT_FULL,
                lambda d: d["allocations"][1].update(tensor_columns=256),
            ),
            TARGET,
        )
        self.assertNotIn("ALLOCATION_TENSOR_COLUMNS_UNDECLARED", _codes(findings))
        self.assertNotIn("ALLOCATION_TENSOR_COLUMNS_MISMATCH", _codes(findings))
        self.assertEqual(_blocking(findings), ())

    def test_column_range_is_bounded_by_the_target(self) -> None:
        def oversize(document: dict) -> None:
            document["allocations"][1].update(
                size_bytes=1024 * 128 * 4, tensor_columns=1024
            )

        findings = verify(_mutated(ASSIGNMENT_FULL, oversize), TARGET)
        self.assertIn("TARGET_TENSOR_COLUMN_LIMIT", _codes(findings))

    def test_declared_instruction_must_be_admitted_by_the_target(self) -> None:
        good = verify(
            _mutated(
                ASSIGNMENT_FULL,
                lambda d: d["operations"][2]["parameters"].update(
                    instruction="tcgen05.mma.cta_group::1.kind::f16"
                ),
            ),
            TARGET,
        )
        self.assertNotIn("MMA_INSTRUCTION_UNDECLARED", _codes(good))
        self.assertNotIn("TARGET_INSTRUCTION_UNSUPPORTED", _codes(good))

        bad = verify(
            _mutated(
                ASSIGNMENT_FULL,
                lambda d: d["operations"][2]["parameters"].update(
                    instruction="wgmma.mma_async.sync.aligned"
                ),
            ),
            TARGET,
        )
        unsupported = next(
            f for f in bad if f.code == "TARGET_INSTRUCTION_UNSUPPORTED"
        )
        self.assertTrue(unsupported.blocks_lowering)

    def test_mma_tile_must_match_its_accumulator(self) -> None:
        matching = verify(
            _mutated(
                ASSIGNMENT_FULL,
                lambda d: d["operations"][2]["parameters"].update(
                    tile_shape=[128, 256, 64]
                ),
            ),
            TARGET,
        )
        self.assertEqual(_blocking(matching), ())

        mismatched = verify(
            _mutated(
                ASSIGNMENT_FULL,
                lambda d: d["operations"][2]["parameters"].update(
                    tile_shape=[128, 128, 64]
                ),
            ),
            TARGET,
        )
        finding = next(
            f for f in mismatched if f.code == "MMA_TILE_ACCUMULATOR_MISMATCH"
        )
        self.assertIn("128x128 does not match accumulator", finding.message)

    def test_swizzle_commitment_clears_its_hint(self) -> None:
        def commit(document: dict) -> None:
            for buffer in document["buffers"]:
                if buffer["name"] in ("token_stage", "centroid_stage"):
                    buffer["swizzle"] = "swizzle_128b"

        findings = verify(_mutated(ASSIGNMENT_FULL, commit), TARGET)
        self.assertNotIn("BUFFER_SWIZZLE_UNDECLARED", _codes(findings))


class LoopNestRuleTest(unittest.TestCase):
    DECLARED = ROOT / "examples" / "schedules" / "flash-kmeans-assignment-full-declared.json"

    def _mutate(self, mutate) -> tuple[Finding, ...]:
        document = json.loads(self.DECLARED.read_text(encoding="utf-8"))
        mutate(document)
        return verify(Schedule.from_dict(document), TARGET)

    def test_the_declared_nest_is_accepted(self) -> None:
        self.assertEqual(verify(Schedule.load(self.DECLARED), TARGET), ())

    def test_a_body_entry_must_resolve(self) -> None:
        findings = self._mutate(lambda d: d["tile_loops"][0]["body"].append("ghost"))
        self.assertIn("LOOP_BODY_UNKNOWN", _codes(findings))

    def test_a_loop_cannot_contain_itself(self) -> None:
        findings = self._mutate(
            lambda d: d["tile_loops"][0]["body"].append("centroid_loop")
        )
        self.assertIn("LOOP_NEST_SELF", _codes(findings))

    def test_nesting_cycles_are_detected(self) -> None:
        findings = self._mutate(
            lambda d: d["tile_loops"][1]["body"].append("centroid_loop")
        )
        self.assertIn("LOOP_NEST_CYCLE", _codes(findings))

    def test_a_loop_has_one_parent(self) -> None:
        def two_parents(document: dict) -> None:
            rival = dict(document["tile_loops"][0])
            rival["name"] = "rival_loop"
            rival["iterator"] = "rival_tile"
            rival["body"] = ["k_loop"]
            document["tile_loops"].append(rival)

        findings = self._mutate(two_parents)
        self.assertIn("LOOP_NEST_MULTIPLE_PARENTS", _codes(findings))

    def test_an_operation_belongs_to_one_scope(self) -> None:
        findings = self._mutate(
            lambda d: d["tile_loops"][0]["body"].append("load_tokens")
        )
        self.assertIn("LOOP_OPERATION_MULTIPLE_SCOPE", _codes(findings))

    def test_body_operations_follow_declaration_order(self) -> None:
        findings = self._mutate(
            lambda d: d["tile_loops"][1].update(
                body=["dot_mma", "load_tokens", "load_centroids"]
            )
        )
        self.assertIn("LOOP_BODY_ORDER", _codes(findings))
