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
from open_cake_ir.compiler.verifier import Finding, FindingCategory, verify

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


class QuietOnValidScheduleTest(unittest.TestCase):
    def test_every_retained_schedule_verifies_clean(self) -> None:
        for path in CORPUS + [ROOT / "examples" / "gpu" / "flash-kmeans-b32-smoke.json"]:
            with self.subTest(schedule=path.name):
                self.assertEqual(verify(Schedule.load(path), TARGET), ())

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
                self.assertEqual(verify(schedule, TARGET), ())

    def test_findings_are_deterministically_ordered(self) -> None:
        schedule = _mutated(
            B32,
            lambda d: d["roles"][0].update(warps=[0, 1, 4096, 99999]),
        )
        first = verify(schedule, TARGET)
        self.assertEqual(first, verify(schedule, TARGET))
        keys = [(f.category.value, f.code, f.path) for f in first]
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
