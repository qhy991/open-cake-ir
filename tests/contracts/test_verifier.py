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

from open_cake_ir.compiler.ir import Schedule, ScheduleParseError
from open_cake_ir.compiler.target import Target
from open_cake_ir.compiler.verifier import (
    Finding,
    FindingCategory,
    FindingSeverity,
    verify,
)

ROOT = Path(__file__).resolve().parents[2]
TARGET = Target.load(ROOT / "compiler" / "targets" / "sm_100a.json")
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


def _mutated(path: Path, mutate) -> Schedule:
    document = json.loads(path.read_text(encoding="utf-8"))
    mutate(document)
    return Schedule.from_dict(document)


def _codes(findings: tuple[Finding, ...]) -> set[str]:
    return {finding.code for finding in findings}


def _blocking(findings: tuple[Finding, ...]) -> tuple[Finding, ...]:
    return tuple(finding for finding in findings if finding.blocks_lowering)


def _actionable(findings: tuple[Finding, ...]) -> tuple[Finding, ...]:
    """Everything but reports.

    A report is bottleneck attribution the Schedule cannot be "wrong" about, so a test
    asserting that a Schedule has nothing to fix should not have to enumerate them.
    """

    return tuple(f for f in findings if f.severity is not FindingSeverity.REPORT)


def _op(document: dict, op_id: str) -> dict:
    """Address an operation by name.

    Index-addressed mutations silently retarget when a Schedule gains an operation, so
    the composed distance turned several of these into assertions about a different one.
    """

    return next(o for o in document["operations"] if o["id"] == op_id)


class QuietOnValidScheduleTest(unittest.TestCase):
    def test_every_retained_schedule_verifies_clean(self) -> None:
        # The manifest says which cases are meant to be accepted. A case that exists to
        # pin a rejection rule is not a counterexample to the retained ones being clean,
        # and hardcoding the exceptions here would put that fact in a second place.
        manifest = json.loads(
            (ROOT / "corpus" / "manifest.json").read_text(encoding="utf-8")
        )
        accepted = {
            ROOT / case["schedule"]
            for case in manifest["cases"]
            if case["expected"]["accepted"]
        }
        paths = sorted(accepted) + [
            ROOT / "examples" / "gpu" / "flash-kmeans-b32-smoke-v2.json"
        ]
        targets = json.loads(
            (ROOT / "compiler" / "revision.lock.json").read_text(encoding="utf-8")
        )["target_definitions"]
        for path in paths:
            with self.subTest(schedule=path.name):
                schedule = Schedule.load(path)
                target = Target.load(ROOT / targets[schedule.target]["path"])
                self.assertEqual(_blocking(verify(schedule, target)), ())

    def test_a_broadcast_axis_no_shape_rule_could_infer_is_checked(self) -> None:
        """The one arithmetic fact shapes cannot settle.

        A per-row scale spans axis 0 and a per-column weight spans axis 1. When the two
        extents differ a rule over shapes alone could guess; when they are equal nothing
        distinguishes them, so the Schedule declares which and this holds it to it.
        """

        import json

        document = json.loads(
            (ROOT / "corpus/schedules/rmsnorm-b8-smoke.json").read_text(encoding="utf-8")
        )
        for operation in document["operations"]:
            if operation["id"] == "weight":
                operation["parameters"]["broadcast_axis"] = 0
        codes = {f.code for f in _blocking(verify(Schedule.from_dict(document), TARGET))}
        self.assertIn("ELEMENTWISE_BROADCAST", codes)

    def test_shape_drift_is_semantic_not_structural(self) -> None:
        """The drift corpus cases are wrong about shapes, not about structure.

        They must stay clean here so that a structural finding keeps its meaning; the
        shape contract belongs to the Workload, not to the Schedule verifier.
        """

        for name in (
            "flash-kmeans-b32-smoke-shape-drift-v2.json",
            "flash-kmeans-assignment-full-shape-drift.json",
        ):
            with self.subTest(schedule=name):
                schedule = Schedule.load(ROOT / "corpus" / "schedules" / name)
                self.assertEqual(_blocking(verify(schedule, TARGET)), ())

    def test_findings_are_deterministically_ordered(self) -> None:
        schedule = _mutated(
            B32,
            lambda d: d["roles"][0].update(warps=[4096, 4097, 4098, 4099]),
        )
        first = verify(schedule, TARGET)
        self.assertEqual(first, verify(schedule, TARGET))
        order = {FindingSeverity.BLOCKING: 0, FindingSeverity.REPORT: 1, FindingSeverity.HINT: 2}
        keys = [(order[f.severity], f.category.value, f.code, f.path) for f in first]
        self.assertEqual(keys, sorted(keys))

    def test_verification_never_raises(self) -> None:
        schedule = _mutated(
            B32,
            lambda d: (
                _op(d, "load_tokens").update(role="ghost", reads=["nope"], writes=["nope"]),
                d.update(outputs=["absent"]),
            ),
        )
        self.assertTrue(verify(schedule, TARGET))


class ScheduleSemanticsTest(unittest.TestCase):
    def test_a_warp_cannot_belong_to_two_roles(self) -> None:
        findings = verify(
            _mutated(
                ASSIGNMENT_FULL,
                lambda d: d["roles"][1].update(warps=[3]),
            ),
            TARGET,
        )

        overlaps = [f for f in findings if f.code == "ROLE_WARP_OVERLAP"]
        self.assertEqual(len(overlaps), 1)
        self.assertEqual(overlaps[0].category, FindingCategory.SCHEDULE_SEMANTICS)
        self.assertEqual(overlaps[0].path, "roles[1].warps")
        self.assertIn("both role 'epilogue' and role 'mma'", overlaps[0].message)


class HardwareConformanceTest(unittest.TestCase):
    def test_current_top_k_lowering_refuses_non_power_of_two_k(self) -> None:
        schedule = _mutated(
            TOP_K,
            lambda d: _op(d, "select_experts")["parameters"].update(k=7),
        )
        finding = next(
            f for f in verify(schedule, TARGET) if f.code == "TOP_K_K_UNLOWERABLE"
        )
        self.assertIs(finding.category, FindingCategory.HARDWARE_CONFORMANCE)
        self.assertEqual(finding.path, "operations[1].parameters.k")

    def test_current_top_k_lowering_refuses_non_power_of_two_source(self) -> None:
        def change(document) -> None:
            next(buffer for buffer in document["buffers"] if buffer["name"] == "scores")[
                "shape"
            ] = [8, 192]
            next(
                buffer for buffer in document["buffers"] if buffer["name"] == "score_row"
            )["shape"] = [192]

        schedule = _mutated(TOP_K, change)
        finding = next(
            f for f in verify(schedule, TARGET) if f.code == "TOP_K_SOURCE_UNLOWERABLE"
        )
        self.assertIs(finding.category, FindingCategory.HARDWARE_CONFORMANCE)
        self.assertEqual(finding.path, "operations[1].reads")

    def test_warp_index_beyond_the_target_range_is_reported(self) -> None:
        """`len(warps)` treats a warp id as a count; the range check does not."""

        findings = verify(
            _mutated(
                B32,
                lambda d: d["roles"][0].update(
                    warps=[4096, 4097, 4098, 4099]
                ),
            ),
            TARGET,
        )
        self.assertIn("ROLE_WARP_RANGE", _codes(findings))
        self.assertIn("TARGET_WARP_LIMIT", _codes(findings))
        self.assertIn("TARGET_THREAD_LIMIT", _codes(findings))
        ranged = [f for f in findings if f.code == "ROLE_WARP_RANGE"]
        self.assertEqual(
            [f.path for f in ranged],
            [
                "roles[0].warps[0]",
                "roles[0].warps[1]",
                "roles[0].warps[2]",
                "roles[0].warps[3]",
            ],
        )
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
    def test_top_k_contract_failures_are_local(self) -> None:
        cases = (
            (
                "arity",
                lambda d: _op(d, "select_experts").update(writes=["top_indices"]),
                "TOP_K_ARITY",
            ),
            (
                "rank",
                lambda d: next(
                    b for b in d["buffers"] if b["name"] == "score_row"
                ).update(shape=[1, 256]),
                "TOP_K_SOURCE_RANK",
            ),
            (
                "extent",
                lambda d: _op(d, "select_experts")["parameters"].update(k=512),
                "TOP_K_K_OUT_OF_RANGE",
            ),
            (
                "dtype",
                lambda d: next(
                    b for b in d["buffers"] if b["name"] == "score_row"
                ).update(dtype="fp16"),
                "TOP_K_VALUE_DTYPE",
            ),
            (
                "shape",
                lambda d: next(
                    b for b in d["buffers"] if b["name"] == "top_indices"
                ).update(shape=[4]),
                "TOP_K_SHAPE_MISMATCH",
            ),
        )
        for label, mutate, code in cases:
            with self.subTest(contract=label):
                finding = next(
                    f for f in verify(_mutated(TOP_K, mutate), TARGET) if f.code == code
                )
                self.assertTrue(finding.blocks_lowering)
                self.assertRegex(finding.path, r"^operations\[1\]")

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
                lambda d: _op(d, "load_tokens").update(writes=["token_tile", "centroids"]),
            ),
            TARGET,
        )
        self.assertIn("INPUT_WRITTEN", _codes(findings))
        written = next(f for f in findings if f.code == "INPUT_WRITTEN")
        self.assertIn("read-only for the whole Schedule", written.message)

    def test_declared_output_must_be_written_and_exported(self) -> None:
        unwritten = verify(
            _mutated(B32, lambda d: _op(d, "store_assignment").update(writes=[])), TARGET
        )
        self.assertIn("OUTPUT_UNWRITTEN", _codes(unwritten))
        unexported = verify(_mutated(B32, lambda d: d.update(outputs=[])), TARGET)
        self.assertIn("OUTPUT_NOT_EXPORTED", _codes(unexported))

    def test_dependency_cycles_are_detected(self) -> None:
        def cycle(document: dict) -> None:
            document["operations"][0]["depends_on"] = ["store_assignment"]

        findings = verify(_mutated(B32, cycle), TARGET)
        self.assertIn("OP_DEPENDENCY_CYCLE", _codes(findings))
        # One cycle is one finding, and it says where to look. Six findings naming six
        # members with `operations` as the path is a set of names, not a diagnosis.
        cycles = [f for f in findings if f.code == "OP_DEPENDENCY_CYCLE"]
        self.assertEqual(len(cycles), 1)
        self.assertRegex(cycles[0].path, r"^operations\[\d+\]\.depends_on$")

    def test_load_source_must_be_global(self) -> None:
        findings = verify(
            _mutated(B32, lambda d: _op(d, "load_tokens").update(reads=["token_tile"])),
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

    def test_an_mbarrier_has_one_compatible_producer_kind(self) -> None:
        def mix_pipeline_kinds(document: dict) -> None:
            barrier = next(b for b in document["barriers"] if b["name"] == "tiles_ready")
            barrier["producers"].append("mma")
            mma = _op(document, "dot_mma")
            mma["signals"].append("tiles_ready")

        mixed = verify(_mutated(ASSIGNMENT_FULL, mix_pipeline_kinds), TARGET)
        self.assertIn("BARRIER_PIPELINE_KIND_AMBIGUOUS", _codes(mixed))

        def make_synchronous(document: dict) -> None:
            parameters = _op(document, "load_tokens")["parameters"]
            parameters["movement"] = "global"
            parameters.pop("descriptor_box")

        unsupported = verify(
            _mutated(ASSIGNMENT_FULL, make_synchronous),
            TARGET,
        )
        self.assertIn("BARRIER_PIPELINE_PRODUCER_UNSUPPORTED", _codes(unsupported))


class CategoryCoverageTest(unittest.TestCase):
    def test_all_four_paper_contract_classes_are_reachable(self) -> None:
        observed = set()
        cases = [
            (B32, lambda d: d["roles"][0].update(warps=[4096, 4097])),
            (B32, lambda d: _op(d, "load_tokens").update(writes=["token_tile", "centroids"])),
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

    @staticmethod
    def _strip(document: dict) -> None:
        """Remove every hardware commitment, leaving the Schedule semantically identical."""

        for allocation in document["allocations"]:
            allocation.pop("tensor_columns", None)
        for buffer in document["buffers"]:
            buffer.pop("swizzle", None)
        for operation in document["operations"]:
            for field in ("instruction", "tile_shape", "descriptor_box", "subtile", "source_atom"):
                operation["parameters"].pop(field, None)

    def _advisory(self, path: Path) -> dict[str, Finding]:
        """Advisory hardware-conformance findings; synchronization has its own test."""

        document = json.loads(path.read_text(encoding="utf-8"))
        self._strip(document)
        return {
            finding.code: finding
            for finding in verify(Schedule.from_dict(document), TARGET)
            if finding.severity is FindingSeverity.HINT
            and finding.category is FindingCategory.HARDWARE_CONFORMANCE
        }

    def test_a_declined_commitment_is_reported(self) -> None:
        advisory = self._advisory(ASSIGNMENT_FULL)
        self.assertEqual(
            set(advisory),
            {
                "ALLOCATION_TENSOR_COLUMNS_UNDECLARED",
                "BUFFER_SWIZZLE_UNDECLARED",
                "EPILOGUE_SOURCE_ATOM_UNDECLARED",
                "EPILOGUE_SUBTILE_UNDECLARED",
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
        # and the Schedule as retained makes that commitment
        self.assertEqual(_actionable(verify(Schedule.load(ASSIGNMENT_FULL), TARGET)), ())

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
                lambda d: _op(d, "dot_mma")["parameters"].update(
                    instruction={
                        "contract": "tcgen05.mma.cta_group::1.kind::f16",
                        "shape": [128, 256, 16],
                        "cta_group": 1,
                        "operand_source": "shared",
                        "operand_major": ["k", "k"],
                    }
                ),
            ),
            TARGET,
        )
        self.assertNotIn("MMA_INSTRUCTION_UNDECLARED", _codes(good))
        self.assertNotIn("TARGET_INSTRUCTION_UNSUPPORTED", _codes(good))

        bad = verify(
            _mutated(
                ASSIGNMENT_FULL,
                lambda d: _op(d, "dot_mma")["parameters"].update(
                    instruction={
                        "contract": "wgmma.mma_async.sync.aligned",
                        "shape": [64, 128, 16],
                        "cta_group": 1,
                        "operand_source": "shared",
                        "operand_major": ["k", "k"],
                    }
                ),
            ),
            TARGET,
        )
        unsupported = next(
            f for f in bad if f.code == "TARGET_INSTRUCTION_UNSUPPORTED"
        )
        self.assertTrue(unsupported.blocks_lowering)

    def test_elementwise_instruction_is_target_backed_and_dtype_checked(self) -> None:
        admitted = verify(Schedule.load(SWIGLU), TARGET)
        self.assertNotIn("TARGET_INSTRUCTION_UNSUPPORTED", _codes(admitted))
        self.assertNotIn("ELEMENTWISE_INSTRUCTION_DTYPE_DIFFERS", _codes(admitted))

        unsupported = verify(
            _mutated(
                SWIGLU,
                lambda d: _op(d, "tanh_gate")["parameters"]["instruction"].update(
                    contract="tanh.approx.f32"
                ),
            ),
            TARGET,
        )
        finding = next(
            f for f in unsupported if f.code == "TARGET_INSTRUCTION_UNSUPPORTED"
        )
        self.assertTrue(finding.blocks_lowering)
        self.assertIn("operations[3].parameters.instruction.contract", finding.path)

        wrong_kind = verify(
            _mutated(
                SWIGLU,
                lambda d: _op(d, "tanh_gate")["parameters"]["instruction"].update(
                    contract="triton.dot.bf16_fp32"
                ),
            ),
            TARGET,
        )
        self.assertIn("ELEMENTWISE_INSTRUCTION_KIND_DIFFERS", _codes(wrong_kind))

        wrong_dtype = verify(
            _mutated(
                SWIGLU,
                lambda d: next(
                    b for b in d["buffers"] if b["name"] == "half_gate"
                ).update(dtype="fp16"),
            ),
            TARGET,
        )
        self.assertIn("ELEMENTWISE_INSTRUCTION_DTYPE_DIFFERS", _codes(wrong_dtype))

    def test_mma_tile_must_match_its_accumulator(self) -> None:
        matching = verify(
            _mutated(
                ASSIGNMENT_FULL,
                lambda d: _op(d, "dot_mma")["parameters"].update(
                    tile_shape=[128, 256, 64]
                ),
            ),
            TARGET,
        )
        self.assertEqual(_blocking(matching), ())

        mismatched = verify(
            _mutated(
                ASSIGNMENT_FULL,
                lambda d: _op(d, "dot_mma")["parameters"].update(
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
    def _mutate(self, mutate) -> tuple[Finding, ...]:
        document = json.loads(ASSIGNMENT_FULL.read_text(encoding="utf-8"))
        mutate(document)
        return verify(Schedule.from_dict(document), TARGET)

    def test_the_retained_nest_is_accepted(self) -> None:
        self.assertEqual(_actionable(verify(Schedule.load(ASSIGNMENT_FULL), TARGET)), ())

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


class RoleRegisterSplitTest(unittest.TestCase):
    """A per-role register budget divides the CTA's allocation; it does not add to it.

    All four surveyed libraries split registers by role -- one deallocates a transform
    role so its accumulator role can take more, one holds an entirely empty role at the
    minimum for the same reason. None of it could be written down here, so the emitter had
    no budget to honour and the analysis had no split to reason about.
    """

    def _split(self, **changes) -> dict:
        document = json.loads(
            (ROOT / "corpus/schedules/rmsnorm-b8-smoke.json").read_text(encoding="utf-8")
        )
        document["roles"] = [
            {"name": "load", "warps": [0, 1, 2, 3], "registers_per_thread": 64},
            {"name": "compute", "warps": [4, 5, 6, 7], "registers_per_thread": 192},
        ]
        for operation in document["operations"]:
            operation["role"] = "load" if operation["kind"] == "load" else "compute"
        document["residency"] = {"registers_per_thread": 128}
        for key, value in changes.items():
            if key == "warps":
                for role, warps in zip(document["roles"], value):
                    role["warps"] = warps
            elif key == "budgets":
                for role, budget in zip(document["roles"], value):
                    if budget is None:
                        role.pop("registers_per_thread", None)
                    else:
                        role["registers_per_thread"] = budget
            elif key == "total":
                if value is None:
                    # An empty residency block is a parse error in its own right, so the
                    # absent case is the block being absent.
                    document.pop("residency", None)
                else:
                    document["residency"]["registers_per_thread"] = value
        return document

    def _codes(self, document: dict) -> set[str]:
        return {
            f.code
            for f in _blocking(verify(Schedule.from_dict(document), TARGET))
            if f.code.startswith("ROLE_REGISTERS")
        }

    def test_a_conserved_split_is_accepted(self) -> None:
        # 64 and 192 across four warps each is exactly 128 per thread over eight warps.
        self.assertEqual(self._codes(self._split()), set())

    def test_a_split_that_does_not_add_up_is_refused(self) -> None:
        for label, budgets in (("under", (64, 128)), ("over", (128, 192))):
            with self.subTest(direction=label):
                self.assertIn(
                    "ROLE_REGISTERS_NOT_CONSERVED", self._codes(self._split(budgets=budgets))
                )

    def test_a_budget_must_span_whole_warpgroups(self) -> None:
        """setmaxnreg is warpgroup-wide, so two roles sharing one conflict.

        This rule was tested against hardware and the result is why it stays blocking.
        A Schedule whose single-warp roles each issue a different budget from the same
        warpgroup was emitted past the verifier, compiled on a B200, and produced correct
        results. That is not evidence the split is legal: `setmaxnreg.sync.aligned`
        executed by part of a warpgroup is undefined, and undefined behaviour routinely
        looks correct. What the run does establish is that the toolchain accepts the
        violation silently, so nothing downstream would catch it -- which is the argument
        for gating it here rather than against it. Every surveyed library issues the
        instruction warpgroup-aligned without exception.
        """

        codes = self._codes(self._split(warps=([0, 1], [2, 3, 4, 5, 6, 7])))
        self.assertIn("ROLE_REGISTERS_NOT_WARPGROUP_ALIGNED", codes)

    def test_a_partial_split_is_refused(self) -> None:
        self.assertIn("ROLE_REGISTERS_PARTIAL", self._codes(self._split(budgets=(64, None))))

    def test_a_split_without_the_allocation_it_divides_is_refused(self) -> None:
        self.assertIn("ROLE_REGISTERS_WITHOUT_TOTAL", self._codes(self._split(total=None)))

    def test_an_illegal_register_count_is_a_parse_error(self) -> None:
        # Not a tuning choice the backend can decline: setmaxnreg takes a multiple of
        # eight in [24, 256] and anything else is an illegal instruction.
        for bad in (100, 20, 264):
            with self.subTest(registers=bad):
                with self.assertRaisesRegex(ScheduleParseError, "multiple of 8"):
                    Schedule.from_dict(self._split(budgets=(bad, 192)))


class ContractionAccumulatorDiagnosticTest(unittest.TestCase):
    """The refusal a GEMM author would hit, and what it has to tell them.

    The Triton backend cannot accumulate a contraction across a loop -- `_emit_mma`
    assigns a `tl.dot` -- and nothing states that. What prevents the wrong answer is a
    rule about register buffers escaping loops, which is a different rule that happens to
    cover it. The CuTe-DSL backend does accumulate across its `k_loop`, because its
    accumulator lives in tensor memory and the rule exempts spaces that outlive a loop.

    So the same Schedule shape is correct on one backend and refused on the other, and the
    refusal has to say which constraint it is. The paper asks for localized diagnostics;
    "only a reduction result is carried out of a loop" is true and sends a GEMM author
    looking for a rule they broke.
    """

    def test_an_escaping_contraction_names_the_constraint_it_hits(self) -> None:
        document = json.loads(
            (ROOT / "corpus/schedules/flash-kmeans-b32-smoke-v2.json").read_text(
                encoding="utf-8"
            )
        )
        # The GEMM shape: the contraction stays in the loop and its consumer moves out.
        document["tile_loops"][0]["body"] = [
            "load_centroids",
            "load_norm",
            "distance_mma",
        ]
        target = Target.load(ROOT / "compiler/targets/sm_100a.json")
        messages = {
            finding.message.split("'")[1]: finding.message
            for finding in verify(Schedule.from_dict(document), target)
            if finding.code == "BUFFER_ESCAPES_LOOP"
        }

        self.assertIn("accumulator in a space that outlives it", messages["cross"])
        # A staged load that escapes is a different mistake and keeps the general reason.
        self.assertIn("only a reduction result", messages["norm_tile"])


class UnmodelledResourceReportTest(unittest.TestCase):
    """The residency report says what it did not examine, not only what it bounded.

    A resource a Schedule declares nothing for produces no bound, and leaving it out of
    the report reads as "does not constrain". Measurement made the difference concrete:
    `gemm-bias-b1-smoke` declares no shared memory, Triton allocates it for the `tl.dot`
    operands anyway, and on a B200 it bounds residency exactly as tightly as the registers
    the analysis does model (`docs/ANALYSIS_CALIBRATION.md`).

    The paper says a static analysis is a gate only within its modelled domain. A report
    that does not state its domain leaves the reader to assume it.
    """

    def _residency_message(self, name: str) -> str:
        schedule = Schedule.from_dict(
            json.loads(
                (ROOT / "corpus/schedules" / name).read_text(encoding="utf-8")
            )
        )
        target = Target.load(ROOT / "compiler/targets/sm_100a.json")
        return next(
            finding.message
            for finding in verify(schedule, target)
            if finding.code == "RESIDENCY_BOUND"
        )

    def test_a_space_with_no_allocation_is_named_as_unbounded_not_omitted(self) -> None:
        gemm = self._residency_message("gemm-bias-b1-smoke.json")
        self.assertIn("declares no shared_memory or tensor_memory", gemm)
        self.assertIn("a backend may still allocate some", gemm)

    def test_a_schedule_that_declares_them_gets_no_such_caveat(self) -> None:
        # It bounds both, so there is nothing unexamined to warn about.
        assignment = self._residency_message("flash-kmeans-assignment-full.json")
        self.assertNotIn("declares no", assignment)
        self.assertIn("tensor_memory", assignment)
