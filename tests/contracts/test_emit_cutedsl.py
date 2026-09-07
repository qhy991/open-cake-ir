"""Contract tests for CuTe-DSL emission.

The hand-written artifact carries thirteen module-level constants. Each one is a
decision, and each is now computed from the Schedule rather than typed by hand. These
tests compare the emitter's derivation against those constants directly, so the two
cannot drift apart silently.

Scope: this holds the deterministic lowering structure and the runnable operation bodies
to the Schedule facts they consume. On-device correctness remains separate evidence.
"""

from __future__ import annotations

import ast
import json
import re
import unittest
from pathlib import Path

from open_cake_ir.compiler.backends.common import EmitError
from open_cake_ir.compiler.backends.cutedsl import emit
from open_cake_ir.compiler.ir import BarrierMechanism, PipelineKind, Schedule
from open_cake_ir.compiler.target import Target

ROOT = Path(__file__).resolve().parents[2]
TARGET = Target.load(ROOT / "compiler" / "targets" / "sm_100a.json")
SCHEDULE = ROOT / "corpus" / "schedules" / "flash-kmeans-assignment-full.json"

# What the deleted hand-written artifact carried as module constants. Emission derives
# each one from the Schedule, which is why that file is gone.
ARTIFACT_CONSTANTS = {
    "IO_DTYPE": "cutlass.BFloat16",
    "ACC_DTYPE": "cutlass.Float32",
    "MMA_INSTRUCTION_SHAPE": (128, 256, 16),
    "MMA_TILE": (128, 256, 64),
    "PIPELINE_STAGES": 2,
    "THREADS_PER_CTA": 224,
    "EPILOGUE_THREADS": 128,
    "EPILOGUE_WARPS": (0, 1, 2, 3),
    "MMA_WARP": 4,
    "TMA_WARP": 5,
    "REDUCE_WARP": 6,
    "NUM_K_LOOP_TRIPS": 2,           # NUM_K_TILES
    "NUM_CENTROID_LOOP_TRIPS": 4,    # NUM_CENTROID_TILES
    "TMEM_COLUMNS": 256,             # the artifact allocated 512
}

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


class DerivationTest(unittest.TestCase):
    """Each constant the artifact hardcoded, now computed from the Schedule."""

    def test_every_artifact_constant_is_derived(self) -> None:
        constants = emit(Schedule.load(SCHEDULE), TARGET).constants
        for name, value in ARTIFACT_CONSTANTS.items():
            with self.subTest(constant=name):
                self.assertEqual(constants[name], value)

    def test_tensor_memory_is_sized_by_the_accumulator(self) -> None:
        """The artifact called tmem.allocate(512) for a 256-column accumulator."""

        schedule = Schedule.load(SCHEDULE)
        allocation = next(
            a for a in schedule.allocations if a.tensor_columns is not None
        )
        self.assertEqual(allocation.tensor_columns, 256)
        self.assertEqual(allocation.implied_tensor_columns, 256)


class StructureTest(unittest.TestCase):
    def setUp(self) -> None:
        self.schedule = Schedule.load(SCHEDULE)
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
                    self.assertEqual(role.warps, tuple(range(low, high + 1)))
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
        document = json.loads(SCHEDULE.read_text(encoding="utf-8"))
        mutate(document)
        return Schedule.from_dict(document)

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


class BackendCoverageTest(unittest.TestCase):
    """What each backend can lower is a derived fact, and the Compiler reads it.

    The dispatch used to be an `elif` chain, so the set of kinds a backend handled existed
    only in the shape of that chain. Nothing could ask it, which is why a Schedule using a
    kind with no body passed every gate and then raised while emitting.
    """

    def test_every_table_entry_names_a_method_that_exists(self) -> None:
        from open_cake_ir.compiler.backends import cutedsl, triton

        tables = (
            (cutedsl, "_Emitter", cutedsl.BODY_EMITTERS),
            (triton, "_TritonEmitter", triton.OUTSIDE_LOOP_EMITTERS),
            (triton, "_TritonEmitter", triton.INSIDE_LOOP_EMITTERS),
        )
        for module, class_name, table in tables:
            emitter_class = getattr(module, class_name)
            for kind, method in table.items():
                with self.subTest(backend=module.__name__, kind=kind.value):
                    # A string key is only safe while something checks it resolves.
                    self.assertTrue(callable(getattr(emitter_class, method, None)))

    def test_a_backend_names_every_dtype_it_admits_in_both_places(self) -> None:
        """A dtype in one of a backend's tables and not the other lowers until it reaches
        a host tensor, and then raises a KeyError instead of being refused.

        `int64` was exactly that until the Compiler learned to check: it had a byte width
        in the IR, no entry in either backend, and produced a raw KeyError from emission.
        It is gone; this holds the shape that let it hide.
        """

        from open_cake_ir.compiler.backends import cutedsl, triton

        for module, tables in (
            (triton, (triton._TL_DTYPE, triton._TORCH_DTYPE)),
            (cutedsl, (cutedsl._CUTLASS_DTYPE, cutedsl._TORCH_DTYPE)),
        ):
            with self.subTest(backend=module.__name__):
                first, second = (frozenset(table) for table in tables)
                self.assertEqual(first, second)
                self.assertEqual(module.SUPPORTED_DTYPES, first)

    def test_every_dtype_the_ir_admits_has_a_backend(self) -> None:
        from open_cake_ir.compiler.backends import cutedsl, triton
        from open_cake_ir.compiler.ir import DType

        covered = cutedsl.SUPPORTED_DTYPES | triton.SUPPORTED_DTYPES
        # Same rule the operation kinds live under: a word the vocabulary offers and no
        # backend can keep is a promise, not a capability.
        self.assertEqual(set(DType) - covered, set())

    # Tables that are partial on purpose, each with a declared-coverage set saying so and
    # a Compiler finding that refuses a Schedule outside it. Everything else keyed by an
    # IR enum has to be total: a member with no entry is a KeyError from emission rather
    # than a diagnosis, which is how int64 hid.
    DECLARED_PARTIAL = {
        ("triton", "INSIDE_LOOP_EMITTERS"),
        ("triton", "OUTSIDE_LOOP_EMITTERS"),
        ("triton", "_TL_DTYPE"),
        ("triton", "_TORCH_DTYPE"),
        ("cutedsl", "BODY_EMITTERS"),
    }

    def test_every_other_enum_keyed_table_is_total(self) -> None:
        from enum import Enum

        from open_cake_ir.compiler.backends import cutedsl, triton

        for module in (triton, cutedsl):
            short = module.__name__.rsplit(".", 1)[-1]
            for name, table in sorted(vars(module).items()):
                if not isinstance(table, dict) or not table:
                    continue
                keys = list(table)
                if not all(isinstance(key, Enum) for key in keys):
                    continue
                if (short, name) in self.DECLARED_PARTIAL:
                    continue
                with self.subTest(table=f"{short}.{name}"):
                    owner = type(keys[0])
                    self.assertEqual(
                        [member.value for member in owner if member not in table], []
                    )

    def test_the_declared_coverage_is_the_dispatch(self) -> None:
        from open_cake_ir.compiler.backends import cutedsl, triton

        self.assertEqual(
            cutedsl.SUPPORTED_OPERATION_KINDS, frozenset(cutedsl.BODY_EMITTERS)
        )
        self.assertEqual(
            triton.SUPPORTED_OPERATION_KINDS,
            frozenset(triton.OUTSIDE_LOOP_EMITTERS)
            | frozenset(triton.INSIDE_LOOP_EMITTERS),
        )
        # The two backends genuinely differ, which is the reason a profile has to be
        # asked rather than the Target: both of these lower for sm_100a.
        self.assertNotEqual(
            cutedsl.SUPPORTED_OPERATION_KINDS, triton.SUPPORTED_OPERATION_KINDS
        )

    def test_backend_constructor_requirements_have_one_preflight_owner(self) -> None:
        from open_cake_ir.compiler.backends import cutedsl, triton

        triton_document = json.loads(
            (ROOT / "corpus/schedules/flash-kmeans-b32-smoke-v2.json").read_text()
        )
        triton_document["roles"].append({"name": "unused", "warps": [4]})
        triton_failures = triton.preflight(
            Schedule.from_dict(triton_document), TARGET
        )
        self.assertIn("TRITON_ROLE_COUNT", {item.code for item in triton_failures})

        cute_document = json.loads(SCHEDULE.read_text())
        next(
            operation
            for operation in cute_document["operations"]
            if operation["kind"] == "epilogue"
        )["parameters"]["formula"] = "bias_add_bf16_round"
        cute_failures = cutedsl.preflight(
            Schedule.from_dict(cute_document), TARGET
        )
        self.assertEqual(
            [item.code for item in cute_failures],
            ["CUTE_EPILOGUE_FORMULA_UNSUPPORTED"],
        )


class AllocationOwnershipTest(unittest.TestCase):
    """Who takes out the tensor-memory allocation is the Schedule's decision.

    It used to be the emitter's: the role of the first operation, in declaration order,
    whose reads touched tensor memory. The verifier admits a Schedule where two roles read
    tensor memory, so that inference was ambiguous by more than accident -- reordering two
    operations would have moved a `tcgen05.alloc` to a different warp, and the matching
    `relinquish_alloc_permit` with it.
    """

    def _emit_with_owner(self, role: str) -> str:
        document = json.loads(SCHEDULE.read_text(encoding="utf-8"))
        for allocation in document["allocations"]:
            if allocation["space"] == "tensor":
                allocation["allocating_role"] = role
        return emit(Schedule.from_dict(document), TARGET).source

    def _allocating_branch(self, source: str) -> str:
        lines = source.splitlines()
        index = next(i for i, line in enumerate(lines) if "tmem.allocate(" in line)
        return next(line.strip() for line in reversed(lines[:index]) if "warp_idx" in line)

    def test_the_declared_role_is_where_the_allocation_lands(self) -> None:
        # The corpus Schedule declares the epilogue, which is what the inference used to
        # pick, so this half also fixes the behaviour the hardware evidence was taken on.
        self.assertIn("warp_idx <= 3", self._allocating_branch(self._emit_with_owner("epilogue")))
        # And naming a different role moves the instruction, which is the whole point:
        # a decision the Schedule can state is a decision an author can change.
        self.assertIn("warp_idx == MMA_WARP", self._allocating_branch(self._emit_with_owner("mma")))

    def test_an_undeclared_owner_is_refused_rather_than_guessed(self) -> None:
        document = json.loads(SCHEDULE.read_text(encoding="utf-8"))
        for allocation in document["allocations"]:
            if allocation["space"] == "tensor":
                allocation.pop("allocating_role")
        # Structurally incomplete, so it does not survive construction -- there is no
        # ill-typed Schedule for a later stage to reason about.
        with self.assertRaisesRegex(ValueError, "allocating_role"):
            Schedule.from_dict(document)


class BodyEmissionTest(unittest.TestCase):
    """Each of these was a bug the B200 found, now derived rather than written."""

    def setUp(self) -> None:
        self.schedule = Schedule.load(SCHEDULE)
        self.source = emit(self.schedule, TARGET).source

    def test_loop_nesting_and_scope_come_from_the_schedule(self) -> None:
        from open_cake_ir.compiler.backends.cutedsl import _Emitter

        emitter = _Emitter(self.schedule, TARGET)
        scopes = {
            role.name: {
                scope: [op.op_id for op in emitter._ops_in_scope(role, scope)]
                for scope in [None] + [l.name for l in emitter._loops_for_role(role)]
                if emitter._ops_in_scope(role, scope)
            }
            for role in self.schedule.roles
        }
        self.assertEqual(
            scopes,
            {
                "tma": {"k_loop": ["load_tokens", "load_centroids"]},
                "mma": {"k_loop": ["dot_mma"]},
                "epilogue": {"centroid_loop": ["distance_epilogue"]},
                "reduce": {None: ["argmin", "store_assignment"]},
            },
        )

    def test_pipeline_class_is_one_typed_derived_fact(self) -> None:
        from open_cake_ir.compiler.backends.cutedsl import _Emitter

        emitter = _Emitter(self.schedule, TARGET)
        kinds = {
            operation.op_id: operation.produced_pipeline_kind
            for operation in self.schedule.operations
            if operation.produced_pipeline_kind is not None
        }
        self.assertEqual(
            kinds,
            {
                "load_tokens": PipelineKind.TMA_TO_UMMA,
                "load_centroids": PipelineKind.TMA_TO_UMMA,
                "dot_mma": PipelineKind.UMMA_TO_THREAD,
            },
        )
        classes = {b.name: emitter._pipeline_class(b) for b in emitter._mbarriers()}
        self.assertEqual(
            classes,
            {
                "tiles_ready": "PipelineTmaUmma",
                "accumulator_ready": "PipelineUmmaAsync",
            },
        )

    def test_direct_emission_rejects_ambiguous_pipeline_producers(self) -> None:
        document = json.loads(SCHEDULE.read_text(encoding="utf-8"))
        barrier = next(b for b in document["barriers"] if b["name"] == "tiles_ready")
        barrier["producers"].append("mma")
        mma = next(o for o in document["operations"] if o["id"] == "dot_mma")
        mma["signals"].append("tiles_ready")

        with self.assertRaisesRegex(EmitError, "needs one lowerable pipeline kind"):
            emit(Schedule.from_dict(document), TARGET)

    def test_a_cta_barrier_is_emitted_outside_the_warp_dispatch(self) -> None:
        """Emitting it inside one role's branch deadlocks the other six warps.

        The B200 run that found this hit the job's wall clock with no output.
        """

        rendezvous = self.source.index("cute.arch.sync_threads()")
        line_start = self.source.rindex("\n", 0, rendezvous) + 1
        self.assertEqual(
            self.source[line_start:rendezvous], "    ", "the rendezvous must be at kernel scope"
        )
        self.assertIn("distance_ready", self.source[:rendezvous].rsplit("\n", 2)[-2])

    def test_only_a_thread_producer_commits(self) -> None:
        """A TMA transaction completes its own barrier; a thread producer must commit."""

        self.assertIn("accumulator_ready_empty.commit()", self.source)
        self.assertNotIn("tiles_ready_empty.commit()", self.source)

    def test_an_inner_operation_may_address_an_outer_axis(self) -> None:
        """The B operand is tiled in N by the outer loop and read by an inner load."""

        self.assertIn(
            "tma_global_load_centroids[(None, centroid_tile, k_tile)]", self.source
        )
        self.assertIn("tma_global_load_tokens[(None, k_tile)]", self.source)
        self.assertNotIn("for _ in cutlass.range", self.source)

    def test_a_store_fed_by_a_reduction_lives_in_its_row_loop(self) -> None:
        body = self.source[self.source.index("# CAKE_OP:argmin"):]
        store = body.index("assignments[row] = best_index")
        indent = body[body.rindex("\n", 0, store) + 1 : store]
        self.assertEqual(len(indent), 12, "the store must be inside the row loop")

    def test_the_allocating_role_releases_tensor_memory(self) -> None:
        self.assertIn("tmem.allocate(TMEM_COLUMNS)", self.source)
        self.assertIn("tmem.relinquish_alloc_permit()", self.source)

    def test_every_operation_has_a_body(self) -> None:
        self.assertNotIn("pass  # body for", self.source)
        for operation in self.schedule.operations:
            with self.subTest(operation=operation.op_id):
                self.assertIn(f"# CAKE_OP:{operation.op_id}", self.source)


class EmittedKernelObservationTest(unittest.TestCase):
    """The retained B200 observation for the generated kernel.

    Correctness only. No timing was taken and no comparison against the hand-written
    artifact is claimed.

    The record names the artifact that ran. Compiler v24 changed Schedule and generated
    source bytes when it replaced profiles with lowering routes, so this test preserves
    that historical/current split. The answer to the gap is a successor observation,
    never an edit or relabelling of this frozen record.

    This one is stronger than the records it supersedes. They were taken with a
    centroid-norm vector constructed to equal the squared row sums of the centroids, which
    is how the kernel is called in production and is one input family. The instrument now
    derives its inputs from the Schedule's declarations, so the vector is arbitrary and
    the check is of the arithmetic the kernel actually promises.
    """

    RECORD = ROOT / "inventory" / "FLASH_KMEANS_OBSERVATION_20260824.json"

    def test_the_observation_remains_historical_after_route_migration(self) -> None:
        from open_cake_ir.compiler import Compiler

        record = json.loads(self.RECORD.read_text(encoding="utf-8"))
        # Compare the current source candidate with the frozen observation. Released
        # lock/source validation has its own Compiler and release contract tests.
        compiler = Compiler.load(ROOT, ROOT / "compiler" / "revision.json")
        lowering = compiler.lower(compiler.assess_file(SCHEDULE))
        self.assertEqual(lowering.generated, record["lowering"]["generated"])
        self.assertNotEqual(
            lowering.schedule_sha256, record["schedule"]["canonical_sha256"]
        )
        self.assertNotEqual(lowering.source_sha256, record["lowering"]["source_sha256"])
        self.assertNotEqual(
            lowering.compiler_revision_id,
            record["compiler_revision"]["revision_id"],
        )
        self.assertEqual(record["result"]["mismatch_count"], 0)
        self.assertEqual(record["result"]["max_chosen_distance_excess"], 0.0)
        self.assertTrue(record["result"]["passed"])
        self.assertFalse(record["scientific_claim_authorized"])
        self.assertFalse(record["performance_measured"])


class SubRangeIsRefusedNotIgnoredTest(unittest.TestCase):
    """A sub-range is `AccessIndex` vocabulary, so this backend must answer for it.

    This emitter builds addresses without reading `offset` or `extent`. Before it said
    so, a Schedule naming half an axis lowered to source covering all of it -- the same
    bytes as the whole-axis Schedule, silently answering a different question. Honouring
    the range or refusing it are both fine; ignoring it is not.

    The two documents differ in exactly one component, so the code this asserts can only
    come from the sub-range itself and not from some unrelated CuTe precondition.
    """

    def _mapped(self) -> dict:
        document = json.loads(SCHEDULE.read_text(encoding="utf-8"))
        whole = lambda dim: {"source": "dimension", "dimension": dim}
        loop = lambda name: {"source": "loop_tile", "name": name}
        edge = lambda op, buf, idx: {
            "operation": op, "buffer": buf, "boundary": "mask_tiled_axes", "indices": idx
        }
        document["access_maps"] = [
            edge("load_tokens", "tokens", [whole(0), loop("k_tile")]),
            edge("load_centroids", "centroids", [loop("centroid_tile"), loop("k_tile")]),
            edge("distance_epilogue", "centroid_sq", [loop("centroid_tile")]),
            edge("distance_epilogue", "distance_scratch", [whole(0), loop("centroid_tile")]),
            edge("argmin", "distance_scratch", [whole(0), whole(1)]),
            edge("store_assignment", "assignments", [whole(0)]),
        ]
        return document

    def test_the_backend_blocks_the_range_it_cannot_address(self) -> None:
        from open_cake_ir.compiler import Compiler

        compiler = Compiler.load(ROOT, ROOT / "compiler" / "revision.json")
        whole = self._mapped()
        sub = json.loads(json.dumps(whole))
        sub["access_maps"][4]["indices"][0] = {
            "source": "dimension", "dimension": 0, "extent": 64
        }

        whole_assessment = compiler.assess(whole)
        sub_assessment = compiler.assess(sub)

        self.assertTrue(whole_assessment.lowering_eligible)
        # The Schedule is well formed; only this backend cannot carry it.
        self.assertTrue(sub_assessment.accepted)
        self.assertFalse(sub_assessment.lowering_eligible)

        blocking = lambda a: {f.code for f in a.findings if f.blocks_lowering}
        self.assertEqual(
            blocking(sub_assessment) - blocking(whole_assessment),
            {"CUTE_ACCESS_SUBRANGE_UNSUPPORTED"},
        )


class RangeOptionsAreRefusedNotIgnoredTest(unittest.TestCase):
    """A declared loop control must survive lowering or produce a localized refusal.

    The accepted pipeline previously emitted identical CuTe source with unroll=1 or
    unroll=4, flatten=False or True, disable_licm=False or True, and inner stages=2 or
    3. These probes change only the option under test, so an unrelated blocker cannot
    lend the backend a false claim that it checks the loop control.
    """

    @classmethod
    def setUpClass(cls) -> None:
        from open_cake_ir.compiler import Compiler

        cls.compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")

    def _assert_option_refused(self, document: dict, index: int, field: str) -> None:
        from open_cake_ir.compiler import CompilerError
        from open_cake_ir.compiler.backends.cutedsl import preflight

        schedule = Schedule.from_dict(document)
        failures = preflight(schedule, TARGET)
        expected = [("CUTE_RANGE_OPTION_UNSUPPORTED", f"tile_loops[{index}].range_options.{field}")]
        self.assertEqual([(item.code, item.path) for item in failures], expected)

        assessment = self.compiler.assess(document)
        self.assertTrue(assessment.accepted)
        self.assertFalse(assessment.lowering_eligible)
        blocking = [item for item in assessment.findings if item.blocks_lowering]
        self.assertEqual([(item.code, item.path) for item in blocking], expected)
        self.assertFalse(blocking[0].blocks_acceptance)
        with self.assertRaises(CompilerError):
            self.compiler.lower(assessment)
        with self.assertRaisesRegex(EmitError, field):
            emit(schedule, TARGET)

    def test_unimplemented_controls_are_refused_on_each_loop(self) -> None:
        original = json.loads(SCHEDULE.read_text(encoding="utf-8"))
        self.assertTrue(self.compiler.assess(original).lowering_eligible)
        for index in range(len(original["tile_loops"])):
            for field, value in (
                ("loop_unroll_factor", 4), ("flatten", True), ("disable_licm", True),
            ):
                with self.subTest(loop=index, field=field):
                    document = json.loads(SCHEDULE.read_text(encoding="utf-8"))
                    document["tile_loops"][index]["range_options"][field] = value
                    self._assert_option_refused(document, index, field)

    def test_nondefault_stages_must_match_the_pipeline_acquisition_scope(self) -> None:
        # The Pipeline's barrier is acquired in the inner loop, whose requested count
        # must match the emitted storage. The outer loop does not acquire that barrier.
        for index, stages in ((0, 2), (1, 3)):
            with self.subTest(loop=index, stages=stages):
                document = json.loads(SCHEDULE.read_text(encoding="utf-8"))
                document["tile_loops"][index]["range_options"]["num_stages"] = stages
                self._assert_option_refused(document, index, "num_stages")

    def test_an_operation_pipeline_tag_does_not_move_stage_consumption(self) -> None:
        document = json.loads(SCHEDULE.read_text(encoding="utf-8"))
        baseline = emit(Schedule.from_dict(document), TARGET).source
        epilogue = next(op for op in document["operations"] if op["id"] == "distance_epilogue")
        epilogue["pipeline"] = "main"
        self.assertTrue(self.compiler.assess(document).lowering_eligible)
        self.assertEqual(emit(Schedule.from_dict(document), TARGET).source, baseline)

        # This tag formerly admitted two outer-loop stages, although the emitted
        # Pipeline barrier was still acquired only in k_loop and source was unchanged.
        document["tile_loops"][0]["range_options"]["num_stages"] = 2
        self._assert_option_refused(document, 0, "num_stages")

    def test_default_loop_stages_defer_to_the_explicit_pipeline(self) -> None:
        document = json.loads(SCHEDULE.read_text(encoding="utf-8"))
        baseline = emit(Schedule.from_dict(document), TARGET).source
        document["tile_loops"][1]["range_options"]["num_stages"] = 1
        self.assertTrue(self.compiler.assess(document).lowering_eligible)
        self.assertEqual(emit(Schedule.from_dict(document), TARGET).source, baseline)

    def test_existing_pipeline_preserves_its_structural_controls(self) -> None:
        from open_cake_ir.compiler.backends.cutedsl import preflight

        for path in (SCHEDULE, ROOT / "examples/python/kmeans_pipeline.py"):
            with self.subTest(path=path):
                assessment = self.compiler.assess_file(path)
                schedule = Schedule.from_dict(json.loads(assessment.schedule_bytes))
                self.assertEqual(preflight(schedule, TARGET), ())
                self.assertTrue(assessment.lowering_eligible)
                source = self.compiler.lower(assessment).source
                self.assertIn("PIPELINE_STAGES = 2", source)
                self.assertIn("num_stages=PIPELINE_STAGES", source)
                self.assertIn("warp_idx == TMA_WARP", source)
                self.assertIn("warp_idx == MMA_WARP", source)
                self.assertEqual(source.count("tmem.allocate(TMEM_COLUMNS)"), 1)

    def test_python_unroll_request_is_localized_to_its_symbolic_loop(self) -> None:
        from open_cake_ir.compiler.frontend import parse

        source = (ROOT / "examples/python/kmeans_pipeline.py").read_text(encoding="utf-8")
        source = source.replace("tile=64, num_stages=2,", "tile=64, num_stages=2, loop_unroll_factor=4,")
        authored = parse(source, filename="unrolled-kmeans.py")
        self._assert_option_refused(authored.document, 1, "loop_unroll_factor")
        location = authored.location_for("tile_loops[1].range_options.loop_unroll_factor")
        self.assertIsNotNone(location)
        self.assertIn("lm.range", source.splitlines()[location.line - 1])
