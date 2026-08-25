from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REVISION_PATH = ROOT / "compiler/revision.lock.json"
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.compiler import Compiler, CompilerError  # noqa: E402
from open_cake_ir.compiler.release import build_release  # noqa: E402
from open_cake_ir.evidence import EvidenceStore  # noqa: E402
from open_cake_ir.lab import KernelSeed, lower_specialists  # noqa: E402



def _decisive(assessment):
    """Findings that decide acceptance or lowering.

    An Assessment also carries reports -- bottleneck attribution that reaches the agent
    without affecting either decision. Those have their own contract tests; a test about
    a decision should not have to enumerate them.
    """

    return [f for f in assessment.findings if f.blocks_acceptance or f.blocks_lowering]


class CompilerContractTests(unittest.TestCase):
    def test_release_archive_index_externally_anchors_the_terminal_seal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root / "compiler.py"
            source.write_bytes(b"released compiler source\n")
            revision = {
                "schema_version": 1,
                "revision_id": "compiler-anchor-contract-v1",
                "state": "released",
                "sources": [
                    {
                        "path": source.name,
                        "sha256": sha256(source.read_bytes()).hexdigest(),
                        "size_bytes": source.stat().st_size,
                    }
                ],
            }
            revision_path = root / "revision.lock.json"
            revision_path.write_text(json.dumps(revision), encoding="utf-8")
            evidence_root = root / "evidence"
            index_path = root / "compiler-release-index.json"

            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools/archive_compiler_release.py"),
                    "--project-root",
                    str(root),
                    "--revision",
                    str(revision_path),
                    "--evidence-root",
                    str(evidence_root),
                    "--index-output",
                    str(index_path),
                    "--run-id",
                    "compiler-anchor-contract",
                ],
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr.decode())
            audit = EvidenceStore.open(evidence_root).audit_run(
                "compiler-anchor-contract"
            )
            index = json.loads(index_path.read_text(encoding="utf-8"))
            self.assertEqual(index_path.stat().st_mode & 0o444, 0o444)
            self.assertTrue(audit.integrity)
            self.assertEqual(index["terminal_seal_sha256"], audit.terminal_seal_sha256)
            self.assertEqual(index["authority_sha256"], audit.authority_sha256)
            self.assertEqual(index["source_count"], 1)
            self.assertEqual(index["object_count"], 2)

    def test_release_archive_will_not_replace_an_existing_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root / "compiler.py"
            source.write_bytes(b"released compiler source\n")
            revision_path = root / "revision.lock.json"
            revision_path.write_text(
                json.dumps(
                    {
                        "revision_id": "compiler-create-only-contract-v1",
                        "state": "released",
                        "sources": [
                            {
                                "path": source.name,
                                "sha256": sha256(source.read_bytes()).hexdigest(),
                                "size_bytes": source.stat().st_size,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            evidence_root = root / "evidence"
            index_path = root / "compiler-release-index.json"
            index_path.write_bytes(b"preexisting anchor\n")

            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools/archive_compiler_release.py"),
                    "--project-root",
                    str(root),
                    "--revision",
                    str(revision_path),
                    "--evidence-root",
                    str(evidence_root),
                    "--index-output",
                    str(index_path),
                    "--run-id",
                    "compiler-create-only-contract",
                ],
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertEqual(index_path.read_bytes(), b"preexisting anchor\n")
            self.assertFalse(evidence_root.exists())

    def test_released_compiler_source_closure_replays_from_evidence_v2(self) -> None:
        audit = EvidenceStore.open(ROOT / "evidence/compiler-release-v2").audit_run(
            "compiler-release-v2"
        )

        self.assertTrue(audit.integrity)
        self.assertEqual(
            audit.authority_sha256,
            "dafc18995831c29f198abd25f7386a815f38bb1d3ad040f806e8b298be635843",
        )
        self.assertEqual(audit.endpoint["source_count"], 14)

    def test_compiler_import_is_independent_of_lab_evaluation_and_evidence(self) -> None:
        script = (
            "import json,sys; import open_cake_ir.compiler; "
            "print(json.dumps(sorted(name for name in sys.modules if "
            "name.startswith(('open_cake_ir.lab','open_cake_ir.evaluation',"
            "'open_cake_ir.evidence')))))"
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        )

        self.assertEqual(json.loads(completed.stdout), [])

    def test_target_mismatch_and_missing_calibration_are_explicit(self) -> None:
        compiler = Compiler.load(ROOT, REVISION_PATH)
        schedule = json.loads(
            (ROOT / "corpus/schedules/flash-kmeans-b32-smoke-v2.json").read_text()
        )
        schedule["target"] = "sm_90"

        assessment = compiler.assess(schedule)

        self.assertFalse(assessment.accepted)
        self.assertIn("TARGET_UNSUPPORTED", [item.code for item in assessment.findings])
        self.assertFalse(assessment.calibration_available)

    def test_lower_rejects_a_forged_assessment_projection(self) -> None:
        compiler = Compiler.load(ROOT, REVISION_PATH)
        assessment = compiler.assess_file(
            ROOT / "corpus/schedules/flash-kmeans-b32-smoke-v2.json"
        )

        with self.assertRaisesRegex(ValueError, "canonical Schedule replay"):
            compiler.lower(replace(assessment, target="sm_90"))

    def test_profile_cannot_generate_a_kernel_from_missing_schedule_semantics(self) -> None:
        compiler = Compiler.load(ROOT, REVISION_PATH)
        schedule = json.loads(
            (ROOT / "corpus/schedules/flash-kmeans-b32-smoke-v2.json").read_text()
        )
        schedule["operations"] = []

        assessment = compiler.assess(schedule)

        self.assertFalse(assessment.accepted)
        self.assertFalse(assessment.lowering_eligible)
        self.assertIn("SCHEDULE_OPERATIONS_EMPTY", [finding.code for finding in assessment.findings])
        self.assertIn("OUTPUT_UNWRITTEN", [finding.code for finding in assessment.findings])

    def test_program_tile_drift_without_its_buffers_is_refused(self) -> None:
        """There is no static template left to reuse, so the reason changed.

        Retiling the program axis and leaving the accumulator behind used to be caught
        because one file could serve one shape. It is caught now because the tile axis and the
        buffer it stages into disagree, which is a fact about the Schedule rather than
        about the file the Compiler happened to keep.
        """

        compiler = Compiler.load(ROOT, REVISION_PATH)
        schedule = json.loads(
            (ROOT / "corpus/schedules/flash-kmeans-b32-smoke-v2.json").read_text()
        )
        schedule["program_map"]["axes"][0]["tile"] = 128

        assessment = compiler.assess(schedule)

        self.assertEqual(assessment.analysis["grid"], (4, 32, 1))
        # Not merely unlowerable: a Schedule whose tile axis and staged buffer disagree
        # is internally inconsistent, so it is not accepted either. The shape-drift
        # Corpus cases stay accepted-but-unlowerable, because a shape is a Workload
        # question rather than a contradiction inside the Schedule.
        self.assertFalse(assessment.accepted)
        self.assertFalse(assessment.lowering_eligible)
        self.assertIn(
            "ACCESS_TILE_MISMATCH",
            [finding.code for finding in _decisive(assessment)],
        )

    def test_coherent_program_tile_revision_changes_the_lowered_program(self) -> None:
        compiler = Compiler.load(ROOT, REVISION_PATH)
        schedule = json.loads(
            (ROOT / "corpus/schedules/flash-kmeans-b32-smoke-v2.json").read_text()
        )
        schedule["schedule_id"] = "flash-kmeans-b32-smoke-block128"
        schedule["program_map"]["axes"][0]["tile"] = 128
        buffers = {buffer["name"]: buffer for buffer in schedule["buffers"]}
        buffers["token_tile"]["shape"][0] = 128
        buffers["distance_tile"]["shape"][0] = 128
        buffers["best_index_tile"]["shape"][0] = 128
        # A coherent retiling reaches every tile the composition names, not only the
        # ones the packed form happened to declare.
        buffers["cross"]["shape"][0] = 128
        buffers["scaled_cross"]["shape"][0] = 128
        for operation in schedule["operations"]:
            if operation["kind"] == "mma":
                operation["parameters"]["tile_shape"][0] = 128

        assessment = compiler.assess(schedule)
        lowering = compiler.lower(assessment)

        self.assertTrue(assessment.accepted)
        self.assertTrue(assessment.lowering_eligible)
        self.assertEqual(assessment.analysis["grid"], (4, 32, 1))
        # The tile now reaches the source through emission, not a parameter table.
        self.assertIn("BLOCK_TOKEN_BLOCK=128", compiler.lower(assessment).source)
        self.assertIn("[(4, 32, 1)]", lowering.source)
        self.assertIn("BLOCK_TOKEN_BLOCK=128", lowering.source)
        self.assertNotEqual(
            lowering.source_sha256,
            compiler.lower(
                compiler.assess_file(
                    ROOT / "corpus/schedules/flash-kmeans-b32-smoke-v2.json"
                )
            ).source_sha256,
        )

    def test_unlowered_operation_and_access_commitments_fail_closed(self) -> None:
        """Each violation names its own path instead of one opaque profile code.

        Every one of these used to report `PROFILE_SEMANTICS_MISMATCH` at
        `metadata.profile` -- true, but no repair target. A closed-vocabulary violation
        is now a structural Finding at the offending path, and an access map naming an
        axis that does not exist is a contract Finding from the verifier.
        """

        compiler = Compiler.load(ROOT, REVISION_PATH)
        original = json.loads(
            (ROOT / "corpus/schedules/flash-kmeans-b32-smoke-v2.json").read_text()
        )
        cases = (
            (
                # The arithmetic vocabulary took this position when the formula that
                # named a whole operator's math was removed; what is pinned is that a
                # closed vocabulary reports its path and its admitted values.
                lambda schedule: schedule["operations"][4]["parameters"].update(
                    {"op": "unsupported_op"}
                ),
                "SCHEDULE_STRUCTURE",
                "schedule.operations[4].parameters.op",
                "square",
            ),
            (
                lambda schedule: schedule["operations"][6]["parameters"].update(
                    {"tie_break": "highest_index"}
                ),
                "SCHEDULE_STRUCTURE",
                "schedule.operations[6].parameters.tie_break",
                "lowest_index",
            ),
            (
                lambda schedule: schedule["access_maps"][0]["indices"][0].update(
                    {"name": "wrong_batch"}
                ),
                "ACCESS_PROGRAM_AXIS_UNKNOWN",
                "access_maps[0].indices[0]",
                "wrong_batch",
            ),
        )
        for mutate, code, path, detail in cases:
            with self.subTest(code=code, path=path):
                schedule = json.loads(json.dumps(original))
                mutate(schedule)
                assessment = compiler.assess(schedule)
                self.assertFalse(assessment.accepted)
                self.assertFalse(assessment.lowering_eligible)
                finding = next(
                    item for item in assessment.findings if item.code == code
                )
                self.assertEqual(finding.path, path)
                self.assertIn(detail, finding.message)

    def test_passing_corpus_builds_a_content_bound_compiler_release(self) -> None:
        release = build_release(
            ROOT,
            ROOT / "compiler/revision.json",
            ROOT / "compiler/source_set.json",
            ROOT / "compiler/corpus-gate-report.json",
            ROOT / "compiler/release-approval.json",
        )

        self.assertEqual(release.document["state"], "released")
        self.assertEqual(release.document["corpus_gate"]["case_count"], 19)
        self.assertEqual(release.document["corpus_gate"]["matched_case_count"], 19)
        self.assertEqual(
            len(release.document["sources"]),
            len(json.loads((ROOT / "compiler" / "source_set.json").read_text())["paths"]),
        )
        self.assertTrue(release.verify(ROOT))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "revision.lock.json"
            path.write_text(json.dumps(release.document), encoding="utf-8")
            released_compiler = Compiler.load(ROOT, path)

        released_gate = released_compiler.check_corpus()
        # The released id advances with every Compiler change; assert against the lock
        # rather than a literal, so a Revision bump is not a test edit.
        released = json.loads(
            (ROOT / "compiler" / "revision.lock.json").read_text(encoding="utf-8")
        )
        self.assertEqual(released_gate.compiler_revision_id, released["revision_id"])
        self.assertTrue(released_gate.passed)

    def test_the_architecture_map_counts_the_profiles_that_exist(self) -> None:
        """The alignment table drifted twice in one day, the second time by one operator.

        It is the document that tells a reader what this implementation is missing, so a
        count in it that lags the registry sends someone to build what is already there.
        Both times the fix was to read the registry; this reads it.
        """

        import re

        from open_cake_ir.compiler.core import _PROFILES

        text = (ROOT / "docs/ARCHITECTURE.md").read_text(encoding="utf-8")
        match = re.search(r"generates for (\d+) of the (\d+) admitted profiles", text)
        self.assertIsNotNone(match, "the alignment table no longer states the counts")
        generating, total = (int(value) for value in match.groups())
        self.assertEqual(total, len(_PROFILES))
        self.assertEqual(
            generating, sum(1 for profile in _PROFILES.values() if profile.backend)
        )
        for name, profile in _PROFILES.items():
            with self.subTest(profile=name):
                # And every one of them is named, so the table cannot count right while
                # listing the wrong ones.
                self.assertIn(f"`{name}`", text)

    def test_a_schedule_outside_the_corpus_is_history_something_else_pins(self) -> None:
        """Why an ungated Schedule is allowed to sit in the corpus directory.

        Two do. They are the r16 shape the vocabulary has since moved past -- they name an
        epilogue formula this IR no longer admits, so they do not parse -- and a frozen
        KernelSeed and two historical Study Contracts reference them by path. Deleting
        them would break references that are supposed to be immutable; gating them is
        impossible, because the Compiler cannot read them.

        So the rule is: an ungated Schedule has to be both. Unreadable by the current
        Revision, and pinned by something frozen. A file that is readable and ungated is
        a corpus case someone forgot to register; one that is unreadable and unpinned is
        a leftover.
        """

        from open_cake_ir.compiler.ir import Schedule, ScheduleParseError

        manifest = json.loads(
            (ROOT / "corpus/manifest.json").read_text(encoding="utf-8")
        )
        gated = {case["schedule"] for case in manifest["cases"]}
        pinning = list((ROOT / "contracts").rglob("*.json")) + list(
            (ROOT / "evidence").rglob("*.json")
        )
        haystack = "\n".join(
            path.read_text(encoding="utf-8", errors="replace") for path in pinning
        )

        for path in sorted((ROOT / "corpus/schedules").glob("*.json")):
            relative = f"corpus/schedules/{path.name}"
            if relative in gated:
                continue
            with self.subTest(schedule=path.name):
                with self.assertRaises(ScheduleParseError):
                    Schedule.from_dict(json.loads(path.read_text(encoding="utf-8")))
                self.assertIn(relative, haystack)

    def test_the_revision_binds_every_schedule_its_gate_reads(self) -> None:
        """The Corpus Gate's own inputs have to be sealed, or the gate proves nothing.

        `corpus/manifest.json` is bound, so adding a case invalidates the lock. The
        Schedule a case points at is bound only if the source set lists it too, and the
        two lists were kept in step by hand -- so a gated Schedule could be edited under a
        released Revision without the Revision noticing. It was: two Schedules admitted
        with softmax sat outside the source set until this was checked.

        The release refuses that now. This holds the checked-in pair as well, because a
        Revision released before the check is the one nobody would re-run.
        """

        source_set = json.loads(
            (ROOT / "compiler/source_set.json").read_text(encoding="utf-8")
        )
        manifest = json.loads(
            (ROOT / "corpus/manifest.json").read_text(encoding="utf-8")
        )
        bound = set(source_set["paths"])
        gated = {case["schedule"] for case in manifest["cases"]}
        self.assertEqual(gated - bound, set())
        # And nothing bound as a corpus Schedule that no case reads: an unread Schedule
        # in the source set is content nobody gates.
        stale = {p for p in bound if p.startswith("corpus/schedules/")} - gated
        self.assertEqual(stale, set())

    def test_every_admitted_profile_has_a_corpus_case_that_lowers(self) -> None:
        """A profile is admitted by a row; a row that nothing exercises is a claim.

        The registry was reshaped so that an operator is one record plus the Schedules
        that claim it. Nothing held the second half: a profile could be added with a
        toolchain, a conformance rule and a backend, and no Schedule anywhere proving the
        combination lowers. This is what makes the corpus the evidence for the row.
        """

        from open_cake_ir.compiler.core import _PROFILES

        manifest = json.loads(
            (ROOT / "corpus/manifest.json").read_text(encoding="utf-8")
        )
        lowering: dict[str, list[str]] = {}
        for case in manifest["cases"]:
            document = json.loads(
                (ROOT / case["schedule"]).read_text(encoding="utf-8")
            )
            profile = document.get("metadata", {}).get("profile")
            if case["expected"]["lowering_eligible"]:
                lowering.setdefault(profile, []).append(case["case_id"])

        for name in _PROFILES:
            with self.subTest(profile=name):
                self.assertTrue(
                    lowering.get(name),
                    f"profile {name!r} is admitted but no corpus case lowers through it",
                )

    def test_full_compiler_corpus_gate_passes(self) -> None:
        compiler = Compiler.load(ROOT, REVISION_PATH)

        report = compiler.check_corpus()

        self.assertTrue(report.passed, report.cases)
        self.assertEqual(report.case_count, 19)
        # Every operator lands as a kernel plus the drift that proves its profile rule
        # fires. The three normalization drifts are rejected rather than merely
        # unlowerable: their staged tile contradicts the extent it addresses, which the
        # access-map rule could not see until it stopped comparing a store's global
        # output against the tile axes.
        self.assertEqual(report.accepted_case_count, 13)
        self.assertEqual(report.rejected_case_count, 6)
        self.assertEqual(report.lowerable_case_count, 9)
        self.assertEqual(report.nonlowerable_case_count, 10)

    def test_r16_program_map_schedule_uses_the_canonical_compiler(self) -> None:
        compiler = Compiler.load(ROOT, REVISION_PATH)

        assessment = compiler.assess_file(
            ROOT / "corpus/schedules/flash-kmeans-b32-smoke-v2.json"
        )

        self.assertTrue(assessment.accepted)
        self.assertTrue(assessment.lowering_eligible)
        self.assertEqual(assessment.analysis["grid"], (2, 32, 1))
        self.assertEqual(
            assessment.analysis["operation_counts"],
            {"load": 3, "mma": 1, "elementwise": 2, "reduce_argmin": 1, "store": 1},
        )

    def test_r16_workload_shape_drift_blocks_lowering(self) -> None:
        compiler = Compiler.load(ROOT, REVISION_PATH)
        schedule = json.loads(
            (ROOT / "corpus/schedules/flash-kmeans-b32-smoke-v2.json").read_text()
        )
        schedule["buffers"][0]["shape"][1] = 768

        assessment = compiler.assess(schedule)

        self.assertTrue(assessment.accepted)
        self.assertFalse(assessment.lowering_eligible)
        self.assertEqual(
            [(finding.code, finding.path) for finding in _decisive(assessment)],
            [("PROFILE_SHAPE_MISMATCH", "buffers.tokens.shape")],
        )

    def test_triton_warp_specialized_argmin_is_structured_feedback(self) -> None:
        compiler = Compiler.load(ROOT, REVISION_PATH)

        assessment = compiler.assess_file(
            ROOT
            / "corpus/schedules/flash-kmeans-b32-warp-specialized-argmin.json"
        )

        self.assertTrue(assessment.accepted)
        self.assertFalse(assessment.lowering_eligible)
        self.assertEqual(
            [(finding.code, finding.path) for finding in _decisive(assessment)],
            [
                (
                    "TRITON_WARP_SPECIALIZED_ARGMIN_UNSUPPORTED",
                    "tile_loops[0].range_options.warp_specialize",
                )
            ],
        )

    def test_r16_lowering_is_available_through_the_same_interface(self) -> None:
        compiler = Compiler.load(ROOT, REVISION_PATH)
        assessment = compiler.assess_file(
            ROOT / "corpus/schedules/flash-kmeans-b32-smoke-v2.json"
        )

        lowering = compiler.lower(assessment)

        self.assertEqual(lowering.entry_point, "cake_flash_kmeans_assign")
        self.assertIn("N_TOKEN_BLOCK=512", lowering.source)
        self.assertIn("[(2, 32, 1)]", lowering.source)
        self.assertEqual(
            set(lowering.source_map),
            {"load_tokens", "load_centroids", "load_norm", "distance_mma",
             "scale_cross", "distance", "argmin", "store_assignment"},
        )
        self.assertEqual(lowering.toolchain_requirements["target"], "sm_100a")
        self.assertEqual(
            lowering.toolchain_requirements["compile_constants"]["N_TOKEN_BLOCK"], 512
        )

    def test_frozen_seed_lowers_three_distinct_exact_shape_specialists(self) -> None:
        compiler = Compiler.load(ROOT, REVISION_PATH)
        seed = KernelSeed.load(
            ROOT, ROOT / "contracts/kernel-seeds/r42-cake-r1-turn1-v2.json"
        )
        workload = json.loads(
            (ROOT / "contracts/workloads/flash-kmeans-assign-v2.json").read_text()
        )
        by_id = {case["case_id"]: case["shape"] for case in workload["cases"]}
        cases = {
            case_id: by_id[case_id]
            for case_id in ("headline_b32", "b32_smoke", "public_b1")
        }

        first = lower_specialists(compiler, seed, cases)
        second = lower_specialists(compiler, seed, cases)

        self.assertEqual(
            [item.lowering.source_sha256 for item in first],
            [item.lowering.source_sha256 for item in second],
        )
        self.assertEqual(len({item.lowering.source_sha256 for item in first}), 3)
        self.assertEqual(
            [item.assessment.analysis["grid"] for item in first],
            [(256, 32, 1), (2, 32, 1), (256, 1, 1)],
        )
        self.assertEqual(seed.block_n, 256)

    def test_frozen_seed_rejects_a_non_exact_tail_without_retuning(self) -> None:
        compiler = Compiler.load(ROOT, REVISION_PATH)
        seed = KernelSeed.load(
            ROOT, ROOT / "contracts/kernel-seeds/r42-cake-r1-turn1-v2.json"
        )

        with self.assertRaisesRegex(ValueError, "exactly tiled"):
            lower_specialists(
                compiler,
                seed,
                {"tail_nk": {"B": 1, "N": 257, "K": 257, "D": 128}},
            )

    def test_exact_target_enforces_resource_and_instruction_contracts(self) -> None:
        compiler = Compiler.load(ROOT, REVISION_PATH)
        schedule = json.loads(
            (ROOT / "corpus/schedules/flash-kmeans-assignment-full.json").read_text()
        )
        schedule["allocations"][0]["size_bytes"] = 300000
        schedule["metadata"]["mma_instruction"] = "unbound.instruction"

        assessment = compiler.assess(schedule)

        self.assertFalse(assessment.accepted)
        self.assertEqual(
            [finding.code for finding in assessment.findings[:2]],
            ["TARGET_SHARED_MEMORY_LIMIT", "TARGET_INSTRUCTION_UNSUPPORTED"],
        )

        schedule = json.loads(
            (ROOT / "corpus/schedules/flash-kmeans-b32-smoke-v2.json").read_text()
        )
        # A space outside the IR vocabulary is a structural violation, reported at the
        # offending path. TARGET_MEMORY_SPACE_UNSUPPORTED remains reachable for a Target
        # that admits fewer spaces than the vocabulary; sm_100a admits all four.
        schedule["buffers"][0]["space"] = "unknown_space"
        assessment = compiler.assess(schedule)
        self.assertEqual(assessment.findings[0].path, "schedule.buffers[0].space")
        self.assertIn(
            "SCHEDULE_STRUCTURE",
            [finding.code for finding in _decisive(assessment)],
        )

    def test_r25_schedule_is_accepted_and_lowering_eligible(self) -> None:
        compiler = Compiler.load(ROOT, REVISION_PATH)

        assessment = compiler.assess_file(
            ROOT / "corpus/schedules/flash-kmeans-assignment-full.json"
        )

        self.assertTrue(assessment.accepted)
        self.assertTrue(assessment.lowering_eligible)
        self.assertEqual(assessment.target, "sm_100a")
        self.assertEqual(tuple(_decisive(assessment)), ())
        self.assertEqual(
            assessment.analysis["operation_counts"],
            {
                "epilogue": 1,
                "load": 2,
                "mma": 1,
                "reduce_argmin": 1,
                "store": 1,
            },
        )

    def test_r25_full_assignment_shape_drift_blocks_lowering(self) -> None:
        compiler = Compiler.load(ROOT, REVISION_PATH)
        schedule = json.loads(
            (ROOT / "corpus/schedules/flash-kmeans-assignment-full.json").read_text()
        )
        schedule["buffers"][3]["shape"] = [128, 256]

        assessment = compiler.assess(schedule)

        self.assertTrue(assessment.accepted)
        self.assertFalse(assessment.lowering_eligible)
        self.assertEqual(
            [(finding.code, finding.path) for finding in _decisive(assessment)],
            [("PROFILE_SHAPE_MISMATCH", "buffers.distance_scratch.shape")],
        )

    def test_r25_lowering_is_deterministic_and_inspectable(self) -> None:
        compiler = Compiler.load(ROOT, REVISION_PATH)
        assessment = compiler.assess_file(
            ROOT / "corpus/schedules/flash-kmeans-assignment-full.json"
        )

        first = compiler.lower(assessment)
        second = Compiler.load(ROOT, REVISION_PATH).lower(assessment)

        self.assertEqual(first.source, second.source)
        self.assertEqual(first.source_sha256, second.source_sha256)
        self.assertTrue(first.generated)
        self.assertEqual(first.entry_point, "cake_flash_kmeans_assignment_full")
        self.assertIn(f"# schedule_sha256={assessment.schedule_sha256}", first.source)
        self.assertEqual(
            set(first.source_map),
            {
                "load_tokens",
                "load_centroids",
                "dot_mma",
                "distance_epilogue",
                "argmin",
                "store_assignment",
            },
        )

    def test_r31_is_a_second_accepted_compiler_family(self) -> None:
        compiler = Compiler.load(ROOT, REVISION_PATH)

        assessment = compiler.assess_file(
            ROOT / "corpus/schedules/tinygemm2-stage4-split-k.json"
        )

        self.assertTrue(assessment.accepted)
        self.assertTrue(assessment.lowering_eligible)
        self.assertEqual(assessment.analysis["grid"], (64, 1, 1))
        self.assertEqual(assessment.analysis["total_warps"], 12)
        self.assertEqual(
            assessment.analysis["operation_counts"],
            {"epilogue": 1, "load": 3, "mma": 1, "reduce": 1},
        )

    def test_r31_reduction_semantic_drift_is_rejected(self) -> None:
        compiler = Compiler.load(ROOT, REVISION_PATH)
        schedule = json.loads(
            (ROOT / "corpus/schedules/tinygemm2-stage4-split-k.json").read_text()
        )
        # The part count is the extent of the collapsed axis, so drift is expressed
        # where that fact lives; the operation can no longer disagree with the buffer.
        for buffer in schedule["buffers"]:
            if buffer["name"] == "partial_accumulators":
                buffer["shape"] = [3, 16, 8]

        assessment = compiler.assess(schedule)

        self.assertFalse(assessment.accepted)
        self.assertFalse(assessment.lowering_eligible)
        self.assertEqual(
            [(finding.code, finding.path) for finding in _decisive(assessment)],
            [("REDUCE_SUM_SEMANTICS", "operations.reduce_partials.parameters.axis")],
        )

    def test_r31_lowering_uses_the_same_compiler_interface(self) -> None:
        compiler = Compiler.load(ROOT, REVISION_PATH)
        assessment = compiler.assess_file(
            ROOT / "corpus/schedules/tinygemm2-stage4-split-k.json"
        )

        lowering = compiler.lower(assessment)

        self.assertFalse(lowering.generated)
        self.assertEqual(lowering.entry_point, "cake_tinygemm2_stage4_split_k")
        self.assertIn(assessment.schedule_sha256, lowering.source)
        self.assertEqual(
            set(lowering.source_map),
            {
                "load_bias",
                "load_weight",
                "load_activation",
                "split_k_mma",
                "reduce_partials",
                "bias_epilogue",
            },
        )


if __name__ == "__main__":
    unittest.main()


class CandidateRankingTest(unittest.TestCase):
    """The pre-GPU filter stage, and the boundary it must not cross.

    The paper's loop ranks a set of candidates before spending GPU time. What matters as
    much as the order is that ranking happens *after* the gates and never argues with them:
    a candidate the verifier refused has no score, because a good score for a rejected
    Schedule would put the cost model in a position to overrule a hard gate.
    """

    def _variant(self, block_n: int, schedule_id: str) -> dict:
        document = json.loads(
            (ROOT / "corpus/schedules/flash-kmeans-b32-smoke-v2.json").read_text(
                encoding="utf-8"
            )
        )
        buffers = {item["name"]: item for item in document["buffers"]}
        document["schedule_id"] = schedule_id
        for axis in document["program_map"]["axes"]:
            if axis["name"] == "token_block":
                axis["tile"] = block_n
        buffers["token_tile"]["shape"] = [block_n, 128]
        buffers["best_index_tile"]["shape"] = [block_n]
        for name in ("distance_tile", "cross", "scaled_cross"):
            buffers[name]["shape"] = [block_n, 64]
        for operation in document["operations"]:
            if operation["kind"] == "mma" and "tile_shape" in operation["parameters"]:
                operation["parameters"]["tile_shape"] = [block_n, 64, 128]
        return document

    def test_uncalibrated_and_refused_candidates_are_withheld(self) -> None:
        compiler = Compiler.load(ROOT, REVISION_PATH)
        assessments = [
            compiler.assess(self._variant(64, "fits-a")),
            compiler.assess(self._variant(128, "fits-b")),
            compiler.assess(self._variant(512, "no-cta-is-resident")),
        ]
        self.assertFalse(assessments[2].lowering_eligible)

        scored, withheld = compiler.rank(assessments)

        self.assertEqual(scored, ())
        self.assertEqual(
            withheld,
            ("fits-a", "fits-b", "no-cta-is-resident"),
        )

    def test_ranking_rejects_forged_calibration_coverage(self) -> None:
        compiler = Compiler.load(ROOT, REVISION_PATH)
        assessment = compiler.assess(self._variant(64, "fits-a"))

        with self.assertRaisesRegex(CompilerError, "canonical Schedule replay"):
            compiler.rank([dataclasses.replace(assessment, calibration_available=True)])

    def test_ranking_refuses_an_assessment_from_another_revision(self) -> None:
        compiler = Compiler.load(ROOT, REVISION_PATH)
        assessment = compiler.assess(self._variant(64, "fits-a"))
        foreign = dataclasses.replace(assessment, compiler_revision_id="other-revision")
        with self.assertRaisesRegex(CompilerError, "different Compiler Revision"):
            compiler.rank([foreign])
