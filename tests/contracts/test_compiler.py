from __future__ import annotations

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

from open_cake_ir.compiler import Compiler  # noqa: E402
from open_cake_ir.compiler.release import build_release  # noqa: E402
from open_cake_ir.evidence import EvidenceStore  # noqa: E402
from open_cake_ir.lab import KernelSeed, lower_specialists  # noqa: E402


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
            (ROOT / "corpus/schedules/flash-kmeans-b32-smoke.json").read_text()
        )
        schedule["target"] = "sm_90"

        assessment = compiler.assess(schedule)

        self.assertFalse(assessment.accepted)
        self.assertIn("TARGET_UNSUPPORTED", [item.code for item in assessment.findings])
        self.assertFalse(assessment.calibration_available)

    def test_lower_rejects_a_forged_assessment_projection(self) -> None:
        compiler = Compiler.load(ROOT, REVISION_PATH)
        assessment = compiler.assess_file(
            ROOT / "corpus/schedules/flash-kmeans-b32-smoke.json"
        )

        with self.assertRaisesRegex(ValueError, "canonical Schedule replay"):
            compiler.lower(replace(assessment, target="sm_90"))

    def test_profile_cannot_generate_a_kernel_from_missing_schedule_semantics(self) -> None:
        compiler = Compiler.load(ROOT, REVISION_PATH)
        schedule = json.loads(
            (ROOT / "corpus/schedules/flash-kmeans-b32-smoke.json").read_text()
        )
        schedule["operations"] = []

        assessment = compiler.assess(schedule)

        self.assertFalse(assessment.accepted)
        self.assertFalse(assessment.lowering_eligible)
        self.assertIn("SCHEDULE_OPERATIONS_EMPTY", [finding.code for finding in assessment.findings])
        self.assertIn("OUTPUT_UNWRITTEN", [finding.code for finding in assessment.findings])

    def test_program_tile_drift_cannot_reuse_a_static_lowering_template(self) -> None:
        compiler = Compiler.load(ROOT, REVISION_PATH)
        schedule = json.loads(
            (ROOT / "corpus/schedules/flash-kmeans-b32-smoke.json").read_text()
        )
        schedule["program_map"]["axes"][0]["tile"] = 128

        assessment = compiler.assess(schedule)

        self.assertEqual(assessment.analysis["grid"], (4, 32, 1))
        self.assertTrue(assessment.accepted)
        self.assertFalse(assessment.lowering_eligible)
        self.assertEqual(
            [finding.code for finding in assessment.findings],
            ["PROFILE_SEMANTICS_MISMATCH"],
        )

    def test_coherent_program_tile_revision_changes_the_lowered_program(self) -> None:
        compiler = Compiler.load(ROOT, REVISION_PATH)
        schedule = json.loads(
            (ROOT / "corpus/schedules/flash-kmeans-b32-smoke.json").read_text()
        )
        schedule["schedule_id"] = "flash-kmeans-b32-smoke-block128"
        schedule["program_map"]["axes"][0]["tile"] = 128
        buffers = {buffer["name"]: buffer for buffer in schedule["buffers"]}
        buffers["token_tile"]["shape"][0] = 128
        buffers["distance_tile"]["shape"][0] = 128
        buffers["best_index_tile"]["shape"][0] = 128

        assessment = compiler.assess(schedule)
        lowering = compiler.lower(assessment)

        self.assertTrue(assessment.accepted)
        self.assertTrue(assessment.lowering_eligible)
        self.assertEqual(assessment.analysis["grid"], (4, 32, 1))
        self.assertEqual(assessment.lowering_parameters["block_n"], 128)
        self.assertIn("grid = (4, 32, 1)", lowering.source)
        self.assertIn("BLOCK_N=128", lowering.source)
        self.assertNotEqual(
            lowering.source_sha256,
            compiler.lower(
                compiler.assess_file(
                    ROOT / "corpus/schedules/flash-kmeans-b32-smoke.json"
                )
            ).source_sha256,
        )

    def test_unlowered_operation_and_access_commitments_fail_closed(self) -> None:
        compiler = Compiler.load(ROOT, REVISION_PATH)
        original = json.loads(
            (ROOT / "corpus/schedules/flash-kmeans-b32-smoke.json").read_text()
        )
        mutations = (
            lambda schedule: schedule["operations"][2]["parameters"].update(
                {"formula": "unsupported_formula"}
            ),
            lambda schedule: schedule["operations"][3]["parameters"].update(
                {"tie_break": "highest_index"}
            ),
            lambda schedule: schedule["access_maps"][0]["indices"][0].update(
                {"name": "wrong_batch"}
            ),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                schedule = json.loads(json.dumps(original))
                mutate(schedule)
                assessment = compiler.assess(schedule)
                self.assertTrue(assessment.accepted)
                self.assertFalse(assessment.lowering_eligible)
                self.assertIn(
                    "PROFILE_SEMANTICS_MISMATCH",
                    [finding.code for finding in assessment.findings],
                )

    def test_passing_corpus_builds_a_content_bound_compiler_release(self) -> None:
        release = build_release(
            ROOT,
            ROOT / "compiler/revision.json",
            ROOT / "compiler/source_set.json",
            ROOT / "compiler/corpus-gate-report.json",
            ROOT / "compiler/release-approval.json",
        )

        self.assertEqual(release.document["state"], "released")
        self.assertEqual(release.document["corpus_gate"]["case_count"], 6)
        self.assertEqual(release.document["corpus_gate"]["matched_case_count"], 6)
        self.assertEqual(len(release.document["sources"]), 16)
        self.assertTrue(release.verify(ROOT))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "revision.lock.json"
            path.write_text(json.dumps(release.document), encoding="utf-8")
            released_compiler = Compiler.load(ROOT, path)

        released_gate = released_compiler.check_corpus()
        self.assertEqual(released_gate.compiler_revision_id, "open-cake-ir-sm100a-v3")
        self.assertTrue(released_gate.passed)

    def test_full_compiler_corpus_gate_passes(self) -> None:
        compiler = Compiler.load(ROOT, REVISION_PATH)

        report = compiler.check_corpus()

        self.assertTrue(report.passed, report.cases)
        self.assertEqual(report.case_count, 6)
        self.assertEqual(report.accepted_case_count, 5)
        self.assertEqual(report.rejected_case_count, 1)
        self.assertEqual(report.lowerable_case_count, 3)
        self.assertEqual(report.nonlowerable_case_count, 3)

    def test_r16_program_map_schedule_uses_the_canonical_compiler(self) -> None:
        compiler = Compiler.load(ROOT, REVISION_PATH)

        assessment = compiler.assess_file(
            ROOT / "corpus/schedules/flash-kmeans-b32-smoke.json"
        )

        self.assertTrue(assessment.accepted)
        self.assertTrue(assessment.lowering_eligible)
        self.assertEqual(assessment.analysis["grid"], (2, 32, 1))
        self.assertEqual(
            assessment.analysis["operation_counts"],
            {"load": 2, "mma": 1, "reduce_argmin": 1, "store": 1},
        )

    def test_r16_workload_shape_drift_blocks_lowering(self) -> None:
        compiler = Compiler.load(ROOT, REVISION_PATH)
        schedule = json.loads(
            (ROOT / "corpus/schedules/flash-kmeans-b32-smoke.json").read_text()
        )
        schedule["buffers"][0]["shape"][1] = 768

        assessment = compiler.assess(schedule)

        self.assertTrue(assessment.accepted)
        self.assertFalse(assessment.lowering_eligible)
        self.assertEqual(
            [(finding.code, finding.path) for finding in assessment.findings],
            [("PROFILE_SHAPE_MISMATCH", "buffers.tokens.shape")],
        )

    def test_r16_lowering_is_available_through_the_same_interface(self) -> None:
        compiler = Compiler.load(ROOT, REVISION_PATH)
        assessment = compiler.assess_file(
            ROOT / "corpus/schedules/flash-kmeans-b32-smoke.json"
        )

        lowering = compiler.lower(assessment)

        self.assertEqual(lowering.entry_point, "cake_flash_kmeans_assign")
        self.assertIn("N=512", lowering.source)
        self.assertIn("grid = (2, 32, 1)", lowering.source)
        self.assertEqual(
            set(lowering.source_map),
            {"load_tokens", "load_centroids", "distance_mma", "argmin", "store_assignment"},
        )
        self.assertEqual(lowering.toolchain_requirements["target"], "sm_100a")
        self.assertEqual(
            lowering.toolchain_requirements["compile_constants"],
            {"B": 32, "N": 512, "K": 1024, "D": 128, "BLOCK_N": 256, "BLOCK_K": 64, "NUM_STAGES": 2},
        )

    def test_frozen_seed_lowers_three_distinct_exact_shape_specialists(self) -> None:
        compiler = Compiler.load(ROOT, REVISION_PATH)
        seed = KernelSeed.load(
            ROOT, ROOT / "contracts/kernel-seeds/r42-cake-r1-turn1.json"
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
            ROOT, ROOT / "contracts/kernel-seeds/r42-cake-r1-turn1.json"
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
            (ROOT / "corpus/schedules/flash-kmeans-b32-smoke.json").read_text()
        )
        schedule["buffers"][0]["space"] = "unknown_space"
        assessment = compiler.assess(schedule)
        self.assertIn(
            "TARGET_MEMORY_SPACE_UNSUPPORTED",
            [finding.code for finding in assessment.findings],
        )

    def test_r25_schedule_is_accepted_and_lowering_eligible(self) -> None:
        compiler = Compiler.load(ROOT, REVISION_PATH)

        assessment = compiler.assess_file(
            ROOT / "corpus/schedules/flash-kmeans-assignment-full.json"
        )

        self.assertTrue(assessment.accepted)
        self.assertTrue(assessment.lowering_eligible)
        self.assertEqual(assessment.target, "sm_100a")
        self.assertEqual(assessment.findings, ())
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
            [(finding.code, finding.path) for finding in assessment.findings],
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
            {"epilogue": 1, "load": 3, "mma": 1, "reduce_sum": 1},
        )

    def test_r31_reduction_semantic_drift_is_rejected(self) -> None:
        compiler = Compiler.load(ROOT, REVISION_PATH)
        schedule = json.loads(
            (ROOT / "corpus/schedules/tinygemm2-stage4-split-k.json").read_text()
        )
        schedule["operations"][4]["parameters"]["parts"] = 3

        assessment = compiler.assess(schedule)

        self.assertFalse(assessment.accepted)
        self.assertFalse(assessment.lowering_eligible)
        self.assertEqual(
            [(finding.code, finding.path) for finding in assessment.findings],
            [("REDUCE_SUM_SEMANTICS", "operations.reduce_partials.parameters.parts")],
        )

    def test_r31_lowering_uses_the_same_compiler_interface(self) -> None:
        compiler = Compiler.load(ROOT, REVISION_PATH)
        assessment = compiler.assess_file(
            ROOT / "corpus/schedules/tinygemm2-stage4-split-k.json"
        )

        lowering = compiler.lower(assessment)

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
