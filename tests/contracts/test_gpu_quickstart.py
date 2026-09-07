from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from open_cake_ir.tasks.workloads import load_workload

from open_cake_ir.compiler import Compiler  # noqa: E402
from open_cake_ir.evaluation import LaunchableCandidate, WorkloadContract  # noqa: E402
from open_cake_ir.lab import CandidateSubmission, ExecutorRevision
from open_cake_ir.tasks.environments import TaskOpenCakeEnvironment as OpenCakeEnvironment


def _resolve_executor(executor_id: str) -> dict:
    """Find a Revision descriptor by id: current, superseded, or archived."""

    inventory = json.loads(
        (ROOT / "inventory/EXECUTOR_REVISIONS.json").read_text(encoding="utf-8")
    )
    for candidate in [
        inventory["current"],
        *inventory.get("superseded", []),
        *inventory["archives"],
    ]:
        if candidate["executor_id"] == executor_id:
            return candidate
    raise AssertionError(f"executor {executor_id!r} is not resolvable")


class GpuQuickstartContractTests(unittest.TestCase):
    def test_existing_output_blocks_before_any_gpu_work(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            output.write_bytes(b"existing\n")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "examples/gpu/flash_kmeans_quickstart.py"),
                    "--project-root",
                    str(ROOT),
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=30,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn(b"--output must be a new path", completed.stderr)
            self.assertEqual(output.read_bytes(), b"existing\n")

    def test_quickstart_case_and_executor_authorities_cannot_be_overridden(self) -> None:
        for option in ("--case-id", "--executor"):
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "examples/gpu/flash_kmeans_quickstart.py"),
                    "--project-root",
                    str(ROOT),
                    "--prepare-only",
                    option,
                    "outside-authority",
                ],
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=30,
            )

            self.assertNotEqual(completed.returncode, 0, option)

    def test_live_quickstart_inventory_binds_its_runner_and_executor(self) -> None:
        """The observation names the Executor Revision it actually ran under.

        Once a runtime source changes, that Revision is superseded and its closure no
        longer verifies against this tree -- verifying it belongs to its archive. The
        binding is still checked, by descriptor identity rather than by loading, because
        restamping the observation to the current Revision would claim a run that never
        happened.
        """

        inventory = json.loads(
            (
                ROOT / "inventory/GPU_QUICKSTART_QUALIFICATION_V3_20260823.json"
            ).read_text(encoding="utf-8")
        )
        runner = ROOT / inventory["runner"]["path"]
        executor_inventory = _resolve_executor(
            inventory["executor_revision"]["executor_id"]
        )
        descriptor = json.loads(
            (ROOT / executor_inventory["path"]).read_text(encoding="utf-8")
        )

        self.assertEqual(inventory["status"], "passed")
        self.assertFalse(inventory["scientific_claim_authorized"])
        self.assertFalse(inventory["performance_measured"])
        # The runner moved with the Schedule when the packed mma form was retired. A
        # superseded record keeps the bytes its run actually used, so what is checkable
        # is that the record says so -- not that the tree still holds those bytes.
        self.assertNotEqual(
            sha256(runner.read_bytes()).hexdigest(), inventory["runner"]["raw_sha256"]
        )
        self.assertEqual(
            inventory["superseded"]["observed_runner_raw_sha256"],
            inventory["runner"]["raw_sha256"],
        )
        # That supersession statement is itself frozen. Later runner revisions do not
        # turn its then-current digest into a writable pointer to HEAD.
        self.assertNotEqual(
            inventory["superseded"]["current_runner_raw_sha256"],
            sha256(runner.read_bytes()).hexdigest(),
        )
        # The Schedule this run observed has since gained its hardware commitments, and
        # the broker on this host does not export GPUQ_JOB_ID, so the run cannot be
        # re-observed. The record says so rather than being restamped to bytes it never
        # saw; what stays checkable is checked.
        superseded = inventory["superseded"]
        self.assertEqual(
            superseded["observed_schedule_raw_sha256"], inventory["schedule"]["raw_sha256"]
        )
        self.assertEqual(
            sha256((ROOT / inventory["schedule"]["path"]).read_bytes()).hexdigest(),
            superseded["current_schedule_raw_sha256"],
        )
        self.assertIn("GPUQ_JOB_ID", superseded["requalification_blocked_by"])
        self.assertEqual(
            executor_inventory["canonical_sha256"],
            inventory["executor_revision"]["canonical_sha256"],
        )
        self.assertEqual(
            sha256((ROOT / executor_inventory["path"]).read_bytes()).hexdigest(),
            inventory["executor_revision"]["descriptor_raw_sha256"],
        )
        executor_sources = {
            record["path"]: record for record in descriptor["sources"]
        }
        self.assertEqual(
            executor_sources[inventory["runner"]["path"]]["sha256"],
            inventory["runner"]["raw_sha256"],
        )
        for observation in inventory["raw_observation"].values():
            path = ROOT / observation["path"]
            self.assertEqual(
                sha256(path.read_bytes()).hexdigest(),
                observation["raw_sha256"],
            )
        prior_fault = inventory["prior_custody_fault"]
        self.assertFalse(prior_fault["result_retained"])
        self.assertIsNone(prior_fault["kernel_calls"])
        self.assertFalse(prior_fault["candidate_conclusion_authorized"])
        for observation in prior_fault["raw_observation"].values():
            path = ROOT / observation["path"]
            self.assertEqual(
                sha256(path.read_bytes()).hexdigest(),
                observation["raw_sha256"],
            )
        prior_stderr = (
            ROOT / prior_fault["raw_observation"]["stderr"]["path"]
        ).read_text(encoding="utf-8")
        self.assertIn(prior_fault["job_id"], prior_stderr)
        self.assertIn("PermissionError", prior_stderr)
        result = json.loads(
            (ROOT / inventory["raw_observation"]["result"]["path"]).read_text(
                encoding="utf-8"
            )
        )
        stderr = (
            ROOT / inventory["raw_observation"]["stderr"]["path"]
        ).read_text(encoding="utf-8")
        stdout = ROOT / inventory["raw_observation"]["stdout"]["path"]
        result_path = ROOT / inventory["raw_observation"]["result"]["path"]

        self.assertEqual(stdout.read_bytes(), result_path.read_bytes())
        self.assertEqual(result["status"], inventory["status"])
        self.assertEqual(result["compiler_revision"], inventory["compiler_revision"])
        self.assertEqual(result["workload"], inventory["workload"])
        self.assertEqual(result["candidate"], inventory["candidate"])
        self.assertEqual(
            {
                key: result["executor_revision"][key]
                for key in ("executor_id", "canonical_sha256")
            },
            {
                key: inventory["executor_revision"][key]
                for key in ("executor_id", "canonical_sha256")
            },
        )
        self.assertEqual(
            result["executor_revision"]["executor_id"],
            executor_inventory["executor_id"],
        )
        self.assertEqual(
            {
                "cubin_sha256": result["build"]["cubin_sha256"],
                "cubin_size_bytes": result["build"]["cubin_size_bytes"],
                "lowering_source_sha256": result["lowering"]["source_sha256"],
            },
            inventory["build"],
        )
        self.assertEqual(
            result["assessment"]["schedule_id"],
            inventory["schedule"]["schedule_id"],
        )
        self.assertEqual(
            result["assessment"]["schedule_sha256"],
            inventory["schedule"]["canonical_sha256"],
        )
        result_gpu = result["gpu"]
        inventory_gpu = inventory["gpu"]
        for key in (
            "name",
            "compute_capability",
            "mode",
            "exclusive_b200_preflight_passed",
        ):
            self.assertEqual(result_gpu[key], inventory_gpu[key], key)
        result_evaluation = result["evaluation"]
        inventory_evaluation = inventory["evaluation"]
        for key in (
            "evaluation_protocol_sha256",
            "evaluation_receipt_sha256",
            "correctness_passed",
            "kernel_calls",
            "fallback_calls",
            "module_unloaded",
        ):
            self.assertEqual(result_evaluation[key], inventory_evaluation[key], key)
        for key in (
            "exact_match",
            "tie_aware_distance_match",
            "mismatch_count",
            "total_assignments",
        ):
            self.assertEqual(
                result_evaluation["metrics"][key],
                inventory_evaluation[key],
                key,
            )
        self.assertEqual(
            result_evaluation["performance_measured"],
            inventory["performance_measured"],
        )
        self.assertEqual(
            result_evaluation["scientific_claim_authorized"],
            inventory["scientific_claim_authorized"],
        )
        self.assertIn(inventory["gpu"]["job_id"], stderr)
        self.assertTrue(inventory["evaluation"]["correctness_passed"])
        self.assertEqual(inventory["evaluation"]["kernel_calls"], 1)
        self.assertEqual(inventory["evaluation"]["fallback_calls"], 0)
        self.assertTrue(inventory["evaluation"]["module_unloaded"])

    def test_tutorial_schedule_crosses_the_workload_bound_launchable_seam(self) -> None:
        class RecordingToolchain:
            request = None

            def build(self, request):
                self.request = request
                payloads = {
                    "lowered_source": request.source,
                    "cubin": b"\x7fELF teaching fixture",
                    "launch_manifest": b'{"teaching":true}',
                }
                return LaunchableCandidate(
                    candidate_sha256=request.candidate_sha256,
                    target=request.target,
                    entry_point=request.entry_point,
                    artifact_roles={
                        role: sha256(payload).hexdigest()
                        for role, payload in payloads.items()
                    },
                    launch_spec_sha256=sha256(payloads["launch_manifest"]).hexdigest(),
                    artifact_payloads=payloads,
                )

        compiler = Compiler.load(ROOT, ROOT / "compiler/revision.lock.json")
        workload = load_workload(
            ROOT / "contracts/workloads/flash-kmeans-assign-v2.json"
        )
        toolchain = RecordingToolchain()
        environment = OpenCakeEnvironment(
            compiler,
            toolchain,
            authority_document={
                "lowering_route": {
                    "backend": "triton",
                    "entry_point": "cake_flash_kmeans_assign",
                }
            },
            workload=workload,
            case_id="b32_smoke",
        )
        schedule = (ROOT / "examples/gpu/flash-kmeans-b32-smoke-v2.json").read_bytes()

        result = environment.build(
            CandidateSubmission.seal(
                "application/vnd.open-cake.schedule+json",
                schedule,
            )
        )

        self.assertEqual(result.disposition, "launchable")
        self.assertIsNotNone(result.launchable)
        self.assertEqual(toolchain.request.target, "sm_100a")
        self.assertEqual(
            json.loads(schedule)["metadata"]["workload_contract_sha256"],
            workload.canonical_sha256,
        )

    def test_prepare_only_explains_the_compiler_path_without_submitting_gpu_work(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "examples/gpu/flash_kmeans_quickstart.py"),
                "--project-root",
                str(ROOT),
                "--prepare-only",
            ],
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr.decode())
        summary = json.loads(completed.stdout)
        self.assertEqual(summary["kind"], "open_cake_gpu_quickstart_v1")
        self.assertEqual(summary["status"], "prepared")
        self.assertTrue(summary["assessment"]["accepted"])
        self.assertTrue(summary["assessment"]["lowering_eligible"])
        # The teaching path shows what the compiler reports, and for this Schedule that
        # includes why residency is bounded. Nothing here blocks acceptance.
        self.assertEqual(
            sorted(f["code"] for f in summary["assessment"]["findings"]),
            ["REGISTER_PRESSURE", "RESIDENCY_BOUND"],
        )
        self.assertTrue(summary["assessment"]["accepted"])
        self.assertTrue(summary["assessment"]["lowering_eligible"])
        self.assertEqual(summary["evaluation"], {"gpu_submitted": False})


if __name__ == "__main__":
    unittest.main()
