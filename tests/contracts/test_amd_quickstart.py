from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples/gpu"))
sys.path.insert(0, str(ROOT / "src"))

import amd_triton_quickstart as quickstart  # noqa: E402
from open_cake_ir.evaluation.triton_hip import amdgcn_resource_record  # noqa: E402

SCRIPT = ROOT / "examples/gpu/swiglu_amd_quickstart.py"
RMSNORM_SCRIPT = ROOT / "examples/gpu/rmsnorm_amd_quickstart.py"
RMSNORM_ONE_ROW = (
    ROOT / "corpus/schedules/llama-rmsnorm-mul-b8-gfx1151-r1-w8.json"
)
SWIGLU_SCHEDULE = ROOT / "corpus/schedules/swiglu-b8-smoke-gfx1151.json"
SWIGLU_WORKLOAD = ROOT / "contracts/workloads/swiglu-fp32-v1.json"


class AmdQuickstartContractTests(unittest.TestCase):
    def test_amdgcn_resources_are_structured_without_inventing_occupancy(self) -> None:
        source = b"""
        .amdhsa_kernel _fixture
          .amdhsa_group_segment_fixed_size 256
          .amdhsa_private_segment_fixed_size 64
          .amdhsa_kernarg_size 40
          .amdhsa_wavefront_size32 1
          .amdhsa_uses_dynamic_stack 0
          .amdhsa_next_free_vgpr 117
          .amdhsa_next_free_sgpr 39
          .amdhsa_shared_vgpr_count 0
          .amdhsa_workgroup_processor_mode 1
        .end_amdhsa_kernel
        """

        record = amdgcn_resource_record(source)

        self.assertEqual(record["kernel_name"], "_fixture")
        self.assertEqual(record["wave_size"], 32)
        self.assertEqual(record["vgpr_count"], 117)
        self.assertEqual(record["sgpr_count"], 39)
        self.assertEqual(record["lds_bytes_per_workgroup"], 256)
        self.assertEqual(record["scratch_bytes_per_workitem"], 64)
        self.assertFalse(record["occupancy_derived"])
        self.assertEqual(
            record["occupancy_limit"], "gfx1151_target_facts_unavailable"
        )

        with self.assertRaisesRegex(ValueError, "resource declarations differ"):
            amdgcn_resource_record(
                source.replace(b"wavefront_size32 1", b"wavefront_size32 0")
            )

    def test_gpu_execution_refuses_a_draft_or_failed_compiler_gate(self) -> None:
        draft = SimpleNamespace(state="draft", check_corpus=Mock())
        with self.assertRaisesRegex(ValueError, "released Compiler"):
            quickstart._admit_released_compiler(draft)
        draft.check_corpus.assert_not_called()

        failed = SimpleNamespace(
            state="released", check_corpus=Mock(return_value=SimpleNamespace(passed=False))
        )
        with self.assertRaisesRegex(ValueError, "passing released Corpus Gate"):
            quickstart._admit_released_compiler(failed)

        gate = SimpleNamespace(passed=True)
        released = SimpleNamespace(
            state="released", check_corpus=Mock(return_value=gate)
        )
        self.assertIs(quickstart._admit_released_compiler(released), gate)

    def test_live_preflight_failure_retains_exact_attempt_authority(self) -> None:
        compiler = SimpleNamespace(state="draft", check_corpus=Mock())
        executor = Mock()
        executor.reference = {
            "path": "runtime/executors/open-cake-ir-gfx1151-v1.json",
            "executor_id": "open-cake-ir-gfx1151-v1",
            "canonical_sha256": "a" * 64,
        }
        summary = {
            "kind": "fixture-quickstart",
            "compiler_revision": {
                "revision_id": "open-cake-ir-v29-draft",
                "canonical_sha256": "b" * 64,
            },
            "assessment": {"schedule_id": "fixture"},
            "workload": {"workload_id": "fixture"},
        }
        with tempfile.TemporaryDirectory() as directory:
            artifact_dir = Path(directory) / "attempt"
            with self.assertRaisesRegex(ValueError, "released Compiler"):
                quickstart._run_gpu(
                    ROOT,
                    compiler,
                    executor,
                    SimpleNamespace(toolchain_requirements={}),
                    Mock(),
                    Mock(),
                    summary,
                    artifact_dir=artifact_dir,
                )

            failure = json.loads(
                (artifact_dir / "failure.json").read_text(encoding="utf-8")
            )
            self.assertEqual(failure["failed_stage"], "compiler_admission")
            self.assertEqual(failure["failure_class"], "AUTHORITY_BLOCKED")
            self.assertEqual(failure["status"], "AUTHORITY_BLOCKED")
            self.assertEqual(
                failure["authority"]["executor"]["executor_id"],
                "open-cake-ir-gfx1151-v1",
            )
            self.assertFalse(failure["gpu_result_authorized"])
            self.assertTrue((artifact_dir / "attempt-authority.json").is_file())
            self.assertTrue((artifact_dir / "manifest.json").is_file())
            executor.admit_hip_host.assert_not_called()

    def test_prepare_only_reaches_the_exact_uncalibrated_amd_lowering(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--project-root",
                str(ROOT),
                "--revision",
                str(ROOT / "compiler/revision.json"),
                "--prepare-only",
            ],
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout)
        self.assertEqual(result["status"], "prepared")
        self.assertEqual(result["assessment"]["target"], "gfx1151")
        self.assertEqual(
            result["assessment"]["route"],
            {
                "backend": "triton",
                "entry_point": "cake_swiglu_b8_smoke_gfx1151",
            },
        )
        self.assertFalse(result["assessment"]["calibration_available"])
        self.assertEqual(result["workload"]["workload_id"], "swiglu-fp32-independent-v1")
        self.assertEqual(
            result["workload"]["case_ids"], ["seeded_random", "signed_saturation"]
        )
        self.assertEqual(result["lowering"]["toolchain_requirements"]["binary_role"], "hsaco")
        self.assertFalse(result["evaluation"]["gpu_submitted"])
        self.assertFalse(result["evaluation"]["performance_measured"])

    def test_prepare_only_reaches_the_pinned_llama_rmsnorm_mul_slice(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                str(RMSNORM_SCRIPT),
                "--project-root",
                str(ROOT),
                "--revision",
                str(ROOT / "compiler/revision.json"),
                "--prepare-only",
            ],
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout)
        self.assertEqual(result["status"], "prepared")
        self.assertEqual(result["assessment"]["target"], "gfx1151")
        self.assertEqual(
            result["assessment"]["route"],
            {
                "backend": "triton",
                "entry_point": "cake_llama_rmsnorm_mul_gfx1151_r64_w4",
            },
        )
        self.assertEqual(
            result["workload"]["workload_id"],
            "llama-rmsnorm-mul-fp32-independent-v2",
        )
        self.assertEqual(
            result["workload"]["case_ids"],
            ["seeded_random", "reduction_rsqrt_stress"],
        )
        self.assertEqual(
            result["lowering"]["toolchain_requirements"]["binary_role"],
            "hsaco",
        )
        self.assertFalse(result["evaluation"]["gpu_submitted"])

    def test_schedule_owns_the_one_row_candidate_entry_point(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                str(RMSNORM_SCRIPT),
                "--project-root",
                str(ROOT),
                "--revision",
                str(ROOT / "compiler/revision.json"),
                "--schedule",
                str(RMSNORM_ONE_ROW),
                "--prepare-only",
            ],
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout)
        self.assertEqual(result["status"], "prepared")
        self.assertEqual(
            result["assessment"]["route"]["entry_point"],
            "cake_llama_rmsnorm_mul_gfx1151_r1_w8",
        )
        self.assertEqual(
            result["lowering"]["toolchain_requirements"]["compile_options"],
            {"num_warps": 8},
        )
        self.assertFalse(result["evaluation"]["gpu_submitted"])

    def test_custom_schedule_cannot_switch_the_workload_operator(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                str(RMSNORM_SCRIPT),
                "--project-root",
                str(ROOT),
                "--revision",
                str(ROOT / "compiler/revision.json"),
                "--schedule",
                str(SWIGLU_SCHEDULE),
                "--workload",
                str(SWIGLU_WORKLOAD),
                "--prepare-only",
            ],
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("Workload operator differs", completed.stderr)

    def test_gpu_execution_requires_an_exact_executor_before_compiler_loading(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                str(RMSNORM_SCRIPT),
                "--project-root",
                str(ROOT),
                "--revision",
                str(ROOT / "compiler/revision.json"),
            ],
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("--executor is required for GPU execution", completed.stderr)

    def test_artifacts_must_stay_outside_the_checkout(self) -> None:
        blocked = ROOT / "amd-quickstart-artifacts-forbidden"
        self.assertFalse(blocked.exists())
        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--project-root",
                str(ROOT),
                "--revision",
                str(ROOT / "compiler/revision.json"),
                "--prepare-only",
                "--artifact-dir",
                str(blocked),
            ],
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("outside the checkout", completed.stderr)
        self.assertFalse(blocked.exists())


if __name__ == "__main__":
    unittest.main()
