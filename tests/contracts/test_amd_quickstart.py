from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "examples/gpu/swiglu_amd_quickstart.py"
RMSNORM_SCRIPT = ROOT / "examples/gpu/rmsnorm_amd_quickstart.py"


class AmdQuickstartContractTests(unittest.TestCase):
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
