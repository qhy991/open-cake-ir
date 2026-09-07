from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples/gpu"))

import llama_q8_1_amd_quickstart as quickstart  # noqa: E402
from open_cake_ir.compiler import Compiler  # noqa: E402
from open_cake_ir.evaluation import (  # noqa: E402
    WorkloadContract,
    materialize_q4_mmvq_case,
    q4_mmvq_reference,
)
from tools.release_executor import _source_paths  # noqa: E402


SCRIPT = ROOT / "examples/gpu/llama_q8_1_amd_quickstart.py"
WORKLOAD = ROOT / "contracts/workloads/llama-q4_0-q8_1-mmvq-f32-v1.json"


def _executor(*, owns_runner: bool) -> SimpleNamespace:
    sources = (
        ({"path": quickstart.RUNNER_SOURCE} if owns_runner else {"path": "other.py"}),
    )
    return SimpleNamespace(
        executor_id="open-cake-ir-gfx1151-v1",
        document={
            "schema_version": 2,
            "sources": sources,
            "host_environment": {"runtime_kind": "hip"},
        },
    )


class LlamaQ81AmdQuickstartTests(unittest.TestCase):
    def test_prepare_only_is_deterministic_on_the_draft_revision(self) -> None:
        command = [
            sys.executable,
            str(SCRIPT),
            "--project-root",
            str(ROOT),
            "--prepare-only",
        ]
        first = subprocess.run(
            command,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        second = subprocess.run(
            command,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(first.stdout, second.stdout)
        result = json.loads(first.stdout)
        self.assertEqual(result["status"], "prepared")
        self.assertEqual(result["compiler_revision"]["state"], "draft")
        self.assertEqual(result["compiler_revision"]["path"], "compiler/revision.json")
        self.assertEqual(
            tuple(result["workload"]["case_ids"]), quickstart._EXPECTED_CASE_IDS
        )
        self.assertEqual(
            result["assessment"]["route"],
            {
                "backend": "triton",
                "entry_point": "cake_packed_q8_1_producer_gfx1151",
            },
        )
        requirements = result["lowering"]["toolchain_requirements"]
        self.assertEqual(requirements["target"], "gfx1151")
        self.assertEqual(requirements["triton_target"],
                         {"backend": "hip", "arch": "gfx1151", "warp_size": 32})
        self.assertEqual(requirements["binary_role"], "hsaco")
        self.assertEqual(requirements["assembly_role"], "amdgcn")
        self.assertEqual(requirements["grid"], [1, 1, 1])
        self.assertEqual(requirements["signature"],
                         {"activation": "*fp32", "q8_workspace": "*u8"})
        self.assertFalse(result["evaluation"]["gpu_submitted"])
        self.assertFalse(result["scope"]["workload_claim_complete"])

    def test_live_fails_at_compiler_before_executor_host_or_runtime(self) -> None:
        compiler = SimpleNamespace(state="draft", check_corpus=Mock())
        executor = Mock()
        executor.reference = {
            "path": "runtime/executors/open-cake-ir-gfx1151-v1.json",
            "executor_id": "open-cake-ir-gfx1151-v1",
            "canonical_sha256": "a" * 64,
        }
        lowering = SimpleNamespace(
            toolchain_requirements=dict(quickstart._prepare(
                ROOT, ROOT / "compiler/revision.json"
            )[1].toolchain_requirements)
        )
        summary = {
            "kind": quickstart.RESULT_KIND,
            "compiler_revision": {
                "revision_id": "open-cake-ir-test-draft",
                "canonical_sha256": "b" * 64,
                "state": "draft",
            },
            "assessment": {"schedule_id": "packed-q8_1-producer-gfx1151-v1"},
            "workload": {"workload_id": "fixture"},
            "scope": {"kernel": "live_fp32_to_q8_1"},
        }
        with tempfile.TemporaryDirectory() as directory:
            evidence_root = Path(directory) / "attempt"
            with (
                patch.object(
                    quickstart,
                    "git_state",
                    side_effect=AssertionError("Git admission must not run"),
                ),
                patch.object(
                    quickstart,
                    "admit_exact_hip",
                    side_effect=AssertionError("runtime admission must not run"),
                ),
                self.assertRaisesRegex(ValueError, "released Compiler lock"),
            ):
                quickstart._run_live(
                    ROOT,
                    ROOT / "compiler/revision.json",
                    compiler,
                    executor,
                    lowering,
                    Mock(),
                    summary,
                    evidence_root,
                )

            failure = json.loads(
                (evidence_root / "failure.json").read_text(encoding="utf-8")
            )
            self.assertEqual(failure["failed_stage"], "compiler_admission")
            self.assertEqual(failure["failure_class"], "AUTHORITY_BLOCKED")
            self.assertFalse(failure["gpu_result_authorized"])
            compiler.check_corpus.assert_not_called()
            executor.admit_hip_host.assert_not_called()

    def test_released_compiler_must_be_the_lock_and_have_a_passing_gate(self) -> None:
        released = SimpleNamespace(state="released", check_corpus=Mock())
        with self.assertRaisesRegex(ValueError, "released Compiler lock"):
            quickstart._admit_released_compiler_lock(
                ROOT, ROOT / "compiler/revision.json", released
            )
        released.check_corpus.assert_not_called()

        released.check_corpus.return_value = SimpleNamespace(passed=False)
        with self.assertRaisesRegex(ValueError, "passing released Corpus Gate"):
            quickstart._admit_released_compiler_lock(
                ROOT,
                (ROOT / "compiler/revision.lock.json").resolve(),
                released,
            )

        gate = SimpleNamespace(passed=True)
        released.check_corpus.return_value = gate
        self.assertIs(
            quickstart._admit_released_compiler_lock(
                ROOT,
                (ROOT / "compiler/revision.lock.json").resolve(),
                released,
            ),
            gate,
        )

    def test_workspace_endpoints_reject_padding_bytes_and_input_mutation(self) -> None:
        expected = bytes(quickstart.Q8_1_WORKSPACE_BYTES)
        exact = quickstart._workspace_metrics(
            expected, expected, activation_unchanged=True
        )
        self.assertTrue(exact["passed"])
        self.assertTrue(exact["activation_unchanged"])
        self.assertTrue(exact["padding_records_zero"])
        self.assertTrue(exact["q8_workspace_byte_exact"])
        self.assertEqual(exact["q8_workspace_bytes_compared"], 576)

        changed = bytearray(expected)
        changed[-1] = 0xA5
        padding = quickstart._workspace_metrics(
            bytes(changed), expected, activation_unchanged=True
        )
        self.assertFalse(padding["passed"])
        self.assertFalse(padding["padding_records_zero"])
        self.assertFalse(padding["q8_workspace_byte_exact"])
        self.assertEqual(padding["q8_workspace_mismatch_count"], 1)

        mutated = quickstart._workspace_metrics(
            expected, expected, activation_unchanged=False
        )
        self.assertFalse(mutated["passed"])
        self.assertFalse(mutated["activation_unchanged"])
        self.assertTrue(mutated["q8_workspace_byte_exact"])

        truncated = quickstart._workspace_metrics(
            expected[:-1], expected, activation_unchanged=True
        )
        self.assertFalse(truncated["passed"])
        self.assertEqual(truncated["first_mismatch_byte"], 575)

    def test_all_seven_cases_retain_complete_observed_and_reference_bytes(self) -> None:
        workload = WorkloadContract.load(WORKLOAD)
        with tempfile.TemporaryDirectory() as directory:
            evidence_root = Path(directory)
            for case_id in workload.case_ids:
                material = materialize_q4_mmvq_case(workload, case_id)
                reference = q4_mmvq_reference(workload, material).q8_1_workspace
                records = quickstart._write_case_artifacts(
                    evidence_root,
                    case_id,
                    reference,
                    reference,
                )
                self.assertEqual(set(records), {"observed", "reference"})
                for record in records.values():
                    self.assertEqual(record["size_bytes"], 576)
                    self.assertEqual(
                        (evidence_root / record["path"]).stat().st_size, 576
                    )

            self.assertEqual(len(list(evidence_root.glob("*.bin"))), 14)
            with self.assertRaisesRegex(ValueError, "case id differs"):
                quickstart._write_case_artifacts(
                    evidence_root,
                    "../escape",
                    bytes(576),
                    bytes(576),
                )

    def test_executor_must_own_the_q8_runner_and_release_closure_includes_it(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not own the Q8_1 runner"):
            quickstart._admit_executor_contract(_executor(owns_runner=False))
        quickstart._admit_executor_contract(_executor(owns_runner=True))

        relative_sources = {
            path.relative_to(ROOT).as_posix() for path in _source_paths(ROOT)
        }
        self.assertIn(quickstart.RUNNER_SOURCE, relative_sources)

    def test_evidence_root_must_be_outside_the_checkout(self) -> None:
        blocked = ROOT / "q8-evidence-forbidden"
        self.assertFalse(blocked.exists())
        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--project-root",
                str(ROOT),
                "--prepare-only",
                "--evidence-root",
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
