from __future__ import annotations

import copy
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
from open_cake_ir.tasks.workloads import load_workload  # noqa: E402
from open_cake_ir.compiler import Compiler  # noqa: E402
from open_cake_ir.evaluation import (  # noqa: E402
    WorkloadContract,
)
from open_cake_ir.tasks.amd.llama_q4_mmvq import (  # noqa: E402
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

    def test_lowering_rejects_target_artifact_and_producer_abi_drift(self) -> None:
        compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")
        lowering = compiler.lower(compiler.assess_file(ROOT / quickstart.SCHEDULE))
        original = dict(lowering.toolchain_requirements)
        quickstart._validate_lowering(ROOT, original)
        for label, mutate in {
            "arch": lambda r: r["triton_target"].__setitem__("arch", "gfx942"),
            "wave": lambda r: r["triton_target"].__setitem__("warp_size", True),
            "binary": lambda r: r.__setitem__("binary_role", "cubin"),
            "assembly": lambda r: r.pop("assembly_role"),
            "store_domain": lambda r: r["grid"].__setitem__(0, 2),
            "input_type": lambda r: r["signature"].__setitem__("activation", "*fp16"),
        }.items():
            with self.subTest(drift=label):
                requirements = copy.deepcopy(original)
                mutate(requirements)
                with self.assertRaises(ValueError):
                    quickstart._validate_lowering(ROOT, requirements)

    def test_matching_q8_bytes_still_reject_input_byte_or_storage_mutation(self) -> None:
        from tests.contracts.test_amd_rmsnorm_search import _FakeTensor, _FakeTorch
        from open_cake_ir.compiler.target import Target
        workload = load_workload(WORKLOAD)
        target = Target.load(ROOT / "compiler/targets/gfx1151.json")
        requirements = {**copy.deepcopy(quickstart._PRODUCER_ABI),
            "target": target.target_id, "triton_target": dict(target.triton_target)}
        lowering = SimpleNamespace(toolchain_requirements=requirements, source="fixture")
        expected = [q4_mmvq_reference(workload, materialize_q4_mmvq_case(workload, case)).q8_1_workspace
                    for case in workload.case_ids]
        for mutation in ("none", "signed_zero", "storage"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                class Tensor(_FakeTensor):
                    def detach(self):
                        return self

                torch = SimpleNamespace(
                    version=SimpleNamespace(hip="fixture"), float32=_FakeTorch.float32,
                    uint8=_FakeTorch.uint8, cuda=_FakeTorch.cuda, equal=_FakeTorch.equal,
                    tensor=lambda values, **kwargs: Tensor(values[0], 10),
                    full=lambda shape, *args, **kwargs: Tensor(0.0, 30, shape=shape),
                )
                properties = SimpleNamespace(name="fixture", gcnArchName="gfx1151",
                    warp_size=32, multi_processor_count=1, total_memory=1024)
                compiled = SimpleNamespace(
                    metadata=SimpleNamespace(name="fixture", shared=0),
                    asm={role: (b"\x7fELFfixture" if role == "hsaco" else b"fixture")
                        for role in ("source", "ttir", "ttgir", "llir", "amdgcn", "hsaco")},
                )

                def launch(activation, output, **kwargs):
                    if mutation == "signed_zero" and activation.value == 0.0:
                        activation.value = -0.0
                    elif mutation == "storage":
                        activation.pointer += 1
                    return compiled

                executor = _executor(owns_runner=True)
                executor.reference = {}
                executor.admit_hip_host = lambda: SimpleNamespace(
                    torch_hip_version="fixture", device_monitor={}, profilers=())
                summary = {}
                with (
                    patch.object(quickstart, "_admit_released_compiler_lock"),
                    patch.object(quickstart, "git_state", return_value={"tree_clean": True}),
                    patch.object(quickstart, "admit_exact_hip", return_value=(torch, object(), properties)),
                    patch.object(quickstart, "load_generated_module", return_value=(
                        SimpleNamespace(**{requirements["kernel_entry_point"]: SimpleNamespace(run=launch)}),
                        SimpleNamespace(cleanup=lambda: None))),
                    patch.object(quickstart, "_tensor_bytes", side_effect=expected),
                    patch.object(quickstart.importlib.metadata, "version", return_value="fixture"),
                ):
                    code = quickstart._run_live_impl(
                        ROOT, ROOT / "compiler/revision.lock.json", object(), executor,
                        lowering, workload, summary, {}, Path(directory),
                    )
                self.assertEqual(code, 0 if mutation == "none" else 2)
                records = summary["evaluation"]["cases"]
                self.assertEqual(len(records), 7)
                self.assertTrue(all(record["q8_workspace_byte_exact"] for record in records))
                self.assertEqual(all(record["activation_unchanged"] for record in records), mutation == "none")
                self.assertFalse(summary["evaluation"]["workload_claim_complete"])

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
        workload = load_workload(WORKLOAD)
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
