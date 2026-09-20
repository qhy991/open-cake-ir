"""One ExecutionPlatform registry; the Evaluation and Lab layers stop re-partitioning targets.

Before this registry the facts a declared code object implies -- build roles, paired
assay kinds, protocol timing, job-id prefixes, host kinds, the worker's evaluate table --
were spelled in seven independent vocabularies, and CUDA was the fall-through in several.
These hold every consumer to the rows: each declared Target resolves to one, each
consumer's table has exactly the rows' keys, and an object no row implements is refused
by name from every entry point rather than reaching whichever branch followed.

No device is touched. The D6 probe mocks the nvidia-smi/torch observation and asserts
only what the admission records admit and refuse.
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from open_cake_ir.compiler.ir.vocabulary import LoweringBackend  # noqa: E402
from open_cake_ir.compiler.target import CodeObject, Target  # noqa: E402
from open_cake_ir.evaluation import admission, platforms  # noqa: E402
from open_cake_ir.evaluation.artifacts import allowed_artifact_roles  # noqa: E402
from open_cake_ir.evaluation.attempts import job_mode, valid_job_mode  # noqa: E402
from open_cake_ir.evaluation.core import EvaluationProtocol, _TIMINGS  # noqa: E402
from open_cake_ir.evaluation.cuda_driver import CudaDeviceAdmission  # noqa: E402
from open_cake_ir.evaluation.local_broker import LOCAL_KINDS  # noqa: E402
from open_cake_ir.evaluation.paired import PAIRED_KIND, admit_device_identity  # noqa: E402
from open_cake_ir.evaluation.platforms import (  # noqa: E402
    PLATFORMS, platform_for, platform_for_host_kind, platform_for_paired_kind,
)
from open_cake_ir.lab import executor as lab_executor  # noqa: E402
from open_cake_ir.lab.toolchains import TOOLCHAINS  # noqa: E402
from open_cake_ir.tasks import devices  # noqa: E402
from open_cake_ir.tasks import evaluate as worker  # noqa: E402

TARGET_DOCUMENTS = sorted((ROOT / "compiler/targets").glob("*.json"))
HOST_DOCUMENTS = sorted((ROOT / "runtime/hosts").glob("*.json"))


class EveryDeclaredTargetResolvesToARow(unittest.TestCase):
    def test_each_target_document_selects_the_row_its_code_object_names(self) -> None:
        self.assertTrue(TARGET_DOCUMENTS)
        for path in TARGET_DOCUMENTS:
            with self.subTest(target=path.stem):
                target = Target.load(path)
                row = platform_for(target)
                self.assertIs(row, PLATFORMS[target.code_object])
                # The same row whichever spelling the caller holds.
                self.assertIs(platform_for(path.stem), row)
                self.assertIs(platform_for(target.code_object), row)
                self.assertIs(platform_for(target.code_object.value), row)

    def test_every_row_is_a_code_object_the_compiler_declares(self) -> None:
        self.assertEqual(set(PLATFORMS), set(CodeObject))


class TheTaskSideEvaluateTableCoversEveryRow(unittest.TestCase):
    def test_the_worker_registers_an_evaluate_on_every_row_and_no_more(self) -> None:
        self.assertEqual(set(worker._PLATFORMS), set(PLATFORMS))
        for code_object, entry in worker._PLATFORMS.items():
            with self.subTest(code_object=code_object.value):
                self.assertIsNotNone(entry.evaluate)
                self.assertIs(entry.platform, PLATFORMS[code_object])
                # Attribution facts are read off the row, never restated beside it.
                self.assertEqual(entry.attribution, PLATFORMS[code_object].attribution)
                self.assertEqual(entry.profiled_child, PLATFORMS[code_object].profiled_child)


class ProtocolTimingIsTheRowsProtocolTiming(unittest.TestCase):
    def test_every_timing_but_none_is_declared_by_a_row(self) -> None:
        declared = {row.protocol_timing for row in PLATFORMS.values()
                    if row.protocol_timing is not None}
        self.assertEqual(_TIMINGS - {"none"}, declared)
        # The HIP row's assay could not be declared before the registry.
        self.assertIn("paired_hip", declared)

    def test_a_protocol_admits_each_declared_timing_and_refuses_an_undeclared_one(self) -> None:
        def protocol(timing: str) -> EvaluationProtocol:
            return EvaluationProtocol("p", "confirmatory", "a" * 64, "case", timing)

        for row in PLATFORMS.values():
            if row.protocol_timing is not None:
                with self.subTest(timing=row.protocol_timing):
                    self.assertEqual(protocol(row.protocol_timing).timing, row.protocol_timing)
        protocol("none")
        with self.assertRaisesRegex(ValueError, "EvaluationProtocol differs"):
            protocol("paired_spirv")


class TheExecutorHostTableIsTheRegistrysHostKinds(unittest.TestCase):
    def test_host_validator_keys_equal_the_registry_host_kinds(self) -> None:
        self.assertEqual(lab_executor.declared_host_kinds(),
                         {row.host_kind for row in PLATFORMS.values()})
        self.assertEqual(set(lab_executor._HOST_VALIDATORS), lab_executor.declared_host_kinds())
        # D10: the pre-kind CUDA form is an explicit None row, not a fall-through.
        self.assertIn(None, lab_executor._HOST_VALIDATORS)
        self.assertIs(platform_for_host_kind(None), PLATFORMS[CodeObject.CUBIN])

    def test_the_committed_host_captures_validate_through_their_own_rows(self) -> None:
        self.assertEqual([path.name for path in HOST_DOCUMENTS], [
            "apple_gpu_family7.json", "apple_gpu_family8.json", "apple_gpu_family9.json",
            "gfx1151.json", "gfx938.json", "sm_103a.json", "xcore1002.json"])
        for path in HOST_DOCUMENTS:
            with self.subTest(host=path.stem):
                document = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(document["target"], path.stem)
                host = document["host_environment"]
                kind = host.get("kind")
                # The capture's own kind selects the row, and it is the row the target's
                # code object declares.
                row = lab_executor._host_kind(host)
                self.assertIs(row, lab_executor._HOST_VALIDATORS[kind])
                self.assertEqual(platform_for(path.stem).host_kind, kind)
                lab_executor.ExecutorRevision._validate_host_document(host)
                captured = row.captured_target(host)
                self.assertIn(captured, (None, path.stem))


class JobIdsFollowTheAllocatorThatIssuedThem(unittest.TestCase):
    def test_valid_job_mode_maps_each_prefix_to_its_allocators_mode(self) -> None:
        for prefix, mode in (("gpuq", "exclusive"), ("metal", "local_serialized"),
                             ("hip", "local_serialized"), ("cuda", "local_serialized")):
            with self.subTest(prefix=prefix):
                self.assertTrue(valid_job_mode(f"{prefix}-0123456789ab", mode))
                self.assertEqual(job_mode(f"{prefix}-0123456789ab"), mode)
                other = "local_serialized" if mode == "exclusive" else "exclusive"
                self.assertFalse(valid_job_mode(f"{prefix}-0123456789ab", other))

    def test_an_unknown_prefix_is_refused(self) -> None:
        for job in ("spirv-0123456789ab", "gpuq_0123456789ab", "cuda-01234", "", None):
            with self.subTest(job=job):
                self.assertFalse(valid_job_mode(job, "exclusive"))
                self.assertFalse(valid_job_mode(job, "local_serialized"))
                self.assertIsNone(job_mode(job))

    def test_the_prefixes_are_the_rows_prefixes(self) -> None:
        exclusive = {row.exclusive_job_prefix for row in PLATFORMS.values()} - {None}
        local = {row.local_job_prefix for row in PLATFORMS.values()} - {None}
        self.assertEqual(exclusive, {"gpuq"})
        self.assertEqual(local, {"cuda", "metal", "hip", "maca"})
        self.assertEqual(set(LOCAL_KINDS), local)


class AnObjectNoRowImplementsIsRefusedByName(unittest.TestCase):
    """The enum is not touched: the double removes the hsaco row and names a fourth object.

    A fourth code object cannot exist without a Compiler declaration, so the two shapes
    the registry can meet are exercised: a declared object whose row is absent, and a
    name no declaration carries.
    """

    def _without_hsaco(self):
        rows = {key: row for key, row in PLATFORMS.items() if key is not CodeObject.HSACO}
        return mock.patch.object(platforms, "PLATFORMS", rows)

    def test_platform_for_refuses_a_declared_object_with_no_row_by_name(self) -> None:
        with self._without_hsaco():
            with self.assertRaisesRegex(ValueError, "no execution platform implements the 'hsaco' code object"):
                platform_for(CodeObject.HSACO)
            with self.assertRaisesRegex(ValueError, "no execution platform implements the 'hsaco' code object"):
                platform_for("gfx938")

    def test_platform_for_refuses_a_name_no_declaration_carries(self) -> None:
        with self.assertRaisesRegex(ValueError, "'spirv' names no declared target and no code object"):
            platform_for("spirv")
        with self.assertRaisesRegex(ValueError, "Executor host kind 'opencl' is not admitted"):
            platform_for_host_kind("opencl")
        with self.assertRaisesRegex(ValueError, "declared by no execution platform"):
            platform_for_paired_kind("fixed_baseline_paired_spirv_v1")

    def test_allowed_artifact_roles_refuses_in_the_evaluation_layers_words(self) -> None:
        with self._without_hsaco():
            with self.assertRaisesRegex(ValueError, "has no executable role in this Evaluation layer"):
                allowed_artifact_roles("gfx938")
        with self.assertRaisesRegex(ValueError, "has no executable role in this Evaluation layer"):
            allowed_artifact_roles("spirv")

    def test_the_worker_refuses_an_object_its_table_does_not_implement(self) -> None:
        authority = SimpleNamespace(candidate=SimpleNamespace(target="gfx938"),
                                    manifest=SimpleNamespace(), timed_assay_available=False)
        with mock.patch.dict(worker._PLATFORMS, {CodeObject.HSACO: worker._ExecutionPlatform(evaluate=None)}):
            with self.assertRaisesRegex(ValueError, "no execution platform implements 'hsaco'"):
                worker._platform(authority)
        rows = {key: row for key, row in worker._PLATFORMS.items() if key is not CodeObject.HSACO}
        with mock.patch.object(worker, "_PLATFORMS", rows):
            with self.assertRaisesRegex(ValueError, "no execution platform implements 'hsaco'"):
                worker._platform(authority)

    def test_the_host_validator_lookup_refuses_an_unknown_kind_by_name(self) -> None:
        with self.assertRaisesRegex(ValueError, "Executor host kind 'opencl' is not admitted"):
            lab_executor._host_kind({"kind": "opencl"})
        with self.assertRaisesRegex(ValueError, "Executor host kind 'opencl' is not admitted"):
            lab_executor.ExecutorRevision._validate_host_document({"kind": "opencl"})


class TheLabToolchainTableIsKeyedByLoweringBackend(unittest.TestCase):
    def test_toolchain_keys_equal_the_backend_enum(self) -> None:
        self.assertEqual(set(TOOLCHAINS), set(LoweringBackend))
        for backend, row in TOOLCHAINS.items():
            self.assertIs(row.backend, backend)


class LocalCudaAllocation(unittest.TestCase):
    """D6: a CUDA device outside the cluster allocator, reached through the local broker.

    A synthetic devices row declares `local_broker` for a cubin target. The worker then
    selects the local admission, which admits a `cuda-` job in local_serialized mode for
    a correctness check; the paired CUPTI policy keeps refusing that receipt.
    """

    ROW = {"triton-local-b200": {
        "target": "sm_100a", "device_name": "NVIDIA B200", "provenance_token": "local",
        "route": "triton", "allocation": "local_broker", "tanh_contract": None,
        "timing_source": "cupti", "power_of_two_width": False}}

    def _admission(self, target, visible, job_id, mode):
        # The nvidia-smi compute-apps and torch device checks, stood in for; the record
        # they would have produced still passes through CudaDeviceAdmission's own rules.
        self.observed.append((target.target_id, visible, job_id, mode))
        return CudaDeviceAdmission("NVIDIA B200", (10, 0), "GPU-fixture", job_id, mode)

    def setUp(self) -> None:
        self.observed = []

    def test_the_worker_selects_the_local_admission_from_the_devices_row(self) -> None:
        with mock.patch.object(devices, "BACKENDS", self.ROW):
            mode = devices.allocation_mode("sm_100a")
        self.assertEqual(mode, "local_serialized")
        authority = SimpleNamespace(candidate=SimpleNamespace(target="sm_100a"),
                                    allocation_mode=mode)
        environment = {"CUDA_VISIBLE_DEVICES": "0", "METAL_JOB_ID": "cuda-0123456789ab",
                       "METAL_BROKER_LOCK_FD": "7"}
        with mock.patch.dict(os.environ, environment, clear=False), \
             mock.patch.dict(os.environ, {}, clear=False) as env, \
             mock.patch("open_cake_ir.evaluation.local_broker.observe_local_job",
                        return_value="cuda-0123456789ab") as job, \
             mock.patch.object(admission, "_observe_cuda_device", side_effect=self._admission), \
             mock.patch.object(worker, "observe_exclusive_cuda") as exclusive:
            env.pop("GPUQ_JOB_ID", None)
            record = worker._observe_cuda(authority)
        exclusive.assert_not_called()
        job.assert_called_once_with("cuda")
        self.assertEqual(self.observed, [("sm_100a", "0", "cuda-0123456789ab", "local_serialized")])
        self.assertEqual((record.broker_job_id, record.mode), ("cuda-0123456789ab", "local_serialized"))

    def test_the_cluster_row_still_selects_the_exclusive_admission(self) -> None:
        self.assertEqual(devices.allocation_mode("sm_100a"), "exclusive")
        authority = SimpleNamespace(candidate=SimpleNamespace(target="sm_100a"),
                                    allocation_mode="exclusive")
        with mock.patch.object(worker, "observe_exclusive_cuda", return_value="lease") as exclusive, \
             mock.patch.object(worker, "observe_local_cuda") as local:
            self.assertEqual(worker._observe_cuda(authority), "lease")
        exclusive.assert_called_once_with("sm_100a")
        local.assert_not_called()
        with self.assertRaisesRegex(ValueError, "allocation mode None names no CUDA device admission"):
            worker._observe_cuda(SimpleNamespace(candidate=SimpleNamespace(target="sm_100a"),
                                                 allocation_mode=None))

    def test_the_local_admission_refuses_under_a_cluster_lease(self) -> None:
        with mock.patch.dict(os.environ, {"GPUQ_JOB_ID": "gpuq-0123456789ab"}):
            with self.assertRaisesRegex(ValueError, "gpu_admission_differs"):
                admission.observe_local_cuda("sm_100a")

    def test_cuda_device_admission_admits_each_allocators_job_in_its_own_mode(self) -> None:
        CudaDeviceAdmission("NVIDIA B200", (10, 0), "GPU-fixture", "cuda-0123456789ab", "local_serialized")
        CudaDeviceAdmission("NVIDIA B200", (10, 0), "GPU-fixture", "gpuq-0123456789ab", "exclusive")
        for job, mode in (("cuda-0123456789ab", "exclusive"), ("gpuq-0123456789ab", "local_serialized"),
                          ("hip-0123456789ab", "local_serialized"), ("metal-0123456789ab", "local_serialized"),
                          ("cuda-000000000000", "local_serialized"), (None, "local_serialized")):
            with self.subTest(job=job, mode=mode):
                with self.assertRaisesRegex(ValueError, "CUDA device admission differs"):
                    CudaDeviceAdmission("NVIDIA B200", (10, 0), "GPU-fixture", job, mode)

    def test_paired_cupti_refuses_the_local_receipt(self) -> None:
        participants = {"candidate": {"target": "sm_100a"}, "baseline": {"target": "sm_100a"}}
        launch = {"gpu_uuid": "GPU-fixture"}
        admit_device_identity({"kind": PAIRED_KIND, "job_id": "gpuq-0123456789ab",
                               "gpu_uuid": "GPU-fixture"}, launch, participants)
        with self.assertRaisesRegex(ValueError, "paired CUDA device/host identity differs"):
            admit_device_identity({"kind": PAIRED_KIND, "job_id": "cuda-0123456789ab",
                                   "gpu_uuid": "GPU-fixture"}, launch, participants)


if __name__ == "__main__":
    unittest.main()
