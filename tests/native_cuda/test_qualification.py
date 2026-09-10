"""CPU contract probes. Synthetic records here are never GPU evidence."""
from __future__ import annotations

import copy
from dataclasses import replace
import csv
import io
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess
import sys
import struct
import tempfile
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import native_cuda_contract as c
import native_cuda_evaluate as e
from open_cake_ir.compiler import Compiler
from open_cake_ir.evaluation.profiler import NCU_ATTRIBUTION_METRICS, build_ncu_attribution_profile
from open_cake_ir.evaluation.timing import summarize_cohort


def observed_cupti_phases(function, *, dry_run_iters, repeat_iters, cold_l2_cache, use_cuda_graph,
                          trace=None, omit_last=False):
    """CPU callback simulation of the inspected FlashInfer 0.6.12 no-graph path.

    This is independent of timing_protocol's allocation budget: one initial call,
    five event-estimation calls, requested dry runs, then requested CUPTI samples.
    Actual helper control-flow slicing is also retained with the repair evidence.
    """
    assert cold_l2_cache is True and use_cuda_graph is False
    def call(phase):
        if trace is not None:
            trace.append(phase)
        function()
    call("initial")
    for _ in range(5):
        call("estimate")
    for _ in range(dry_run_iters):
        call("dry_run")
    for _ in range(repeat_iters - int(omit_last)):
        call("cupti_sample")
    return [1.0] * repeat_iters


class PublicContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")

    def test_every_frozen_case_is_required(self):
        rows = c.cases()
        self.assertEqual([r["case_id"] for r in rows if r["family"] == "gemm"],
                         list(c.workloads()["gemm"].case_ids))
        self.assertEqual([r["case_id"] for r in rows if r["family"] == "kmeans"],
                         list(c.workloads()["kmeans"].case_ids))
        self.assertEqual(len(rows), 12)

    def test_native_specializations_preserve_exact_case_abi(self):
        for row in c.cases():
            with self.subTest(row=row["id"]):
                lowering = self.compiler.lower(self.compiler.assess(c.specialize(row)))
                c.validate_metadata(dict(lowering.toolchain_requirements), target=lowering.target,
                                    abi=c.owner(row).tensor_abi(row["case_id"]))
                self.assertEqual(lowering.target, "sm_103a")

    def test_every_existing_baseline_specialization_lowers(self):
        for row in c.cases():
            if row["family"] == "two_mma":
                continue
            with self.subTest(row=row["id"]):
                assessment = self.compiler.assess(c.baseline_schedule(row))
                baseline = self.compiler.lower(assessment)
                self.assertEqual(baseline.target, "sm_103a")
                self.assertEqual(baseline.route.backend.value, "triton")

    def test_metadata_rejects_input_target_and_launch_drift(self):
        row = c.cases()[0]
        lowered = self.compiler.lower(self.compiler.assess(c.specialize(row)))
        original = dict(lowered.toolchain_requirements)
        mutations = [lambda m: m["nvcc_flags"].append("--gpu-code=sm_103"),
                     lambda m: m.update(target="sm_100a"),
                     lambda m: m["arguments"][0].update(shape=[512, 264]),
                     lambda m: m["argument_order"].reverse(),
                     lambda m: m.update(host_abi={"launch": "guess"}),
                     lambda m: m.update(dynamic_shared_bytes=-1)]
        for mutation in mutations:
            metadata = copy.deepcopy(original)
            mutation(metadata)
            with self.assertRaises(c.QualificationError):
                c.validate_metadata(metadata, target="sm_103a", abi=c.owner(row).tensor_abi(row["case_id"]))

    def test_compile_plan_keeps_exact_a_pair_and_host_links(self):
        row = c.cases()[0]
        lowered = self.compiler.lower(self.compiler.assess(c.specialize(row)))
        commands = c.compile_commands("/cuda/nvcc", Path("."), dict(lowered.toolchain_requirements))
        for command in commands.values():
            self.assertIn("--gpu-architecture=compute_103a", command)
            self.assertIn("--gpu-code=sm_103a", command)
            self.assertNotIn("-arch=sm_103", command)
        self.assertIn("-lcuda", commands["shared"])
        self.assertIn("-lcudart", commands["shared"])

    def test_draft_never_authorizes_prepare(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "uncreated"
            with patch.object(e.Compiler, "load", return_value=self.compiler):
                with self.assertRaises(c.QualificationError):
                    e.prepare(root)
            self.assertFalse(root.exists())

    def test_two_mma_requires_both_contractions_numerically(self):
        row = {"family": "two_mma", "case_id": "tiny"}
        workload = c.workloads()["gemm"]
        inputs = c.materialize_case(workload, "tiny")
        doubled = c.gemm_reference(row, workload, inputs)["c"]
        single = c.reference_outputs(workload, "tiny", inputs)["c"]
        self.assertTrue(c.compare_raw(row, workload, doubled, doubled)["passed"])
        self.assertFalse(c.compare_raw(row, workload, single, doubled)["passed"])

    def test_gemm_uses_full_elementwise_tolerance_and_rejects_nan(self):
        row = {"family": "gemm", "case_id": "tiny"}
        workload = c.workloads()["gemm"]
        expected = [0.0] * 6
        self.assertTrue(c.compare_raw(row, workload, [0.0009] * 6, expected)["passed"])
        for bad in (0.0011, float("nan"), float("inf")):
            output = expected.copy()
            output[-1] = bad
            self.assertFalse(c.compare_raw(row, workload, output, expected)["passed"])

    def test_kmeans_exact_ids_overrule_tie_distance_diagnostics(self):
        row = {"family": "kmeans", "case_id": "duplicate_tie"}
        workload = c.workloads()["kmeans"]
        expected = [0, 1, 2, 0]
        for output in ([0, 3, 2, 0], [0, 1, 2, -1], [0, 1, 2, 4], [0.0, 1, 2, 0]):
            self.assertFalse(c.compare_raw(row, workload, output, expected)["passed"])

    def test_raw_paths_refuse_escape_symlinks_and_partial_elements(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "raw").write_bytes(b"x")
            (root / "link").symlink_to(root / "raw")
            for path in ("../raw", "/tmp/raw", "link"):
                with self.assertRaises(c.QualificationError):
                    c.child(root, path)
            with self.assertRaises(c.QualificationError):
                c.read_vector(root / "raw", "fp32")

    def test_artifact_writes_are_create_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw"
            c.write_bytes(path, b"first")
            with self.assertRaises(FileExistsError):
                c.write_bytes(path, b"replacement")
            self.assertEqual(path.read_bytes(), b"first")

    def test_process_failures_keep_actual_failure_class_and_stderr(self):
        record = e.capture(["/nonexistent/native-qualification-executable"], cwd=ROOT)
        self.assertIsNone(record["returncode"])
        self.assertEqual(record["error"], "process_unavailable")
        self.assertTrue(record["stderr"])


class CompileAndHostBoundary(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.row = next(row for row in c.cases() if row["id"] == "gemm-tiny")
        compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")
        lower = compiler.lower(compiler.assess(c.specialize(self.row)))
        self.row.update(target=lower.target, source_sha256=lower.source_sha256, metadata=dict(lower.toolchain_requirements))
        self.directory = self.root / self.row["id"]
        self.directory.mkdir()
        shared = b"\x7fELFunit-test-not-a-GPU-binary"
        (self.directory / "kernel.so").write_bytes(shared)
        (self.directory / "kernel.cubin").write_bytes(shared)
        (self.directory / "kernel.ptx").write_text(".target sm_103a\n")
        self.record = {"status": "compiled", "source_sha256": lower.source_sha256, "target": lower.target,
            "nvcc_version": {"argv": ["/unit-test/nvcc", "--version"], "returncode": 0, "stdout": "release 13.1"},
            "commands": {role: {"argv": argv, "returncode": 0} for role, argv in c.compile_commands("/unit-test/nvcc", Path("."), self.row["metadata"]).items()},
            "shared_object_sha256": sha256(shared).hexdigest(),
            "resources_raw": {"argv": ["/unit-test/cuobjdump", "--dump-resource-usage", "kernel.cubin"], "returncode": 0,
                "stdout": "Function " + self.row["metadata"]["kernel_entry_point"] + ":\n REG:32 STACK:0 SHARED:0 LOCAL:0\n"},
            "inspector_version": {"argv": ["/unit-test/cuobjdump", "--version"], "returncode": 0, "stdout": "unit-test-version"},
            "compiled_resources": e.CompiledResources(source_sha256=lower.source_sha256,
                cubin_sha256=sha256(shared).hexdigest(), target=lower.target,
                entry_point=self.row["metadata"]["kernel_entry_point"],
                threads_per_cta=self.row["metadata"]["threads_per_cta"],
                dynamic_shared_bytes=self.row["metadata"]["dynamic_shared_bytes"],
                compiler_version="release 13.1", inspector_version="unit-test-version",
                registers_per_thread=32, stack_bytes=0, static_shared_bytes=0, local_bytes=0).as_dict()}
        c.write_json(self.directory / "compile.json", self.record)

    def test_compile_receipt_checks_source_flags_raw_resources_and_loaded_bytes(self):
        self.assertEqual(e.verify_compile(self.root, self.row)["status"], "compiled")
        for mutate in (lambda r: r.update(source_sha256="d" * 64),
                       lambda r: r["commands"].pop("ptx"),
                       lambda r: r["commands"]["shared"]["argv"].append("-arch=sm_103"),
                       lambda r: r["compiled_resources"].update(registers_per_thread=64),
                       lambda r: r.update(shared_object_sha256="d" * 64),
                       lambda r: r.update(status="passed")):
            record = copy.deepcopy(self.record)
            mutate(record)
            (self.directory / "compile.json").write_text(json.dumps(record))
            with self.assertRaises(c.QualificationError):
                e.verify_compile(self.root, self.row)

    def test_inspected_cubin_swap_cannot_reuse_old_resource_receipt(self):
        (self.directory / "kernel.cubin").write_bytes(b"\x7fELFdifferent-unit-test-object")
        with self.assertRaisesRegex(c.QualificationError, "CUBIN differs from resource handoff"):
            e.verify_compile(self.root, self.row)

    def test_host_shape_rejection_happens_before_library_loading(self):
        class Tensor:
            shape = (1,)
            dtype = "torch.bfloat16"
            device = "cuda:0"
            is_cuda = True
            def is_contiguous(self): return True
            def data_ptr(self): return 4096
        with patch.object(e.ctypes, "CDLL") as library:
            with self.assertRaises(ValueError):
                e.NativeHandle(self.directory / "kernel.so", self.row["metadata"], [Tensor()] * 4)
            library.assert_not_called()

    def test_native_host_abi_creates_once_launches_and_destroys_once(self):
        tensors = []
        for i, argument in enumerate(self.row["metadata"]["arguments"]):
            tensors.append(SimpleNamespace(shape=tuple(argument["shape"]),
                dtype="torch." + {"bf16": "bfloat16", "fp32": "float32"}[argument["dtype"]],
                device="cuda:0", is_cuda=True, is_contiguous=lambda: True,
                data_ptr=lambda i=i: 4096 * (i + 1)))
        def create(pointers, result):
            self.assertEqual(list(pointers), [4096, 8192, 12288, 16384])
            result._obj.value = 1234
            return 0
        functions = {"create": Mock(side_effect=create), "launch": Mock(return_value=0), "destroy": Mock(return_value=0)}
        library = SimpleNamespace(**{self.row["metadata"]["host_abi"][name]: function for name, function in functions.items()})
        with patch.object(e.ctypes, "CDLL", return_value=library):
            handle = e.NativeHandle(self.directory / "kernel.so", self.row["metadata"], tensors)
            handle.launch(7)
            self.assertEqual(handle.calls, 1)
            functions["launch"].return_value = 700
            with self.assertRaises(c.QualificationError) as failure:
                handle.launch(7)
            self.assertEqual(failure.exception.category, "launch")
            handle.close()
            handle.close()
        functions["create"].assert_called_once()
        functions["destroy"].assert_called_once()


class ReceiptReplay(unittest.TestCase):
    """One mocked authority isolates raw-receipt parsing from release/GPU access."""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        base = next(row for row in c.cases() if row["id"] == "gemm-tiny")
        self.row = {**base, "target": "sm_103a", "source_sha256": "a" * 64,
                    "workload_sha256": c.owner(base).canonical_sha256,
                    "metadata": {"target_device_names": ["NVIDIA B300 SXM6 AC"],
                                 "kernel_entry_point": "native_kernel", "arguments": []},
                    "baseline": {"source_sha256": "b" * 64,
                                 "metadata": {"kernel_entry_point": "baseline_kernel"}}}
        self.directory = self.root / base["id"] / "run"
        self.directory.mkdir(parents=True)
        self.workload = c.owner(base)
        self.inputs = c.materialize_case(self.workload, "tiny")
        self.expected = c.gemm_reference(base, self.workload, self.inputs)["c"]
        for abi in self.workload.tensor_abi("tiny"):
            self.row["metadata"]["arguments"].append({"name": abi.name, "mode": abi.mode,
                "shape": list(abi.shape), "dtype": abi.dtype})
            if abi.mode == "input":
                raw = c.vector_bytes(self.inputs[abi.name], "fp32")
                if abi.dtype == "bf16":
                    raw = b"".join(raw[i + 2:i + 4] for i in range(0, len(raw), 4))
                (self.directory / (abi.name + ".bin")).write_bytes(raw)
        (self.directory / "oracle.bin").write_bytes(c.vector_bytes(self.expected, "fp32"))
        self.admission = {"target": "sm_103a", "compute_capability": [10, 3], "mode": "exclusive",
            "device_name": "NVIDIA B300 SXM6 AC", "gpu_uuid": "unit-test-only",
            "broker_job_id": "gpuq-unit-test", "hostname": "unit-test-no-gpu",
            "uid": 1010, "euid": 1010, "pid": 100, "visible_device": "7",
            "profile_environment": {name: None for name in e.PROFILE_ENVIRONMENT},
            "profile_python": "/unit-test/python", "profile_script": "/unit-test/native_cuda_evaluate.py",
            "evidence_root": str(self.root)}
        self.admission["profile_environment"].update(CUDA_VISIBLE_DEVICES="7", GPUQ_JOB_ID="gpuq-unit-test")
        c.write_json(self.root / "admission.json", self.admission)
        c.write_json(self.directory / "inputs.json", {"workload_sha256": self.workload.canonical_sha256,
            "case": self.workload.case("tiny"), "oracle": self.workload.document["oracle"],
            "two_mma_composition": False, "preparation": {"scope": "none", "wall_seconds": 0.0}})
        self.result = {"status": "observed", "failure_class": None,
            **{key: self.row[key] for key in ("target", "source_sha256", "workload_sha256")},
            "case_id": self.row["id"], "gpu_uuid": "unit-test-only", "broker_job_id": "gpuq-unit-test"}
        for arm in ("candidate", "baseline"):
            for phase in ("preflight", "postflight"):
                self.result[phase + "-" + arm] = self.check(phase + "-" + arm)
        c.write_json(self.directory / "result.json", self.result)
        measurements = []
        for i, order in enumerate(c.timing_protocol().pair_order):
            pair = {"pair_index": i, "order": list(order), "arms": {}}
            for position, arm in enumerate(order):
                pair["arms"][arm] = {"position": position, "samples_ms": [1.0] * 25,
                    "summary": summarize_cohort([1.0] * 25), "route_calls": c.timing_protocol().route_calls_per_cohort,
                    "source_sha256": self.row["source_sha256"] if arm == "candidate" else self.row["baseline"]["source_sha256"],
                    "checks": [self.check(f"timed-{i}-{arm}-{j}") for j in range(c.timing_protocol().route_calls_per_cohort)]}
                record = pair["arms"][arm]
                record.update(status="valid", samples_receipt=f"cohort-{i}-{arm}-samples.json")
                c.write_json(self.directory / record["samples_receipt"], {key: record[key] for key in (
                    "position", "source_sha256", "samples_ms", "route_calls")})
                c.write_json(self.directory / f"cohort-{i}-{arm}.json", record)
            measurements.append(pair)
            c.write_json(self.directory / f"pair-{i}.json", pair)
        self.timing = {"method": "cupti", "cold_l2_cache": True, "use_cuda_graph": False,
            "baseline": "existing_triton_algorithm", "scope": "prepared_inputs_core_launch",
            "measurements": measurements, "summary": c.timing_summary(measurements)}
        c.write_json(self.directory / "timing.json", self.timing)
        for arm, identity, kernel in (("candidate", "a" * 64, "native_kernel"), ("baseline", "b" * 64, "baseline_kernel")):
            raw = io.StringIO()
            writer = csv.writer(raw)
            writer.writerow(["Kernel Name", "Metric Name", "Metric Unit", "Metric Value"])
            for metric in NCU_ATTRIBUTION_METRICS:
                writer.writerow([kernel, metric, "unit", "1"])
            stdout = raw.getvalue()
            payload = build_ncu_attribution_profile(candidate_sha256=identity, case_id=base["id"],
                kernel_name=kernel, ncu_version="unit-test", ncu_executable_sha256="c" * 64,
                stdout=stdout.encode(), stderr=b"")
            (self.directory / ("profile-" + arm + ".json")).write_bytes(payload)
            version = {"argv": ["/unit-test/ncu", "--version"], "returncode": 0, "stdout": "unit-test", "stderr": ""}
            request = {"case_id": base["id"], "arm": arm, "source_sha256": identity,
                "evaluation_pid": 100, "sudo": False, "expected": e.profile_expected(self.admission, False),
                "ncu": "/unit-test/ncu", "ncu_executable_sha256": "c" * 64, "version_probe": version,
                "python": self.admission["profile_python"], "script": self.admission["profile_script"],
                "evidence_root": str(self.root)}
            c.write_json(self.directory / ("profile-" + arm + "-version.json"), version)
            c.write_json(self.directory / ("profile-" + arm + "-request.json"), request)
            c.write_json(self.directory / ("profile-" + arm + "-command.json"), {"returncode": 0, "stdout": stdout, "stderr": "",
                "argv": e.profile_command(request, self.row)})
            c.write_json(self.directory / ("profile-" + arm + "-output.json"), self.check("profile-" + arm + "-output"))
            # Synthetic CPU fixture: observed metadata does not claim a real UID transition.
            c.write_json(self.directory / ("profile-" + arm + "-execution.json"), {
                **request["expected"], "pid": 200, "parent_pid": 199, "status": "observed",
                "output_access": e.profile_output_access(self.directory, "profile-" + arm)})

    def check(self, label):
        (self.directory / (label + ".bin")).write_bytes(c.vector_bytes(self.expected, "fp32"))
        return {**c.compare_raw(self.row, self.workload, self.expected, self.expected),
                "inputs_unchanged": True, "output": label + ".bin"}

    def rewrite(self, name, value):
        (self.directory / name).write_text(json.dumps(value))

    def verify(self):
        with patch.object(e, "load_manifest", return_value={"target": "sm_103a", "rows": [self.row]}), \
             patch.object(e, "verify_compile", return_value={}):
            return e.verify_all(self.root)

    def test_complete_raw_replay_with_mocked_release_and_compile_authority(self):
        self.assertTrue(self.verify()["qualified"])

    def test_passed_status_alone_and_compile_only_are_not_qualification(self):
        self.rewrite("result.json", {"status": "passed"})
        with self.assertRaises(c.QualificationError):
            self.verify()
        (self.directory / "result.json").unlink()
        with self.assertRaises(FileNotFoundError):
            self.verify()

    def test_missing_profiler_timing_or_correctness_never_qualifies(self):
        for name in ("profile-candidate.json", "timing.json", "preflight-baseline.bin"):
            path = self.directory / name
            content = path.read_bytes()
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                self.verify()
            path.write_bytes(content)

    def test_wrong_target_or_source_binding_refused(self):
        for key, value in (("target", "sm_100a"), ("source_sha256", "d" * 64), ("case_id", "gemm-primary")):
            broken = dict(self.result, **{key: value})
            self.rewrite("result.json", broken)
            with self.assertRaises(c.QualificationError):
                self.verify()

    def test_forged_timing_summary_and_missing_samples_refused(self):
        for mutate in (lambda t: t["summary"].update(speedup=10.0),
                       lambda t: t["measurements"][0]["arms"]["candidate"]["samples_ms"].pop(),
                       lambda t: t.update(use_cuda_graph=True),
                       lambda t: t.update(baseline="same_source_native_control")):
            timing = copy.deepcopy(self.timing)
            mutate(timing)
            self.rewrite("timing.json", timing)
            with self.assertRaises(ValueError):
                self.verify()

    def test_correctness_replayed_from_raw_output_not_passed_boolean(self):
        output = self.expected.copy()
        output[-1] += 1.0
        (self.directory / "preflight-candidate.bin").write_bytes(c.vector_bytes(output, "fp32"))
        with self.assertRaises(c.QualificationError):
            self.verify()

    def test_input_and_oracle_byte_drift_refused(self):
        for name in ("a.bin", "oracle.bin"):
            path = self.directory / name
            original = path.read_bytes()
            path.write_bytes(bytes([original[0] ^ 128]) + original[1:])
            with self.assertRaises(c.QualificationError):
                self.verify()
            path.write_bytes(original)

    def test_ncu_raw_projection_tampering_refused(self):
        path = self.directory / "profile-candidate.json"
        document = json.loads(path.read_bytes())
        document["summary"]["occupancy"]["registers_per_thread"] = 100.0
        path.write_bytes(json.dumps(document, sort_keys=True, separators=(",", ":")).encode())
        with self.assertRaises(ValueError):
            self.verify()

    def test_reused_timed_output_receipt_refused(self):
        timing = copy.deepcopy(self.timing)
        first = timing["measurements"][0]["arms"]["candidate"]["checks"]
        first[1] = dict(first[0])
        self.rewrite("timing.json", timing)
        self.rewrite("pair-0.json", timing["measurements"][0])
        self.rewrite("cohort-0-candidate.json", timing["measurements"][0]["arms"]["candidate"])
        with self.assertRaisesRegex(c.QualificationError, "fresh output receipt was reused"):
            self.verify()

    def test_profile_command_cannot_name_another_case(self):
        path = self.directory / "profile-candidate-command.json"
        record = json.loads(path.read_text())
        record["argv"][record["argv"].index("--case") + 1] = "gemm-primary"
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(c.QualificationError, "NCU command binding differs"):
            self.verify()

    def test_missing_evidence_cli_is_nonzero_and_no_skip(self):
        completed = subprocess.run([sys.executable, str(ROOT / "tools/native_cuda_evaluate.py"),
            "verify", "--evidence-root", str(self.root / "absent")], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 3)
        self.assertFalse(json.loads(completed.stderr)["qualified"])


class FailedTimingRetention(unittest.TestCase):
    def test_returned_samples_survive_invalid_or_partial_output_and_teardown(self):
        for failure_kind in ("invalid", "partial", "baseline", "teardown", "count", "sample_count",
                             "interrupt_check", "interrupt_teardown", "invalid_interrupt_teardown"):
            with self.subTest(failure=failure_kind), tempfile.TemporaryDirectory(prefix="TEST_ONLY_native_timing_") as temp:
                directory = Path(temp)
                fake_torch = SimpleNamespace(cuda=SimpleNamespace(
                    current_stream=lambda: SimpleNamespace(cuda_stream=7), synchronize=lambda: None))
                closed = []
                calls = []
                interruption = KeyboardInterrupt("test requested interruption")
                interrupts_cleanup = failure_kind in ("interrupt_teardown", "invalid_interrupt_teardown")
                class Handle:
                    def __init__(self, arguments): self.arguments = arguments
                    def launch(self, stream): calls.append(self.arguments)
                    def close(self):
                        closed.append(self.arguments)
                        if failure_kind == "teardown": raise RuntimeError("test teardown failed")
                        if interrupts_cleanup: raise interruption
                def benchmark(function, **kwargs):
                    samples = observed_cupti_phases(function, **kwargs, omit_last=failure_kind == "count")
                    return samples[:-1] if failure_kind == "sample_count" else samples
                checks = []
                def check(row, directory, label, *_args, **_kwargs):
                    checks.append(label)
                    if failure_kind == "partial" and len(checks) == 2:
                        raise OSError("test output snapshot failed")
                    if failure_kind == "interrupt_check" and len(checks) == 2:
                        raise interruption
                    return {"passed": (failure_kind == "baseline" and "candidate" in label)
                            or failure_kind in ("teardown", "interrupt_teardown"), "output": label + ".bin"}
                row = {"source_sha256": "a" * 64, "baseline": {"source_sha256": "b" * 64}}
                with patch.dict(sys.modules, {"torch": fake_torch,
                        "flashinfer": SimpleNamespace(testing=object()), "flashinfer.testing": object()}), \
                     patch.object(e, "StrictCuptiBenchmark", return_value=benchmark), \
                     patch.object(e, "fresh_arguments", side_effect=lambda *_args: []), \
                     patch.object(e, "handle", side_effect=lambda _r, _d, args, _a: Handle(args)), \
                     patch.object(e, "unchanged_inputs", return_value=True), \
                     patch.object(e, "retain_check", side_effect=check):
                    with self.assertRaises(KeyboardInterrupt if "interrupt" in failure_kind else Exception) as caught:
                        e.measure(row, directory, {}, [])
                if "interrupt" in failure_kind:
                    self.assertIs(caught.exception, interruption)
                self.assertEqual(1 if interrupts_cleanup else len(calls) + (1 if failure_kind == "count" else 0), len(closed))
                arm = "baseline" if failure_kind == "baseline" else "candidate"
                raw = c.read_json(directory / f"cohort-0-{arm}-samples.json")
                outcome = c.read_json(directory / f"cohort-0-{arm}.json")
                self.assertEqual(raw["samples_ms"], [1.0] * (24 if failure_kind == "sample_count" else 25))
                self.assertEqual(raw["route_calls"], 41 if failure_kind == "count" else 42)
                self.assertIn(outcome["status"], ("invalid", "unknown"))
                self.assertEqual(len(outcome["checks"]), 1 if failure_kind in ("partial", "interrupt_check")
                                 else (0 if failure_kind in ("count", "sample_count") else 42))
                if interrupts_cleanup:
                    self.assertEqual(outcome["teardown_interrupt"], {"type": "KeyboardInterrupt",
                        "message": str(interruption), "unattempted_handles": 41})
                    self.assertEqual(outcome["status"], "invalid" if failure_kind == "invalid_interrupt_teardown" else "unknown")
                    if failure_kind == "invalid_interrupt_teardown":
                        self.assertIsInstance(caught.exception.__cause__, c.QualificationError)
                self.assertFalse((directory / "pair-0.json").exists())
                if failure_kind == "baseline":
                    self.assertEqual(c.read_json(directory / "cohort-0-candidate.json")["status"], "valid")


class CuptiInvocationContract(unittest.TestCase):
    def test_all_helper_phases_get_fresh_checked_outputs_and_old_budget_is_rejected(self):
        for variant in ("matched", "old_36", "extra_call", "missing_call"):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory(prefix="TEST_ONLY_cupti_phases_") as temp:
                directory = Path(temp)
                cohorts, launched, checked, handles, arguments = [], [], [], [], []
                settings = []
                def benchmark(function, **kwargs):
                    trace = []
                    cohorts.append(trace)
                    settings.append(kwargs)
                    samples = observed_cupti_phases(function, **kwargs, trace=trace, omit_last=variant == "missing_call")
                    if variant == "extra_call":
                        function()
                    return samples
                helper = SimpleNamespace(bench_gpu_time_with_cupti=benchmark,
                    bench_gpu_time_with_cuda_event=lambda *_a, **_k: self.fail("event fallback"),
                    bench_gpu_time_with_cudagraph=lambda *_a, **_k: self.fail("graph fallback"))
                fake_torch = SimpleNamespace(cuda=SimpleNamespace(
                    current_stream=lambda: SimpleNamespace(cuda_stream=7), synchronize=lambda: None))
                class Handle:
                    def __init__(self, args): self.arguments, self.closed = args, False
                    def launch(self, stream): launched.append(self.arguments)
                    def close(self): self.closed = True
                def fresh(*_args):
                    value = object()
                    arguments.append(value)
                    return value
                def load(_row, _directory, args, _arm):
                    value = Handle(args)
                    handles.append(value)
                    return value
                def check(_row, _directory, label, args, *_a, **_k):
                    checked.append(args)
                    return {"passed": True, "output": label + ".bin"}
                protocol = c.timing_protocol()
                if variant == "old_36":
                    protocol = replace(protocol, route_calls_per_cohort=36)
                row = {"source_sha256": "a" * 64, "baseline": {"source_sha256": "b" * 64}}
                with patch.dict(sys.modules, {"torch": fake_torch,
                        "flashinfer": SimpleNamespace(testing=helper), "flashinfer.testing": helper}), \
                     patch.object(e, "timing_protocol", return_value=protocol), \
                     patch.object(e, "fresh_arguments", side_effect=fresh), \
                     patch.object(e, "handle", side_effect=load), \
                     patch.object(e, "unchanged_inputs", return_value=True), \
                     patch.object(e, "retain_check", side_effect=check):
                    # Use the real StrictCuptiBenchmark wrapper, including its
                    # warning/fallback controls, around independent phase simulation.
                    if variant == "matched":
                        result = e.measure(row, directory, {}, [])
                    else:
                        with self.assertRaises(c.QualificationError) as failure:
                            e.measure(row, directory, {}, [])
                        self.assertEqual(failure.exception.category, "timing")
                self.assertTrue(all(handle.closed for handle in handles))
                self.assertTrue(all(value == {"dry_run_iters": 11, "repeat_iters": 25,
                    "cold_l2_cache": True, "use_cuda_graph": False} for value in settings))
                if variant == "matched":
                    self.assertEqual(cohorts, [["initial"] + ["estimate"] * 5 + ["dry_run"] * 11 + ["cupti_sample"] * 25] * 8)
                    self.assertEqual(len(launched), 8 * 42)
                    self.assertEqual(checked, launched)
                    self.assertEqual(len({id(arg) for arg in launched}), len(launched))
                    self.assertEqual(len(result["measurements"]), 4)
                    for pair in result["measurements"]:
                        for arm in pair["arms"].values():
                            self.assertEqual(len(arm["samples_ms"]), 25)
                            self.assertEqual(arm["route_calls"], 42)
                            self.assertEqual(len(arm["checks"]), 42)
                else:
                    self.assertFalse((directory / "pair-0.json").exists())
                    outcome = c.read_json(directory / "cohort-0-candidate.json")
                    self.assertEqual(outcome["status"], "unknown")
                    self.assertEqual(outcome["route_calls"], {"old_36": 36, "extra_call": 42, "missing_call": 41}[variant])
                    if variant == "old_36":
                        self.assertEqual(len(cohorts[0]), 37)
                        self.assertIn("invocation budget exceeded", outcome["error"])


class KMeansInputReplay(unittest.TestCase):
    """Real byte/domain checks; task oracle results are explicit CPU interface doubles.

    These tests exercise the handoff to the unchanged task owner, not an invented
    numerical qualification. Actual Torch replay is a separately required runtime
    check; missing Torch must fail closed and is covered below without a skip.
    """
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="TEST_ONLY_kmeans_replay_")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.row = next(row for row in c.cases() if row["id"] == "kmeans-tail_nk")
        self.workload = c.owner(self.row)
        for tensor in self.workload.tensor_abi("tail_nk"):
            if tensor.mode == "input":
                (self.directory / (tensor.name + ".bin")).write_bytes(
                    b"\0" * (math.prod(tensor.shape) * {"bf16": 2, "fp32": 4}[tensor.dtype]))
        self.input_receipt = {"materialization_device": "cuda:0", "oracle_device": "cuda:0"}
        self.bind_inputs()

    def bind_inputs(self):
        identities = {}
        for name in ("tokens", "centroids", "centroid_sq"):
            raw = (self.directory / (name + ".bin")).read_bytes()
            identities[name] = {"sha256": sha256(raw).hexdigest(), "size_bytes": len(raw)}
        self.input_receipt.update(input_handoff={name: identities[name] for name in ("tokens", "centroids")},
            preparation_handoff={"input": "centroids", "operation": "fp32_square_sum", "output": identities["centroid_sq"]})

    def test_nonfinite_bf16_and_invalid_prepared_norms_are_rejected_without_torch(self):
        for filename, corruption in (("tokens.bin", b"\xc0\x7f"), ("centroids.bin", b"\x80\xff"),
                                     ("centroid_sq.bin", struct.pack("<f", -1.0))):
            with self.subTest(filename=filename):
                path = self.directory / filename
                original = path.read_bytes()
                path.write_bytes(corruption + original[len(corruption):])
                with patch.dict(sys.modules, {"torch": None}):
                    with self.assertRaises(c.QualificationError) as failure:
                        e.replay_kmeans_inputs(self.row, self.directory, [0] * 257, self.input_receipt)
                self.assertEqual(failure.exception.category, "input_contract")
                path.write_bytes(original)

    def test_missing_cpu_oracle_runtime_never_means_qualified(self):
        with patch.dict(sys.modules, {"torch": None}):
            with self.assertRaises(c.QualificationError) as failure:
                e.replay_kmeans_inputs(self.row, self.directory, [0] * 257, self.input_receipt)
        self.assertEqual(failure.exception.category, "cpu_oracle_unavailable")

    def test_actual_input_handoff_and_task_oracle_are_both_mandatory(self):
        class Tensor:
            def __init__(self, values, dtype="bf16", shape=None):
                self.values, self.dtype, self.shape = list(values), dtype, shape
            def reshape(self, shape):
                self.shape = shape
                return self
            def to(self, dtype): return Tensor(self.values, dtype, self.shape)
            def contiguous(self): return self
            def tolist(self): return list(self.values)
            def view(self, dtype):
                if dtype != "uint8": raise AssertionError("test interface only supports byte views")
                raw = c.vector_bytes(self.values, "fp32")
                return b"".join(raw[i + 2:i + 4] for i in range(0, len(raw), 4)) if self.dtype == "bf16" else raw
            def __mul__(self, other):
                return Tensor([a * b for a, b in zip(self.values, other.values)], self.dtype, self.shape)
            def sum(self, dim, dtype):
                self_test.assertEqual(dim, -1)
                width = self.shape[-1]
                return Tensor([sum(self.values[i:i + width]) for i in range(0, len(self.values), width)], dtype, self.shape[:-1])
        self_test = self
        def frombuffer(raw, dtype):
            payload = bytes(raw)
            if dtype == "bf16":
                payload = b"".join(b"\0\0" + payload[i:i + 2] for i in range(0, len(payload), 2))
            return Tensor([value for value, in struct.iter_unpack("<f", payload)], dtype)
        fake_torch = SimpleNamespace(frombuffer=frombuffer, bfloat16="bf16", float32="fp32",
                                     uint8="uint8", equal=lambda a, b: a == b)
        oracle = Tensor([0] * 257, "int32", (1, 257))
        owner_module = "open_cake_ir.tasks.flash_kmeans.workload"
        with patch.dict(sys.modules, {"torch": fake_torch}), \
             patch(owner_module + ".flash_kmeans_oracle", return_value=oracle) as reference:
            e.replay_kmeans_inputs(self.row, self.directory, [0] * 257, self.input_receipt)
            self.assertEqual(reference.call_args.kwargs, {"case_id": "tail_nk"})
            self.assertEqual(reference.call_args.args[1].values, [0.0] * (257 * 128))
            # Same-sized oracle and every output could be replaced with ones;
            # unchanged external owner still returns the lowest-index zero.
            with self.assertRaisesRegex(c.QualificationError, "external CPU replay"):
                e.replay_kmeans_inputs(self.row, self.directory, [1] * 257, self.input_receipt)
            path = self.directory / "tokens.bin"
            original = path.read_bytes()
            path.write_bytes(b"\x80\x3f" + original[2:])
            with self.assertRaisesRegex(c.QualificationError, "handoff bytes differ"):
                e.replay_kmeans_inputs(self.row, self.directory, [0] * 257, self.input_receipt)
            path.write_bytes(original)
            path = self.directory / "centroid_sq.bin"
            path.write_bytes(c.vector_bytes([1.0] + [0.0] * 256, "fp32"))
            self.bind_inputs()
            with self.assertRaisesRegex(c.QualificationError, "square/sum numerical consistency"):
                e.replay_kmeans_inputs(self.row, self.directory, [0] * 257, self.input_receipt)


class RealCpuOracleReplay(unittest.TestCase):
    """Actual task-owned PyTorch CPU math, with no GPU/admission/release fixture.

    These tests require the qualification CPU dependencies explicitly; missing
    PyTorch is an environment failure, never a skip or numerical success.
    CPU-created test inputs exercise replay of retained bytes; they do not claim
    to reproduce the real CUDA RNG or to be a GPU evidence bundle.
    """
    def setUp(self):
        import torch
        self.torch = torch
        self.temp = tempfile.TemporaryDirectory(prefix="TEST_ONLY_real_cpu_oracle_")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.row = next(row for row in c.cases() if row["id"] == "kmeans-tail_nk")
        self.workload = c.owner(self.row)
        from open_cake_ir.tasks.flash_kmeans.workload import generate_flash_kmeans_case, flash_kmeans_oracle
        self.tokens, self.centroids = generate_flash_kmeans_case(self.workload, "tail_nk", device="cpu")
        self.reference = flash_kmeans_oracle
        self.expected = flash_kmeans_oracle(self.workload, self.tokens, self.centroids, case_id="tail_nk").reshape(-1).tolist()
        for name in ("init", "current_device", "synchronize"):
            guard = patch.object(torch.cuda, name, side_effect=AssertionError("CPU replay attempted CUDA access"))
            guard.start()
            self.addCleanup(guard.stop)
        self.write_inputs()

    def write_inputs(self, *, wrong_norm=False):
        torch = self.torch
        fp32 = self.centroids.to(torch.float32)
        norms = (fp32 * fp32).sum(dim=-1, dtype=torch.float32).contiguous()
        if wrong_norm:
            norms[0, 0] += 1.0
        identities = {}
        for name, tensor in (("tokens", self.tokens), ("centroids", self.centroids), ("centroid_sq", norms)):
            raw = tensor.contiguous().view(torch.uint8).numpy().tobytes()
            (self.directory / (name + ".bin")).write_bytes(raw)
            identities[name] = {"sha256": sha256(raw).hexdigest(), "size_bytes": len(raw)}
        self.receipt = {"materialization_device": "cuda:0", "oracle_device": "cuda:0",
            "input_handoff": {name: identities[name] for name in ("tokens", "centroids")},
            "preparation_handoff": {"input": "centroids", "operation": "fp32_square_sum", "output": identities["centroid_sq"]}}
        (self.directory / "oracle.bin").write_bytes(c.vector_bytes(self.expected, "int32"))

    def test_actual_task_cpu_reference_accepts_matching_retained_inputs(self):
        e.replay_kmeans_inputs(self.row, self.directory, self.expected, self.receipt)

    def test_same_size_nan_cannot_hide_behind_updated_handoff(self):
        self.tokens[0, 0, 0] = float("nan")
        self.write_inputs()
        with self.assertRaisesRegex(c.QualificationError, "nonfinite BF16"):
            e.replay_kmeans_inputs(self.row, self.directory, self.expected, self.receipt)

    def test_zero_scores_require_lowest_index_even_if_all_claimed_ids_agree(self):
        self.tokens.zero_()
        self.centroids.zero_()
        self.expected = [1] * 257
        self.write_inputs()
        preserved = (self.directory / "oracle.bin").read_bytes()
        with self.assertRaises(c.QualificationError) as failure:
            e.replay_kmeans_inputs(self.row, self.directory, self.expected, self.receipt)
        self.assertEqual(failure.exception.category, "oracle_replay_mismatch")
        self.assertEqual((self.directory / "oracle.bin").read_bytes(), preserved)
        actual = self.reference(self.workload, self.tokens, self.centroids, case_id="tail_nk").reshape(-1).tolist()
        self.assertEqual(actual, [0] * 257)
        e.replay_kmeans_inputs(self.row, self.directory, actual, self.receipt)

    def test_wrong_preparation_with_consistent_handoff_is_independently_rejected(self):
        self.write_inputs(wrong_norm=True)
        with self.assertRaises(c.QualificationError) as failure:
            e.replay_kmeans_inputs(self.row, self.directory, self.expected, self.receipt)
        self.assertEqual(failure.exception.category, "preparation_replay_mismatch")

    def test_frozen_constructed_duplicate_tie_uses_real_lowest_index_oracle(self):
        from open_cake_ir.tasks.flash_kmeans.workload import generate_flash_kmeans_case
        self.row = next(row for row in c.cases() if row["id"] == "kmeans-duplicate_tie")
        self.tokens, self.centroids = generate_flash_kmeans_case(self.workload, "duplicate_tie", device="cpu")
        self.expected = self.reference(self.workload, self.tokens, self.centroids, case_id="duplicate_tie").reshape(-1).tolist()
        self.assertEqual(self.expected, [0, 1, 2, 0])
        self.write_inputs()
        e.replay_kmeans_inputs(self.row, self.directory, self.expected, self.receipt)
        with self.assertRaises(c.QualificationError) as failure:
            e.replay_kmeans_inputs(self.row, self.directory, [0, 3, 2, 0], self.receipt)
        self.assertEqual(failure.exception.category, "oracle_replay_mismatch")


if __name__ == "__main__":
    unittest.main()
