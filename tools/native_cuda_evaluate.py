#!/usr/bin/env python3
"""Compile and qualify native CUDA on an already broker-allocated B300.

Usage (all outputs are create-only, outside the checkout)::

  python tools/native_cuda_evaluate.py prepare --evidence-root /external/run
  python tools/native_cuda_evaluate.py compile --evidence-root /external/run
  gpu-run ... --mode exclusive --gpu-count 1 -- python tools/native_cuda_evaluate.py run --evidence-root /external/run
  python tools/native_cuda_evaluate.py verify --evidence-root /external/run

The broker owns allocation. This tool never sets CUDA_VISIBLE_DEVICES or submits
a job. ``prepare`` requires a released Compiler; compilation is CPU-only.
``run`` retains external-oracle output, every fresh timed output, raw CUPTI
cohorts, and NCU CSV. Existing Triton algorithms supply the paired baselines.
This engineering comparison does not constitute an authoring-environment Study.

中文：先在已审批的 Compiler 上生成完整用例，再离线编译。run 必须由已有
broker 独占分配运行；失败收据保留原样。verify 缺少任何层都返回非零。
KMeans 的 centroid_sq 准备成本单列，计时范围是四参数 core launch。

``--ncu`` names a real NCU ELF, not a shell or sudo wrapper. ``run --ncu-sudo``
explicitly elevates NCU and its synchronous profile target Python, with a fixed
task environment allowlist and privileged GNU timeout. The evaluation parent,
input generation and CUPTI remain ordinary-user processes. This path does not
support daemonizing or session-escaping profile targets.

CPU replay requires PyTorch (no CUDA device). KMeans keeps its existing CUDA
generation, oracle and centroid preparation. Replay checks the actual retained
inputs, their generation/preparation handoff, and the external oracle on CPU;
Workload-pinned vectors retain their frozen identities. CPU/GPU oracle or
preparation-consistency failures leave qualification unknown; GPU answers are
preserved. Preparation uses an exact dyadic FP32 error certificate within its
checked arithmetic domain, not bitwise reconstruction of the CUDA reduction tree.
"""
from __future__ import annotations

import argparse
import ctypes
from dataclasses import asdict
from hashlib import sha256
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import signal
import struct
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from native_cuda_contract import (
    ROOT, QualificationError, require, read_json, write_json, write_bytes, child, cases, owner,
    specialize, baseline_schedule, validate_metadata, compile_commands, timing_protocol,
    timing_summary, gemm_reference, compare_raw, vector_bytes, read_vector,
    finite_bf16, CUPTI_DRY_RUN_ITERS,
)
from open_cake_ir.compiler import Compiler
from open_cake_ir.compiler.performance.compiled_resources import CompiledResources
from open_cake_ir.compiler.toolchain import _parse_cuobjdump_resources
from open_cake_ir.evaluation.benchmark import StrictCuptiBenchmark
from open_cake_ir.evaluation.timing import summarize_cohort
from open_cake_ir.evaluation.profiler import (
    NCU_ATTRIBUTION_METRICS, build_ncu_attribution_profile, load_ncu_attribution_profile,
)
from open_cake_ir.tasks.tiles.workload import materialize_case


DEFAULT_NCU = "/usr/local/cuda-13.1/nsight-compute-2025.4.1/target/linux-desktop-glibc_2_11_3-x64/ncu"
PROFILE_ENVIRONMENT = ("CUDA_VISIBLE_DEVICES", "GPUQ_JOB_ID", "TMPDIR", "TRITON_CACHE_DIR",
                       "CUDA_CACHE_PATH", "PYTHONDONTWRITEBYTECODE")
PROFILE_TIMEOUT = 900
PROFILE_KILL_AFTER = 5
PROFILE_CAPTURE_TIMEOUT = PROFILE_TIMEOUT + PROFILE_KILL_AFTER + 25
PROFILE_CLEANUP_TIMEOUT = 15


def profile_environment():
    return {name: os.environ.get(name) for name in PROFILE_ENVIRONMENT}


def profile_context():
    return {"uid": os.getuid(), "euid": os.geteuid(), "pid": os.getpid(),
            "parent_pid": os.getppid(), "environment": profile_environment()}


def profile_expected(admission, sudo):
    require(type(sudo) is bool, "profile privilege request differs", "profiler")
    require(type(admission.get("uid")) is int and admission["uid"] > 0
            and admission.get("euid") == admission["uid"]
            and type(admission.get("pid")) is int and admission["pid"] > 0,
            "ordinary evaluation user identity missing", "admission")
    environment = admission.get("profile_environment", {})
    require(set(environment) == set(PROFILE_ENVIRONMENT)
            and environment["CUDA_VISIBLE_DEVICES"] == admission.get("visible_device")
            and environment["GPUQ_JOB_ID"] == admission.get("broker_job_id")
            and all(value is None or isinstance(value, str) for value in environment.values()),
            "profile parent environment differs", "admission")
    return {"uid": 0 if sudo else admission["uid"], "euid": 0 if sudo else admission["euid"],
            "environment": environment, "gpu": {"gpu_uuid": admission["gpu_uuid"],
                "device_name": admission["device_name"], "compute_capability": admission["compute_capability"]}}


def profile_command(request, row):
    source = row if request["arm"] == "candidate" else row["baseline"]
    prefix = []
    if request["sudo"]:
        tmpdir = request["expected"]["environment"]["TMPDIR"]
        require(isinstance(tmpdir, str) and Path(tmpdir).is_absolute(), "sudo TMPDIR declaration missing", "profiler")
        # TMPDIR does not survive the observed sudo preserve-env path.
        # Pass that one declared value after privilege transition; the child
        # still observes and validates its actual environment without repair.
        prefix = ["/usr/bin/sudo", "-n", "--preserve-env=" + ",".join(
            name for name in PROFILE_ENVIRONMENT if name != "TMPDIR"), "--",
            "/usr/bin/env", "TMPDIR=" + tmpdir]
    # The canonical reader consumes named metric rows; the observed raw page
    # instead puts metrics in columns. Request the required display explicitly.
    return [*prefix, "/usr/bin/timeout", "--signal=TERM", f"--kill-after={PROFILE_KILL_AFTER}s",
        f"{PROFILE_TIMEOUT}s", request["ncu"], "--forward-signals", "--csv", "--page", "details",
        "--print-details", "all", "--print-metric-name", "name", "--print-units", "base",
        "--target-processes", "all", "--kernel-name-base", "function", "--kernel-name",
        "regex:^" + source["metadata"]["kernel_entry_point"] + "$", "--launch-count", "1",
        "--metrics", ",".join(NCU_ATTRIBUTION_METRICS), request["python"], request["script"],
        "_profile", "--evidence-root", request["evidence_root"], "--case", row["id"], "--arm", request["arm"]]


def validate_profile_request(request, admission, row, arm):
    source = row if arm == "candidate" else row["baseline"]
    require(request.get("case_id") == row["id"] and request.get("arm") == arm
            and request.get("source_sha256") == source["source_sha256"]
            and request.get("evaluation_pid") == admission.get("pid")
            and request.get("evidence_root") == admission.get("evidence_root")
            and request.get("python") == admission.get("profile_python")
            and request.get("script") == admission.get("profile_script")
            and request.get("expected") == profile_expected(admission, request.get("sudo")),
            "profile request binding differs", "profiler")
    require(all(isinstance(request.get(name), str) and Path(request[name]).is_absolute()
                for name in ("ncu", "python", "script", "evidence_root")),
            "profile executable or evidence path differs", "profiler")
    version = request.get("version_probe", {})
    require(version.get("argv") == [request["ncu"], "--version"] and version.get("returncode") == 0
            and isinstance(version.get("stdout"), str) and version["stdout"]
            and "error" not in version, "real NCU version probe differs", "profiler")


def validate_profile_context(request, observed):
    expected = request["expected"]
    require(all(type(observed.get(key)) is int and type(expected.get(key)) is int for key in ("uid", "euid"))
            and all(observed.get(key) == expected[key] for key in ("uid", "euid", "environment"))
            and type(observed.get("pid")) is int and observed["pid"] > 0
            and observed["pid"] != request["evaluation_pid"]
            and type(observed.get("parent_pid")) is int and observed["parent_pid"] > 0,
            "profile child identity or inherited allocation differs", "admission")


def validate_profile_execution(request, execution):
    validate_profile_context(request, execution)
    require(execution.get("gpu") == request["expected"]["gpu"]
            and execution.get("status") == "observed", "profile child GPU or outcome differs", "profiler")
    outputs = execution.get("output_access", {})
    label = "profile-" + request["arm"]
    require(set(outputs) == {label + "-output.bin", label + "-output.json"}
            and all(isinstance(value, dict) and set(value) == {"uid", "gid", "mode", "size_bytes"}
                    and all(type(item) is int and item >= 0 for item in value.values())
                    and value["mode"] <= 0o777 and value["size_bytes"] > 0 for value in outputs.values()),
            "profile output ownership/access observation missing", "profiler")


def profile_output_access(directory, label):
    result = {}
    for suffix in ("-output.bin", "-output.json"):
        path = child(directory, label + suffix)
        stat = path.stat()
        result[path.name] = {"uid": stat.st_uid, "gid": stat.st_gid,
                             "mode": stat.st_mode & 0o777, "size_bytes": stat.st_size}
    return result


def capture_profile(argv, *, cwd, receipt):
    """Bound this synchronous NCU invocation without killing its sudo relay first.

    GNU timeout owns the inner process group and TERM/KILL deadline. A caught
    interruption targets only our direct relay PID, then drains/reaps boundedly.
    Reaping that PID is not a general claim that escaped descendants are gone;
    an unfinished drain is retained as cleanup unknown and cannot qualify.
    """
    start = time.time()
    process = None
    stdout = stderr = b""
    record = {"argv": list(argv), "started_at_unix": start, "returncode": None}
    failure = None
    previous_handlers = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)}
    termination_signal = None
    defer_termination = True
    def cancellation():
        return KeyboardInterrupt() if termination_signal == signal.SIGINT else SystemExit(128 + termination_signal)
    def terminate(signum, frame):
        nonlocal termination_signal
        if termination_signal is None:
            termination_signal = signum
        if not defer_termination:
            raise cancellation()
    # Broker cancellation targets the evaluation parent's group. The separate
    # launcher session will not receive it unless this parent forwards it.
    for signum in previous_handlers:
        signal.signal(signum, terminate)
    try:
        process = subprocess.Popen(argv, cwd=cwd, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        # Do not throw from inside Popen after it has spawned but before its
        # handle is assigned here. Once owned, a pending INT/TERM is forwarded.
        defer_termination = False
        if termination_signal is not None:
            raise cancellation()
        stdout, stderr = process.communicate(timeout=PROFILE_CAPTURE_TIMEOUT)
        record["returncode"] = process.returncode
    except BaseException as error:
        # Keep the first INT/TERM pending during bounded cleanup, including when a
        # different exception initiated cleanup. Never discard it with SIG_IGN.
        defer_termination = True
        failure = error
        record.update(error="timeout" if isinstance(error, subprocess.TimeoutExpired) else type(error).__name__,
                      error_message=str(error))
        if termination_signal is not None:
            record["termination_signal"] = termination_signal
        if isinstance(error, subprocess.TimeoutExpired):
            stdout, stderr = error.output or b"", error.stderr or b""
        if process is not None:
            try:
                if process.poll() is None:
                    process.send_signal(signal.SIGTERM)
                    record["cleanup_signal"] = "TERM_to_direct_launcher"
                stdout, stderr = process.communicate(timeout=PROFILE_CLEANUP_TIMEOUT)
                record["returncode"] = process.returncode
                record["cleanup"] = "launcher_reaped_and_pipes_closed"
            except BaseException as cleanup:
                record["cleanup"] = "unknown"
                record["cleanup_error"] = {"type": type(cleanup).__name__, "message": str(cleanup)}
                if isinstance(cleanup, subprocess.TimeoutExpired):
                    stdout, stderr = cleanup.output or stdout, cleanup.stderr or stderr
    finally:
        defer_termination = True
        try:
            if failure is None and termination_signal is not None:
                failure = cancellation()
                record.update(error=type(failure).__name__, error_message=str(failure), termination_signal=termination_signal)
            record.update(stdout=stdout.decode(errors="replace"), stderr=stderr.decode(errors="replace"),
                          elapsed_seconds=time.time() - start)
            write_json(receipt, record)
        finally:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
    # A first INT/TERM can arrive while an already-completed child's receipt is
    # being written. Preserve those bytes, then cancel instead of returning it.
    if failure is None and termination_signal is not None:
        failure = cancellation()
    if failure is not None:
        raise failure
    return record


def external(path):
    path = Path(path).resolve()
    require(path != ROOT and ROOT not in path.parents, "evidence must be outside checkout")
    return path


def capture(argv, *, cwd, timeout=600):
    start = time.time()
    try:
        process = subprocess.run(argv, cwd=cwd, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=False)
        return {"argv": list(map(str, argv)), "returncode": process.returncode,
                "stdout": process.stdout.decode(errors="replace"),
                "stderr": process.stderr.decode(errors="replace"),
                "started_at_unix": start, "elapsed_seconds": time.time() - start}
    except subprocess.TimeoutExpired as error:
        return {"argv": list(map(str, argv)), "returncode": None,
                "stdout": (error.stdout or b"").decode(errors="replace"),
                "stderr": (error.stderr or b"").decode(errors="replace"),
                "started_at_unix": start, "elapsed_seconds": time.time() - start,
                "error": "timeout"}
    except OSError as error:
        return {"argv": list(map(str, argv)), "returncode": None, "stdout": "", "stderr": str(error),
                "started_at_unix": start, "elapsed_seconds": time.time() - start,
                "error": "process_unavailable"}


def revision():
    compiler = Compiler.load(ROOT, ROOT / "compiler/revision.lock.json")
    require(compiler.state == "released", "a released Compiler is required", "release")
    return compiler


def lower_rows(compiler):
    result = []
    for row in cases():
        document = specialize(row)
        assessment = compiler.assess(document)
        require(assessment.lowering_eligible,
                f"{row['id']} is not lowerable: " + "; ".join(
                    f"{f.code} at {f.path}" for f in assessment.findings if f.blocks_lowering), "compiler")
        lowering = compiler.lower(assessment)
        metadata = dict(lowering.toolchain_requirements)
        validate_metadata(metadata, target=lowering.target, abi=owner(row).tensor_abi(row["case_id"]))
        require(lowering.generated and lowering.route.backend.value == "native_cuda", "native generated lowering required")
        record = {**row, "target": lowering.target,
                        "workload_sha256": owner(row).canonical_sha256,
                        "source_sha256": lowering.source_sha256,
                        "compiler_revision_id": lowering.compiler_revision_id,
                        "compiler_revision_sha256": lowering.compiler_revision_sha256,
                        "schedule_sha256": lowering.schedule_sha256,
                        "metadata": metadata}
        if row["family"] != "two_mma":
            baseline_document = baseline_schedule(row)
            baseline = compiler.lower(compiler.assess(baseline_document))
            require(baseline.target == lowering.target and baseline.route.backend.value == "triton",
                    "baseline target or route differs", "target")
            record["baseline"] = {"source_sha256": baseline.source_sha256,
                "schedule_sha256": baseline.schedule_sha256,
                "metadata": dict(baseline.toolchain_requirements)}
        result.append((record, document, lowering.source))
    return result


def prepare(root):
    compiler = revision()
    rows = lower_rows(compiler)
    commit = capture(["git", "rev-parse", "HEAD"], cwd=ROOT)
    require(commit["returncode"] == 0, "source commit unavailable", "release")
    root.mkdir(parents=True, exist_ok=False)
    for row, document, source in rows:
        directory = root / row["id"]
        directory.mkdir()
        write_json(directory / "schedule.json", document)
        write_bytes(directory / "kernel.cu", source.encode())
        if "baseline" in row:
            baseline_document = baseline_schedule(row)
            baseline = compiler.lower(compiler.assess(baseline_document))
            write_json(directory / "baseline-schedule.json", baseline_document)
            write_bytes(directory / "baseline.py", baseline.source.encode())
    write_json(root / "manifest.json", {
        "schema_version": 1, "kind": "native_cuda_qualification",
        "source_commit": commit["stdout"].strip(), "target": "sm_103a",
        "baseline": "existing_triton_algorithm", "timing_scope": "prepared_inputs_core_launch",
        "rows": [row for row, _, _ in rows],
    })
    return {"prepared_cases": len(rows), "gpu_qualification": "pending"}


def load_manifest(root):
    manifest = read_json(child(root, "manifest.json"))
    require(manifest.get("schema_version") == 1 and manifest.get("kind") == "native_cuda_qualification"
            and manifest.get("target") == "sm_103a"
            and manifest.get("baseline") == "existing_triton_algorithm"
            and manifest.get("timing_scope") == "prepared_inputs_core_launch", "qualification manifest differs")
    current = capture(["git", "rev-parse", "HEAD"], cwd=ROOT)
    require(current["returncode"] == 0 and manifest.get("source_commit") == current["stdout"].strip(),
            "qualification is bound to a different source commit", "release")
    # Replaying the public Compiler at this external handoff binds source, ABI,
    # target, and released Revision without introducing another source registry.
    compiler = revision()
    expected = lower_rows(compiler)
    require(manifest.get("rows") == [r for r, _, _ in expected], "manifest differs from released public lowering")
    for row, document, source in expected:
        require(read_json(child(root, row["id"] + "/schedule.json")) == document,
                "retained Schedule differs from workload specialization")
        require(child(root, row["id"] + "/kernel.cu").read_text() == source,
                "retained source differs from released lowering")
        if "baseline" in row:
            baseline_document = baseline_schedule(row)
            baseline = compiler.lower(compiler.assess(baseline_document))
            require(read_json(child(root, row["id"] + "/baseline-schedule.json")) == baseline_document
                    and child(root, row["id"] + "/baseline.py").read_text() == baseline.source,
                    "baseline differs from released lowering")
    return manifest


def compile_all(root, nvcc, cuobjdump):
    manifest = load_manifest(root)
    require(not os.environ.get("GPUQ_JOB_ID"), "compilation must use a CPU-only stage", "admission")
    for row in manifest["rows"]:
        directory = root / row["id"]
        require(not any((directory / name).exists() for name in (
            "compile.json", "kernel.so", "kernel.ptx", "kernel.cubin")), "compile output already exists")
        record = {"source_sha256": row["source_sha256"], "target": row["target"],
                  "commands": {}, "nvcc_version": capture([nvcc, "--version"], cwd=directory)}
        try:
            for role, argv in compile_commands(nvcc, Path("."), row["metadata"]).items():
                record["commands"][role] = capture(argv, cwd=directory)
                require(record["commands"][role]["returncode"] == 0, f"nvcc {role} failed", "compile")
            record["resources_raw"] = capture([cuobjdump, "--dump-resource-usage", "kernel.cubin"], cwd=directory)
            require(record["resources_raw"]["returncode"] == 0, "resource inspection failed", "compile")
            record["inspector_version"] = capture([cuobjdump, "--version"], cwd=directory)
            require(record["inspector_version"]["returncode"] == 0, "resource inspector version unavailable", "compile")
            # The existing typed owner binds the inspected CUBIN to these physical
            # allocation facts. The separately loaded host wrapper has its own
            # narrow handoff identity below; neither file substitutes for the other.
            record["compiled_resources"] = CompiledResources(
                source_sha256=row["source_sha256"],
                cubin_sha256=sha256((directory / "kernel.cubin").read_bytes()).hexdigest(),
                target=row["target"], entry_point=row["metadata"]["kernel_entry_point"],
                threads_per_cta=row["metadata"]["threads_per_cta"],
                dynamic_shared_bytes=row["metadata"]["dynamic_shared_bytes"],
                compiler_version=record["nvcc_version"]["stdout"],
                inspector_version=record["inspector_version"]["stdout"],
                **_parse_cuobjdump_resources(record["resources_raw"]["stdout"], row["metadata"]["kernel_entry_point"]),
            ).as_dict()
            # The host wrapper is the actual loadable artifact. This one identity
            # relation is needed to prevent swapping it after CPU compilation.
            record["shared_object_sha256"] = sha256((directory / "kernel.so").read_bytes()).hexdigest()
            if "baseline" in row:
                from open_cake_ir.compiler.toolchain import compile_triton
                baseline = compile_triton((directory / "baseline.py").read_bytes(), row["baseline"]["metadata"])
                for role, payload in baseline.artifacts.items():
                    write_bytes(directory / ("baseline." + role), payload)
                record["baseline"] = {"source_sha256": row["baseline"]["source_sha256"],
                    "target": baseline.target, "kernel_entry_point": baseline.entry_point,
                    "threads_per_cta": baseline.threads_per_cta,
                    "dynamic_shared_bytes": baseline.dynamic_shared_bytes,
                    "compiler_version": baseline.compiler_version,
                    "cubin_sha256": sha256(baseline.artifacts["cubin"]).hexdigest()}
            record["status"] = "compiled"
        except (ValueError, OSError) as error:
            record["status"] = "failed"
            record["error"] = str(error)
            raise
        finally:
            write_json(directory / "compile.json", record)
    return {"compiled_cases": len(manifest["rows"]), "gpu_qualification": "pending"}


def verify_compile(root, row):
    directory = root / row["id"]
    record = read_json(child(directory, "compile.json"))
    require(record.get("status") == "compiled" and record.get("target") == row["target"]
            and record.get("source_sha256") == row["source_sha256"], "compile binding differs", "compile")
    commands = record.get("commands", {})
    require(set(commands) == {"shared", "ptx", "cubin"}, "compile receipt incomplete", "compile")
    nvcc = record.get("nvcc_version", {}).get("argv", [None])[0]
    require(isinstance(nvcc, str) and Path(nvcc).is_absolute()
            and record["nvcc_version"].get("returncode") == 0
            and "release " in record["nvcc_version"].get("stdout", ""), "nvcc version unavailable", "compile")
    for role, argv in compile_commands(nvcc, Path("."), row["metadata"]).items():
        require(commands[role].get("argv") == argv and commands[role].get("returncode") == 0,
                "compile command or result differs", "compile")
    raw = record.get("resources_raw", {})
    require(raw.get("returncode") == 0 and raw.get("argv", [])[1:] == ["--dump-resource-usage", "kernel.cubin"],
            "compiled resource observation missing", "compile")
    resources = _parse_cuobjdump_resources(raw.get("stdout", ""), row["metadata"]["kernel_entry_point"])
    compiled = CompiledResources.from_dict(record.get("compiled_resources"))
    inspector = record.get("inspector_version", {})
    require(inspector.get("returncode") == 0 and inspector.get("argv") == [raw["argv"][0], "--version"]
            and compiled.inspector_version == inspector.get("stdout")
            and compiled.compiler_version == record["nvcc_version"]["stdout"], "compiled resource toolchain differs", "compile")
    require(all(getattr(compiled, name) == value for name, value in resources.items())
            and compiled.source_sha256 == row["source_sha256"] and compiled.target == row["target"]
            and compiled.entry_point == row["metadata"]["kernel_entry_point"]
            and compiled.threads_per_cta == row["metadata"]["threads_per_cta"]
            and compiled.dynamic_shared_bytes == row["metadata"]["dynamic_shared_bytes"],
            "compiled resource source, target, launch or raw projection differs", "compile")
    shared = child(directory, "kernel.so").read_bytes()
    require(shared.startswith(b"\x7fELF") and sha256(shared).hexdigest() == record.get("shared_object_sha256"),
            "loadable artifact differs from compile handoff", "compile")
    cubin = child(directory, "kernel.cubin").read_bytes()
    require(cubin.startswith(b"\x7fELF") and sha256(cubin).hexdigest() == compiled.cubin_sha256,
            "inspected native CUBIN differs from resource handoff", "compile")
    ptx = child(directory, "kernel.ptx").read_text()
    require(".target sm_103a" in ptx, "PTX exact target differs", "target")
    if "baseline" in row:
        baseline = record.get("baseline", {})
        require(baseline.get("target") == row["target"]
                and baseline.get("source_sha256") == row["baseline"]["source_sha256"]
                and baseline.get("kernel_entry_point") == row["baseline"]["metadata"]["kernel_entry_point"]
                and isinstance(baseline.get("threads_per_cta"), int) and baseline["threads_per_cta"] > 0
                and isinstance(baseline.get("dynamic_shared_bytes"), int) and baseline["dynamic_shared_bytes"] >= 0
                and baseline.get("compiler_version"), "baseline compile binding incomplete", "compile")
        cubin = child(directory, "baseline.cubin").read_bytes()
        require(cubin.startswith(b"\x7fELF") and sha256(cubin).hexdigest() == baseline.get("cubin_sha256")
                and ".target sm_103a" in child(directory, "baseline.ptx").read_text(), "baseline compiled artifacts differ", "compile")
    return record


class NativeHandle:
    """Thin native.so adapter: create only configures the emitted descriptors."""
    def __init__(self, path, metadata, arguments):
        from open_cake_ir.evaluation.cuda_driver import _tensor_contract
        from types import SimpleNamespace
        contract = SimpleNamespace(tensors=tuple((a["name"], tuple(a["shape"]),
            "torch." + {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32", "int32": "int32"}[a["dtype"]])
            for a in metadata["arguments"]))
        _, pointers = _tensor_contract(arguments, contract)
        self.arguments = arguments
        self.library = ctypes.CDLL(str(path))
        self.handle = ctypes.c_void_p()
        create = getattr(self.library, metadata["host_abi"]["create"])
        create.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
        create.restype = ctypes.c_int
        self._launch = getattr(self.library, metadata["host_abi"]["launch"])
        self._launch.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self._launch.restype = ctypes.c_int
        self._destroy = getattr(self.library, metadata["host_abi"]["destroy"])
        self._destroy.argtypes = [ctypes.c_void_p]
        self._destroy.restype = ctypes.c_int
        code = create((ctypes.c_void_p * len(pointers))(*pointers), ctypes.byref(self.handle))
        require(code == 0 and self.handle.value, f"native create failed: {code}", "launch")
        self.calls = 0

    def launch(self, stream):
        code = self._launch(self.handle, ctypes.c_void_p(stream))
        require(code == 0, f"native launch failed: {code}", "launch")
        self.calls += 1

    def close(self):
        if self.handle.value:
            code = self._destroy(self.handle)
            self.handle = ctypes.c_void_p()
            require(code == 0, f"native destroy failed: {code}", "launch")


class BaselineHandle:
    """Load the compiled Triton CUBIN using the existing host-owned Driver path."""
    def __init__(self, directory, row, arguments):
        from open_cake_ir.evaluation.core import LaunchableCandidate, TensorLaunchManifest
        from open_cake_ir.evaluation.cuda_driver import CudaDeviceAdmission, LoadedCudaCandidate
        record = read_json(directory / "compile.json")["baseline"]
        observed = read_json(directory.parent / "admission.json")
        admission = CudaDeviceAdmission(observed["device_name"], tuple(observed["compute_capability"]),
            observed["gpu_uuid"], observed["broker_job_id"], observed["mode"])
        # Flash's historical Workload has no semantics.target. The explicit derived
        # baseline Target belongs to this run; tensor semantics still come from its
        # existing tensor_abi. Do not alter the frozen Workload to satisfy a helper.
        self.manifest = TensorLaunchManifest.from_dict({"schema_version": 1,
            "abi": "workload_tensors_v1", "workload_sha256": row["workload_sha256"], "case_id": row["case_id"],
            "tensor_abi": row["metadata"]["arguments"], "target": row["target"],
            "kernel_name": record["kernel_entry_point"], "grid": row["baseline"]["metadata"]["grid"],
            "block": [record["threads_per_cta"], 1, 1],
            "dynamic_shared_memory_bytes": record["dynamic_shared_bytes"], "hidden_null_pointer_parameters": 2})
        candidate = LaunchableCandidate(row["baseline"]["source_sha256"], row["target"], self.manifest.kernel_name,
            {"cubin": record["cubin_sha256"]}, self.manifest.canonical_sha256)
        self.loaded = LoadedCudaCandidate.load(candidate, (directory / "baseline.cubin").read_bytes(), self.manifest, admission)
        self.arguments = arguments

    def launch(self, stream):
        self.loaded.launch(self.arguments, tensor_contract=self.manifest, stream=stream)

    def close(self):
        import torch
        self.loaded.close(synchronize=torch.cuda.synchronize)


def handle(row, directory, arguments, arm):
    return (NativeHandle(directory / "kernel.so", row["metadata"], arguments)
            if arm == "candidate" else BaselineHandle(directory, row, arguments))


def make_inputs(row, directory):
    import torch
    workload = owner(row)
    preparation = {"scope": "none", "wall_seconds": 0.0, "cupti_samples_ms": None}
    if row["family"] == "kmeans":
        from open_cake_ir.tasks.flash_kmeans.workload import (
            generate_flash_kmeans_case, flash_kmeans_oracle, tensor_raw_sha256, assignment_raw_sha256,
        )
        materialized = workload.case(row["case_id"]).get("materialized", {})
        tokens, centroids = generate_flash_kmeans_case(workload, row["case_id"], device="cuda:0")
        expected = flash_kmeans_oracle(workload, tokens, centroids, case_id=row["case_id"])
        torch.cuda.synchronize()
        start = time.perf_counter()
        centroids_fp32 = centroids.to(torch.float32)
        centroid_sq = (centroids_fp32 * centroids_fp32).sum(dim=-1, dtype=torch.float32).contiguous()
        torch.cuda.synchronize()
        preparation = {"scope": "centroids_to_fp32_square_sum", "wall_seconds": time.perf_counter() - start,
                       "cupti_samples_ms": None}
        inputs = dict(tokens=tokens, centroids=centroids, centroid_sq=centroid_sq)
        # Existing Workload materialization identities are checked once at input
        # handoff. Cases without pinned materialization still retain the real bytes.
        for name, tensor in (("tokens", tokens), ("centroids", centroids), ("oracle_assignments", expected)):
            if name in materialized:
                digest, size = (assignment_raw_sha256(tensor) if name == "oracle_assignments" else tensor_raw_sha256(tensor))
                require(materialized[name] == {"sha256": digest, "size_bytes": size},
                        f"Workload materialization differs: {name}", "input_contract")
    else:
        values = materialize_case(workload, row["case_id"])
        reference = gemm_reference(row, workload, values)
        inputs = {a.name: torch.tensor(values[a.name], dtype={"bf16": torch.bfloat16, "fp32": torch.float32}[a.dtype],
                  device="cuda:0").reshape(a.shape) for a in workload.tensor_abi(row["case_id"]) if a.mode == "input"}
        expected = torch.tensor(reference["c"], dtype=torch.float32, device="cuda:0")
    input_handoff = {}
    for name, tensor in inputs.items():
        payload = tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()
        write_bytes(directory / (name + ".bin"), payload)
        if row["family"] == "kmeans":
            # Snapshot the actual task-generated inputs and actual GPU-prepared
            # core input before either participant can access them. Pinned vectors
            # reuse their already checked Workload identity at this same handoff.
            input_handoff[name] = materialized.get(name) or {
                "sha256": sha256(payload).hexdigest(), "size_bytes": len(payload)}
    expected_values = expected.cpu().reshape(-1).tolist()
    output, = (a for a in workload.tensor_abi(row["case_id"]) if a.mode == "output")
    write_bytes(directory / "oracle.bin", vector_bytes(expected_values, output.dtype))
    write_json(directory / "inputs.json", {"workload_sha256": workload.canonical_sha256,
        "case": workload.case(row["case_id"]), "preparation": preparation,
        "oracle": workload.document["oracle"], "two_mma_composition": row["family"] == "two_mma",
        **({"materialization_device": "cuda:0", "oracle_device": "cuda:0",
            "input_handoff": {name: input_handoff[name] for name in ("tokens", "centroids")},
            "preparation_handoff": {"input": "centroids", "operation": "fp32_square_sum",
                                    "output": input_handoff["centroid_sq"]}}
           if row["family"] == "kmeans" else {})})
    return inputs, expected_values


def fresh_arguments(row, inputs):
    import torch
    return [inputs[a["name"]] if a["mode"] == "input" else torch.full(a["shape"],
        -(2**31) if a["dtype"] == "int32" else float("nan"), dtype=(torch.int32 if a["dtype"] == "int32" else torch.float32),
        device="cuda:0") for a in row["metadata"]["arguments"]]


def unchanged_inputs(directory, inputs):
    for name, tensor in inputs.items():
        before = (directory / (name + ".bin")).read_bytes()
        after = tensor.contiguous().view(__import__("torch").uint8).cpu().numpy().tobytes()
        if before != after:
            return False
    return True


def retain_check(row, directory, label, arguments, expected, inputs, *, unchanged=None):
    import torch
    workload = owner(row)
    result, = (a for a in workload.tensor_abi(row["case_id"]) if a.mode == "output")
    index = row["metadata"]["argument_order"].index(result.name)
    values = arguments[index].cpu().reshape(-1).tolist()
    write_bytes(directory / (label + ".bin"), vector_bytes(values, result.dtype))
    check = compare_raw(row, workload, values, expected)
    if unchanged is None:
        unchanged = unchanged_inputs(directory, inputs)
    check.update(inputs_unchanged=unchanged, output=label + ".bin")
    check["passed"] = check["passed"] and unchanged
    if row["family"] == "kmeans" and not label.startswith("timed-"):
        from open_cake_ir.tasks.flash_kmeans.workload import flash_kmeans_metrics
        reference = torch.tensor(expected, dtype=torch.int32, device="cuda:0").reshape(result.shape)
        try:
            check["diagnostics"] = flash_kmeans_metrics(workload, inputs["tokens"], inputs["centroids"],
                arguments[index], reference, case_id=row["case_id"])
        except ValueError as error:
            check["diagnostics"] = {"error": str(error)}
    return check


def one_launch(row, directory, inputs, expected, label, arm="candidate"):
    import torch
    arguments = fresh_arguments(row, inputs)
    loaded = handle(row, directory.parent, arguments, arm)
    try:
        loaded.launch(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        return retain_check(row, directory, label, arguments, expected, inputs)
    finally:
        loaded.close()


def measure(row, directory, inputs, expected):
    import torch
    import flashinfer.testing as helper
    cupti = StrictCuptiBenchmark(helper)
    protocol = timing_protocol()
    stream = torch.cuda.current_stream().cuda_stream
    measurements = []
    for pair_index, order in enumerate(protocol.pair_order):
        pair = {"pair_index": pair_index, "order": list(order), "arms": {}}
        for position, arm in enumerate(order):
            pool = []
            used = 0
            failure = None
            teardown_interrupt = None
            label = f"cohort-{pair_index}-{arm}"
            cohort = {"position": position, "source_sha256": row["source_sha256"] if arm == "candidate"
                      else row["baseline"]["source_sha256"], "checks": [], "status": "unknown"}
            try:
                for _ in range(protocol.route_calls_per_cohort):
                    arguments = fresh_arguments(row, inputs)
                    pool.append(handle(row, directory.parent, arguments, arm))
                torch.cuda.synchronize()
                def launch_fresh():
                    nonlocal used
                    require(used < len(pool), "CUPTI invocation budget exceeded", "timing")
                    pool[used].launch(stream)
                    used += 1
                returned = cupti(launch_fresh, dry_run_iters=CUPTI_DRY_RUN_ITERS,
                    repeat_iters=protocol.samples_per_cohort, cold_l2_cache=True, use_cuda_graph=False)
                # Preserve every value that the helper returned before any further
                # synchronization, count, quality or correctness check can fail.
                samples = []
                for value in returned:
                    try:
                        number = float(value)
                        samples.append(number if math.isfinite(number) else repr(number))
                    except (TypeError, ValueError):
                        samples.append(repr(value))
                cohort.update(samples_ms=samples, route_calls=used, samples_receipt=label + "-samples.json")
                write_json(directory / cohort["samples_receipt"], {key: cohort[key] for key in (
                    "position", "source_sha256", "samples_ms", "route_calls")})
                torch.cuda.synchronize()
                require(used == len(pool), "CUPTI invocation count differs", "timing")
                require(len(samples) == protocol.samples_per_cohort, "CUPTI returned sample count differs", "timing")
                cohort["summary"] = summarize_cohort(samples)
                unchanged = unchanged_inputs(directory, inputs)
                for i, loaded in enumerate(pool):
                    cohort["checks"].append(retain_check(row, directory, f"timed-{pair_index}-{arm}-{i}",
                        loaded.arguments, expected, inputs, unchanged=unchanged))
                require(all(check["passed"] for check in cohort["checks"]), "timed output failed correctness", "correctness")
                cohort["status"] = "valid"
            except Exception as error:
                failure = error
                cohort.update(status="invalid" if getattr(error, "category", None) == "correctness" else "unknown",
                    failure_class=getattr(error, "category", "runtime_unknown"), error=str(error))
            finally:
                cohort.setdefault("route_calls", used)
                teardown_errors = []
                for index, loaded in enumerate(pool):
                    try:
                        loaded.close()
                    except BaseException as error:
                        teardown_errors.append(str(error))
                        if not isinstance(error, Exception):
                            # Stop this cleanup pass on cancellation. Preserve all
                            # acquired observations before propagating the exact
                            # interruption, even when an earlier check also failed.
                            teardown_interrupt = error
                            cohort["teardown_interrupt"] = {"type": type(error).__name__, "message": str(error),
                                "unattempted_handles": len(pool) - index - 1}
                            break
                        if failure is None:
                            failure = error
                if teardown_errors:
                    cohort["teardown_errors"] = teardown_errors
                    if cohort["status"] == "valid":
                        cohort.update(status="unknown", failure_class="teardown_unknown")
                write_json(directory / (label + ".json"), cohort)
            if teardown_interrupt is not None:
                raise teardown_interrupt from failure
            if failure is not None:
                raise failure
            pair["arms"][arm] = cohort
        measurements.append(pair)
        write_json(directory / f"pair-{pair_index}.json", pair)
    return {"method": "cupti", "cold_l2_cache": True, "use_cuda_graph": False,
            "baseline": "existing_triton_algorithm", "scope": "prepared_inputs_core_launch",
            "measurements": measurements, "summary": timing_summary(measurements)}


def load_inputs(row, directory):
    import torch
    inputs = {}
    for value in row["metadata"]["arguments"]:
        if value["mode"] == "input":
            dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[value["dtype"]]
            raw = bytearray(child(directory, value["name"] + ".bin").read_bytes())
            inputs[value["name"]] = torch.frombuffer(raw, dtype=dtype).clone().reshape(value["shape"]).to("cuda:0")
    result, = (a for a in owner(row).tensor_abi(row["case_id"]) if a.mode == "output")
    return inputs, read_vector(child(directory, "oracle.bin"), result.dtype)


def profile_child(root, case_id, arm):
    # This child inherits its parent's broker allocation. Admission was observed
    # before the parent initialized CUDA; it must not run the clean-card probe again.
    manifest = load_manifest(root)
    row, = (r for r in manifest["rows"] if r["id"] == case_id)
    verify_compile(root, row)
    directory = root / case_id / "run"
    require(arm == "candidate" or "baseline" in row, "structural case has no baseline")
    label = "profile-" + arm
    require(not (directory / (label + "-output.json")).exists(), "profile output already exists")
    request = read_json(child(directory, label + "-request.json"))
    admission = read_json(child(root, "admission.json"))
    execution = {**profile_context(), "status": "failed"}
    try:
        validate_profile_request(request, admission, row, arm)
        require(request["evidence_root"] == str(root), "profile request root differs", "profiler")
        validate_profile_context(request, execution)
        import torch
        require(torch.cuda.device_count() == 1, "profile child device count differs", "admission")
        execution["gpu"] = {"gpu_uuid": str(getattr(torch.cuda.get_device_properties(0), "uuid", "")),
            "device_name": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0))}
        require(execution["gpu"] == request["expected"]["gpu"],
                "profile child GPU differs from parent allocation", "admission")
        inputs, expected = load_inputs(row, directory)
        check = one_launch(row, directory, inputs, expected, label + "-output", arm)
        write_json(directory / (label + "-output.json"), check)
        require(check["passed"], "profile output failed correctness", "correctness")
        execution["output_access"] = profile_output_access(directory, label)
        execution["status"] = "observed"
        return check
    except BaseException as error:
        execution.update(error=str(error), failure_class=getattr(error, "category", "runtime_unknown"),
                         exception_type=type(error).__name__)
        raise
    finally:
        write_json(directory / (label + "-execution.json"), execution)


def profile(root, row, directory, ncu, arm="candidate", *, sudo=False):
    admission = read_json(child(root, "admission.json"))
    expected = profile_expected(admission, sudo)
    context = profile_context()
    require(all(context[key] == admission[key] for key in ("uid", "euid", "pid"))
            and context["environment"] == admission["profile_environment"],
            "profile launcher is not the admitted ordinary parent", "admission")
    if sudo:
        require(expected["environment"]["PYTHONDONTWRITEBYTECODE"] == "1"
                and all(isinstance(expected["environment"][name], str)
                    and Path(expected["environment"][name]).is_dir()
                    and root.resolve() in Path(expected["environment"][name]).resolve().parents
                    for name in ("TMPDIR", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH")),
                "sudo profiling requires existing task-private temporary/cache paths", "profiler")
    ncu = str(Path(ncu).resolve(strict=True))
    executable = Path(ncu).read_bytes()
    require(executable.startswith(b"\x7fELF"),
            "--ncu must name the real NCU ELF executable; shell launchers are refused", "profiler")
    label = "profile-" + arm
    version = capture([ncu, "--version"], cwd=directory, timeout=30)
    write_json(directory / (label + "-version.json"), version)
    require(version["returncode"] == 0, "NCU version unavailable", "profiler")
    # These identities are mandatory at the existing NCU profile handoff boundary.
    executable_id = sha256(executable).hexdigest()
    source = row if arm == "candidate" else row["baseline"]
    entry = source["metadata"]["kernel_entry_point"]
    request = {"case_id": row["id"], "arm": arm, "source_sha256": source["source_sha256"],
        "evaluation_pid": admission["pid"], "sudo": sudo, "expected": expected,
        "ncu": ncu, "ncu_executable_sha256": executable_id, "version_probe": version,
        "python": sys.executable, "script": str(Path(__file__).resolve()), "evidence_root": str(root)}
    validate_profile_request(request, admission, row, arm)
    write_json(directory / (label + "-request.json"), request)
    raw = capture_profile(profile_command(request, row), cwd=directory,
                          receipt=directory / (label + "-command.json"))
    require(raw["returncode"] == 0 and "error" not in raw,
            "NCU execution failed; raw observer failure retained", "profiler")
    execution = read_json(child(directory, label + "-execution.json"))
    validate_profile_execution(request, execution)
    require(profile_output_access(directory, label) == execution.get("output_access"),
            "profile output access changed during child-to-parent handoff", "profiler")
    # Readability is observed here. Never chmod/chown a root-created artifact to
    # make this handoff pass; ownership/modes in the execution record are original.
    read_json(child(directory, label + "-output.json"))
    child(directory, label + "-output.bin").read_bytes()
    payload = build_ncu_attribution_profile(candidate_sha256=source["source_sha256"], case_id=row["id"],
        kernel_name=entry, ncu_version=version["stdout"], ncu_executable_sha256=executable_id,
        stdout=raw["stdout"].encode(), stderr=raw["stderr"].encode())
    write_bytes(directory / (label + ".json"), payload)


def run_all(root, ncu, *, ncu_sudo=False):
    manifest = load_manifest(root)
    for row in manifest["rows"]:
        verify_compile(root, row)
        require(not (root / row["id"] / "run").exists(), "run already exists; retain failure and use a new evidence root")
    from open_cake_ir.evaluation.admission import observe_exclusive_cuda
    try:
        require(os.getuid() > 0 and os.getuid() == os.geteuid(),
                "the evaluation parent and CUPTI must run as an ordinary user", "admission")
        admission = observe_exclusive_cuda(manifest["target"])
    except Exception as error:
        write_json(root / "admission-failure.json", {"failure_class": "admission_unknown", "error": str(error)})
        raise
    import torch
    write_json(root / "admission.json", {**asdict(admission), "target": manifest["target"],
        "hostname": platform.node(), "pid": os.getpid(), "parent_pid": os.getppid(),
        "uid": os.getuid(), "euid": os.geteuid(), "profile_environment": profile_environment(),
        "profile_python": sys.executable, "profile_script": str(Path(__file__).resolve()),
        "evidence_root": str(root),
        "visible_device": os.environ["CUDA_VISIBLE_DEVICES"],
        "python": sys.version, "torch": torch.__version__,
        "flashinfer": importlib.metadata.version("flashinfer-python")})
    for row in manifest["rows"]:
        directory = root / row["id"] / "run"
        directory.mkdir()
        result = {"source_sha256": row["source_sha256"], "workload_sha256": row["workload_sha256"],
                  "target": row["target"], "case_id": row["id"],
                  "gpu_uuid": admission.gpu_uuid, "broker_job_id": admission.broker_job_id,
                  "status": "failed", "failure_class": None}
        try:
            inputs, expected = make_inputs(row, directory)
            arms = ("candidate", "baseline") if "baseline" in row else ("candidate",)
            for arm in arms:
                result["preflight-" + arm] = one_launch(row, directory, inputs, expected, "preflight-" + arm, arm)
                require(result["preflight-" + arm]["passed"], f"{arm} preflight correctness failed", "correctness")
            if "baseline" in row:
                timing = measure(row, directory, inputs, expected)
                write_json(directory / "timing.json", timing)
                require(timing["summary"]["measurement_quality_passed"], "paired timing quality failed", "timing")
            for arm in arms:
                result["postflight-" + arm] = one_launch(row, directory, inputs, expected, "postflight-" + arm, arm)
                require(result["postflight-" + arm]["passed"], f"{arm} postflight correctness failed", "correctness")
                profile(root, row, directory, ncu, arm, sudo=ncu_sudo)
            result["status"] = "observed"
        except Exception as error:
            result["error"] = str(error)
            result["failure_class"] = getattr(error, "category", "runtime_unknown")
            raise
        finally:
            write_json(directory / "result.json", result)
    return {"observed_cases": len(manifest["rows"]), "next": "verify"}


def verify_check(row, directory, record, expected):
    require(isinstance(record, dict), "correctness observation missing", "correctness")
    workload = owner(row)
    result, = (a for a in workload.tensor_abi(row["case_id"]) if a.mode == "output")
    observed = read_vector(child(directory, record.get("output")), result.dtype)
    recomputed = compare_raw(row, workload, observed, expected)
    public = {key: value for key, value in record.items() if key != "diagnostics"}
    require(public == {**recomputed, "inputs_unchanged": True, "output": record["output"]}
            and recomputed["passed"], "correctness projection differs or failed", "correctness")
    if row["family"] == "kmeans" and not record["output"].startswith("timed-"):
        diagnostics = record.get("diagnostics", {})
        require(diagnostics.get("exact_match") is True and diagnostics.get("mismatch_count") == 0
                and diagnostics.get("total_assignments") == len(expected),
                "KMeans diagnostics missing or differ from exact IDs", "correctness")


def verify_centroid_preparation(centroids, centroid_sq, feature_size):
    """Check numerical FP32 square/sum consistency, not a CUDA tree replay.

    Integers encode multiples of 2**-149. Nonzero BF16 squares must be exact
    normal FP32 products. The nonnegative sum must have gamma_(D-1) overflow
    headroom; unsupported domains are refused. Sums in the exact common binary
    quantum region admit no error. Else the bound covers round-to-nearest FP32
    addition orders, without claiming every accepted value is tree-realizable.
    """
    precision = 1 << 24
    maximum = (precision - 1) << 253  # maximum finite FP32, in 2**-149 units
    require(type(feature_size) is int and 0 < feature_size <= precision,
            "centroid preparation term count is outside the FP32 model", "preparation_model_coverage")
    steps = feature_size - 1
    require(len(centroids) > 0 and len(centroids) % (2 * feature_size) == 0
            and len(centroid_sq) == len(centroids) // (2 * feature_size) * 4,
            "centroid preparation BF16/FP32 byte shapes differ", "input_contract")
    values = struct.iter_unpack("<H", centroids)
    for row_index, (norm_bits,) in enumerate(struct.iter_unpack("<I", centroid_sq)):
        magnitude = norm_bits & 0x7fffffff
        require(magnitude < 0x7f800000 and (not norm_bits >> 31 or magnitude == 0),
                f"centroid_sq row {row_index} must be finite and nonnegative", "input_contract")
        exponent, fraction = (magnitude >> 23), magnitude & 0x7fffff
        observed = fraction if exponent == 0 else ((1 << 23) + fraction) << (exponent - 1)
        total, quantum = 0, None
        for feature in range(feature_size):
            bits, = next(values)
            exponent, fraction = (bits >> 7) & 0xff, bits & 0x7f
            if exponent == 0 and fraction == 0:
                continue
            require(exponent != 0xff, f"centroid row {row_index} feature {feature} is nonfinite", "input_contract")
            require(64 <= exponent <= 190,
                    f"centroid row {row_index} feature {feature} square is outside the exact normal FP32 model",
                    "preparation_model_coverage")
            # BF16 = (128+fraction)*2**(exponent-134); its square has <=16
            # significant bits, so the checked normal product is exact in FP32.
            product = (128 + fraction) ** 2 << (2 * exponent - 119)
            total += product
            binary_quantum = product & -product
            quantum = binary_quantum if quantum is None else min(quantum, binary_quantum)
        require(total * precision <= maximum * (precision - steps),
                f"centroid_sq row {row_index} lacks FP32 summation overflow headroom", "preparation_model_coverage")
        exact = total == 0 or total // quantum <= precision
        consistent = observed == total if exact else abs(observed - total) * (precision - steps) <= steps * total
        require(consistent, f"centroid_sq row {row_index} violates FP32 square/sum numerical consistency",
                "preparation_replay_mismatch")


def replay_kmeans_inputs(row, directory, expected, input_receipt):
    """Verify input domain, seed or frozen identity, preparation and external oracle.

    Only already frozen Workload materializations replace oracle computation.
    Newly recorded input identities detect drift after actual GPU generation;
    they do not self-certify an oracle. Unpinned references are independently
    recomputed by the unchanged task-owned oracle on actual retained inputs.
    """
    workload = owner(row)
    case = workload.case(row["case_id"])
    materialized = case.get("materialized", {})
    require(input_receipt.get("materialization_device") == "cuda:0"
            and input_receipt.get("oracle_device") == "cuda:0",
            "KMeans materialization/oracle device differs", "input_contract")
    raw = {}
    for value in workload.tensor_abi(row["case_id"]):
        if value.mode != "input":
            continue
        raw[value.name] = child(directory, value.name + ".bin").read_bytes()
        require(len(raw[value.name]) == math.prod(value.shape) * {"bf16": 2, "fp32": 4}[value.dtype],
                "retained KMeans input shape differs", "input_contract")
        if value.dtype == "bf16":
            finite_bf16(raw[value.name], value.name)
        else:
            norms = read_vector(child(directory, value.name + ".bin"), "fp32")
            require(all(math.isfinite(value) and value >= 0 for value in norms),
                    "prepared centroid norms must be finite and nonnegative", "input_contract")
    identities = {name: {"sha256": sha256(payload).hexdigest(), "size_bytes": len(payload)}
                  for name, payload in raw.items()}
    require(input_receipt.get("input_handoff") == {name: identities[name] for name in ("tokens", "centroids")}
            and input_receipt.get("preparation_handoff") == {"input": "centroids", "operation": "fp32_square_sum",
                                                             "output": identities["centroid_sq"]},
            "KMeans generation/preparation handoff bytes differ", "input_contract")
    for name, identity in materialized.items():
        payload = vector_bytes(expected, "int32") if name == "oracle_assignments" else raw[name]
        observed = {"size_bytes": len(payload), "sha256": sha256(payload).hexdigest()} if name == "oracle_assignments" else identities[name]
        require(observed == identity,
                "Workload pinned input/oracle differs", "input_contract")
    try:
        import torch
    except ImportError as error:
        raise QualificationError("cpu_oracle_unavailable", "KMeans CPU replay requires PyTorch; GPU qualification remains unknown") from error
    from open_cake_ir.tasks.flash_kmeans.workload import flash_kmeans_oracle
    tensors = {value.name: torch.frombuffer(bytearray(raw[value.name]), dtype=(
        torch.bfloat16 if value.dtype == "bf16" else torch.float32)).reshape(value.shape)
        for value in workload.tensor_abi(row["case_id"]) if value.mode == "input"}
    if not materialized:
        oracle = flash_kmeans_oracle(workload, tensors["tokens"], tensors["centroids"], case_id=row["case_id"])
        require(oracle.reshape(-1).tolist() == expected,
                "KMeans GPU oracle differs from the task-owned external CPU replay", "oracle_replay_mismatch")
    verify_centroid_preparation(raw["centroids"], raw["centroid_sq"], case["shape"]["D"])


def verify_all(root):
    manifest = load_manifest(root)
    admission = read_json(child(root, "admission.json"))
    require(admission.get("target") == "sm_103a" and admission.get("compute_capability") == [10, 3]
            and admission.get("mode") == "exclusive" and admission.get("gpu_uuid")
            and admission.get("broker_job_id", "").startswith("gpuq-")
            and admission.get("hostname"), "exact exclusive CUDA admission missing", "admission")
    verified = []
    for row in manifest["rows"]:
        require(admission.get("device_name") in row["metadata"]["target_device_names"], "device name differs", "target")
        verify_compile(root, row)
        directory = root / row["id"] / "run"
        result = read_json(child(directory, "result.json"))
        require(result.get("status") == "observed" and result.get("failure_class") is None
                and result.get("target") == row["target"] and result.get("case_id") == row["id"]
                and result.get("source_sha256") == row["source_sha256"]
                and result.get("workload_sha256") == row["workload_sha256"]
                and result.get("gpu_uuid") == admission["gpu_uuid"]
                and result.get("broker_job_id") == admission["broker_job_id"], "run binding or terminal observation differs")
        workload = owner(row)
        inputs = read_json(child(directory, "inputs.json"))
        require(inputs.get("workload_sha256") == workload.canonical_sha256
                and inputs.get("case") == workload.case(row["case_id"])
                and inputs.get("oracle") == workload.document["oracle"]
                and inputs.get("two_mma_composition") == (row["family"] == "two_mma"), "input/oracle binding differs", "input_contract")
        preparation = inputs.get("preparation", {})
        require(preparation.get("scope") == ("centroids_to_fp32_square_sum" if row["family"] == "kmeans" else "none")
                and isinstance(preparation.get("wall_seconds"), (float, int))
                and math.isfinite(preparation["wall_seconds"]) and preparation["wall_seconds"] >= 0,
                "preparation cost/scope missing")
        for value in row["metadata"]["arguments"]:
            if value["mode"] == "input":
                size = math.prod(value["shape"]) * {"bf16": 2, "fp32": 4}[value["dtype"]]
                require(child(directory, value["name"] + ".bin").stat().st_size == size, "retained input bytes incomplete", "input_contract")
        output, = (a for a in workload.tensor_abi(row["case_id"]) if a.mode == "output")
        expected = read_vector(child(directory, "oracle.bin"), output.dtype)
        if row["family"] != "kmeans":
            source_inputs = materialize_case(workload, row["case_id"])
            external_expected = gemm_reference(row, workload, source_inputs)[output.name]
            require(expected == external_expected, "retained GEMM oracle differs from external math.fsum", "correctness")
            # Verify exact deterministic input bytes, including BF16 rounding.
            for value in workload.tensor_abi(row["case_id"]):
                if value.mode != "input":
                    continue
                raw = vector_bytes(source_inputs[value.name], "fp32")
                if value.dtype == "bf16":
                    raw = b"".join(raw[i + 2:i + 4] for i in range(0, len(raw), 4))
                require(child(directory, value.name + ".bin").read_bytes() == raw, "input bytes differ from Workload generator", "input_contract")
        else:
            replay_kmeans_inputs(row, directory, expected, inputs)
        arms = ("candidate", "baseline") if "baseline" in row else ("candidate",)
        for arm in arms:
            for phase in ("preflight", "postflight"):
                verify_check(row, directory, result.get(phase + "-" + arm), expected)
        if "baseline" in row:
            timing = read_json(child(directory, "timing.json"))
            require(timing.get("method") == "cupti" and timing.get("cold_l2_cache") is True
                    and timing.get("use_cuda_graph") is False and timing.get("baseline") == "existing_triton_algorithm"
                    and timing.get("scope") == "prepared_inputs_core_launch", "CUPTI scope/policy differs", "timing")
            summary = timing_summary(timing.get("measurements"))
            require(summary == timing.get("summary") and summary["measurement_quality_passed"], "timing raw summary/quality differs", "timing")
            seen_outputs = set()
            for pair in timing["measurements"]:
                require(read_json(child(directory, f"pair-{pair['pair_index']}.json")) == pair, "paired raw receipt differs", "timing")
                for name, arm in pair["arms"].items():
                    identity = row["source_sha256"] if name == "candidate" else row["baseline"]["source_sha256"]
                    require(arm.get("source_sha256") == identity
                            and arm.get("status") == "valid"
                            and len(arm.get("checks", [])) == timing_protocol().route_calls_per_cohort,
                            "paired artifact or launch checks differ", "timing")
                    cohort_name = f"cohort-{pair['pair_index']}-{name}"
                    require(read_json(child(directory, cohort_name + ".json")) == arm
                            and arm.get("samples_receipt") == cohort_name + "-samples.json"
                            and read_json(child(directory, arm["samples_receipt"])) == {key: arm[key] for key in (
                                "position", "source_sha256", "samples_ms", "route_calls")},
                            "raw cohort handoff differs", "timing")
                    for check in arm["checks"]:
                        require(check.get("output") not in seen_outputs, "fresh output receipt was reused", "timing")
                        seen_outputs.add(check["output"])
                        verify_check(row, directory, check, expected)
        for arm in arms:
            source = row if arm == "candidate" else row["baseline"]
            label = "profile-" + arm
            request = read_json(child(directory, label + "-request.json"))
            validate_profile_request(request, admission, row, arm)
            require(read_json(child(directory, label + "-version.json")) == request["version_probe"],
                    "profile NCU version observation differs", "profiler")
            execution = read_json(child(directory, label + "-execution.json"))
            validate_profile_execution(request, execution)
            profile_document = load_ncu_attribution_profile(child(directory, label + ".json").read_bytes(),
                expected_candidate_sha256=source["source_sha256"], expected_case_id=row["id"])
            require(profile_document["kernel_name"] == source["metadata"]["kernel_entry_point"], "profile kernel differs", "profiler")
            require(profile_document["tool"] == {"version": request["version_probe"]["stdout"],
                    "executable_sha256": request["ncu_executable_sha256"]},
                    "real NCU executable/version binding differs", "profiler")
            raw = read_json(child(directory, label + "-command.json"))
            require(raw.get("returncode") == 0 and "error" not in raw
                    and profile_document["raw"] == {"stdout": raw.get("stdout"), "stderr": raw.get("stderr")},
                    "profile raw command evidence differs", "profiler")
            argv = raw.get("argv", [])
            require(argv == profile_command(request, row), "NCU command binding differs", "profiler")
            output_check = read_json(child(directory, label + "-output.json"))
            require(output_check.get("output") == label + "-output.bin"
                    and all(execution["output_access"][label + suffix]["size_bytes"]
                            == child(directory, label + suffix).stat().st_size
                            for suffix in ("-output.bin", "-output.json")),
                    "profile output/access binding differs", "profiler")
            verify_check(row, directory, output_check, expected)
        verified.append(row["id"])
    return {"qualified": True, "scope": "native_compile_external_oracle_core_timing_profiler",
            "target": manifest["target"], "cases": verified,
            "baseline": "existing_triton_algorithm", "framework_e2e": "pending",
            "two_mma_scope": "structural_numerical_and_profiler_no_performance_comparison",
            "trust_boundary": "cooperating_agents_retained_raw_observations"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("prepare", "compile", "run", "verify", "_profile"))
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--nvcc", default="/usr/local/cuda-13.1/bin/nvcc")
    parser.add_argument("--cuobjdump", default="/usr/local/cuda-13.1/bin/cuobjdump")
    parser.add_argument("--ncu", default=DEFAULT_NCU, help="real NCU ELF executable; shell launchers are refused")
    parser.add_argument("--ncu-sudo", action="store_true",
        help="explicitly run NCU and its profile target Python through sudo; parent/CUPTI stay ordinary")
    parser.add_argument("--case")
    parser.add_argument("--arm", choices=("candidate", "baseline"), default="candidate")
    args = parser.parse_args(argv)
    try:
        require(not args.ncu_sudo or args.command == "run", "--ncu-sudo is only valid for run", "profiler")
        root = external(args.evidence_root)
        if args.command == "prepare":
            result = prepare(root)
        elif args.command == "compile":
            result = compile_all(root, args.nvcc, args.cuobjdump)
        elif args.command == "run":
            result = run_all(root, args.ncu, ncu_sudo=args.ncu_sudo)
        elif args.command == "_profile":
            result = profile_child(root, args.case, args.arm)
        else:
            result = verify_all(root)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    except Exception as error:
        print(json.dumps({"qualified": False, "failure_class": getattr(error, "category", "evidence_or_runtime_unknown"),
                          "error": str(error)}, sort_keys=True), file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
